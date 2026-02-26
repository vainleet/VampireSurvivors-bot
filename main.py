import cv2
import numpy as np
import time
import mss
from pynput.keyboard import Controller
from ultralytics import YOLO

VIEWPORT_SIZE = (960, 640)
CONF_THRESHOLD = 0.45

PLAYER_OFFSET_X = -15
PLAYER_OFFSET_Y = -10
PLAYER_CENTER = (
    VIEWPORT_SIZE[0] // 2 + PLAYER_OFFSET_X,
    VIEWPORT_SIZE[1] // 2 + PLAYER_OFFSET_Y
)

CENTER_IGNORE_RADIUS = 20
EXP_IGNORE_RADIUS = 30

EMERGENCY_RADIUS = 110

MONSTER_FORCE = 500.0
PREDICT_FORCE = 320.0


KITE_MIN_DIST = 115
KITE_MAX_DIST = 185
KITE_TANGENT_FORCE = 380.0

RUNE_ATTRACT_FORCE = 1600.0
LOCAL_THREAT_LIMIT = 0.0012


def capture_region(sct, bbox):
    img = np.array(sct.grab(bbox))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

class EnemyDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.prev_positions = []
        self.enemies = []
        self.exps = []

    def scan(self, frame):
        results = self.model(frame, verbose=False, imgsz=640)[0]

        current_positions = []
        enemies = []
        exps = []

        for box, cls, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):
            conf = float(conf)
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            name = self.model.names[int(cls)].lower()
            dist = np.hypot(cx - PLAYER_CENTER[0], cy - PLAYER_CENTER[1])

            if any(t in name for t in ["monster", "enemy", "bat", "zombie"]):
                if dist > CENTER_IGNORE_RADIUS:
                    enemies.append((cx, cy, conf))
                    current_positions.append((cx, cy))

            elif any(t in name for t in ["exp", "gem", "crystal", "coin", "rune"]):
                if dist > EXP_IGNORE_RADIUS:
                    exps.append((cx, cy, conf))

        self.prev_positions = current_positions
        self.enemies = enemies
        self.exps = exps

    def predict(self, dt):
        predictions = []
        for i, (cx, cy, _) in enumerate(self.enemies):
            if i < len(self.prev_positions):
                px, py = self.prev_positions[i]
                vx = (cx - px) / max(dt, 0.001)
                vy = (cy - py) / max(dt, 0.001)
                predictions.append((int(cx + vx * 0.25),
                                    int(cy + vy * 0.25)))
            else:
                predictions.append((cx, cy))
        return predictions

def detect_fires(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fires = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        fires.append((cx, cy))
    return fires


class MovementBrain:
    def nearest_enemy(self, player, enemies):
        if not enemies:
            return None
        return min(enemies, key=lambda e: np.hypot(e[0] - player[0], e[1] - player[1]))

    def kite_vector(self, player, enemies):
        nearest = self.nearest_enemy(player, enemies)
        if not nearest:
            return 0.0, 0.0

        ex, ey, conf = nearest

        dx = player[0] - ex
        dy = player[1] - ey
        dist = max(1, np.hypot(dx, dy))
        rx = dx / dist
        ry = dy / dist

        fx, fy = 0.0, 0.0

        if dist < KITE_MIN_DIST:
            f = (KITE_MIN_DIST - dist) * 4
            fx += rx * f
            fy += ry * f

        elif dist > KITE_MAX_DIST:
            f = (dist - KITE_MAX_DIST) * 2
            fx -= rx * f
            fy -= ry * f

        tx = -ry
        ty = rx
        fx += tx * KITE_TANGENT_FORCE / (dist + 40)
        fy += ty * KITE_TANGENT_FORCE / (dist + 40)

        return fx, fy

    def local_threat(self, point, enemies):

        threat = 0.0
        px, py = point
        for x, y, conf in enemies:
            d = max(1, np.hypot(x - px, y - py))
            threat += conf / (d * d)
        return threat

    def attractive_vector(self, player, target, base_force=1600.0):
        tx, ty, conf = target
        dx = tx - player[0]
        dy = ty - player[1]
        d = max(1, np.hypot(dx, dy))

        f = base_force * conf / (d + 25)
        return (dx / d * f, dy / d * f)

    def avoid_crowd(self, player, enemies, predicted):

        fx, fy = 0.0, 0.0
        for x, y, conf in enemies:
            dx = player[0] - x
            dy = player[1] - y
            d = max(1, np.hypot(dx, dy))
            f = MONSTER_FORCE * conf / (d + 30)
            fx += dx / d * f
            fy += dy / d * f
        for x, y in predicted:
            dx = player[0] - x
            dy = player[1] - y
            d = max(1, np.hypot(dx, dy))
            f = PREDICT_FORCE / (d + 30)
            fx += dx / d * f
            fy += dy / d * f

        return fx, fy

    def decide(self, player, enemies, exps, predicted, frame):
        fx, fy = 0.0, 0.0

        if enemies:
            kx, ky = self.kite_vector(player, enemies)
            fx += kx
            fy += ky

        attract_candidates = []

        for rune in exps:
            local_th = self.local_threat((rune[0], rune[1]), enemies)
            if local_th < LOCAL_THREAT_LIMIT:
                attract_candidates.append((rune, RUNE_ATTRACT_FORCE))

        fires = detect_fires(frame)
        for fire in fires:
            local_th = self.local_threat(fire, enemies)
            if local_th < LOCAL_THREAT_LIMIT:
                attract_candidates.append(((fire[0], fire[1], 0.9), 1200.0))

        if attract_candidates:
            attract_candidates.sort(
                key=lambda item: np.hypot(item[0][0] - player[0], item[0][1] - player[1])
            )
            target, force = attract_candidates[0]
            ax, ay = self.attractive_vector(player, target, force)
            fx = fx * 0.4 + ax
            fy = fy * 0.4 + ay

        ax, ay = self.avoid_crowd(player, enemies, predicted)
        fx += ax * 0.7
        fy += ay * 0.7

        mag = np.hypot(fx, fy)
        if mag > 0:
            fx /= mag
            fy /= mag

        return fx, fy


class VisualDebug:
    def draw(self, frame, player, enemies, exps, predicted, vec):
        cx, cy = player
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 3)
        for x, y, _ in enemies:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        for x, y, _ in exps:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
        for x, y in predicted:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), 1)
        fires = detect_fires(frame)
        for fx, fy in fires:
            cv2.circle(frame, (int(fx), int(fy)), 8, (0,165,255), 2)
        mvx = int(cx + vec[0] * 60)
        mvy = int(cy + vec[1] * 60)
        cv2.arrowedLine(frame, (cx, cy), (mvx, mvy), (255, 255, 255), 2)
        cv2.putText(frame, f"Enemies: {len(enemies)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return frame

class SmoothInput:
    def __init__(self):
        self.keyboard = Controller()
        self.current = set()
    def update(self, vec):
        wanted = set()
        if vec[0] > 0.3: wanted.add('d')
        elif vec[0] < -0.3: wanted.add('a')
        if vec[1] > 0.3: wanted.add('s')
        elif vec[1] < -0.3: wanted.add('w')
        for k in self.current - wanted:
            self.keyboard.release(k)
        for k in wanted - self.current:
            self.keyboard.press(k)
        self.current = wanted
    def release_all(self):
        for k in self.current: self.keyboard.release(k)
        self.current.clear()

def main():
    detector = EnemyDetector("monster.pt")
    brain = MovementBrain()
    inp = SmoothInput()
    debug = VisualDebug()
    game_area = {"top": 80, "left": 200, "width": 1240, "height": 760}
    sct = mss.mss()

    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision", 960, 640)

    last_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        frame = capture_region(sct, game_area)
        frame = cv2.resize(frame, VIEWPORT_SIZE)

        detector.scan(frame)
        predicted = detector.predict(dt)

        move_vec = brain.decide(
            PLAYER_CENTER,
            detector.enemies,
            detector.exps,
            predicted,
            frame
        )

        inp.update(move_vec)

        frame = debug.draw(
            frame,
            PLAYER_CENTER,
            detector.enemies,
            detector.exps,
            predicted,
            move_vec
        )

        cv2.imshow("Vision", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.03)

    inp.release_all()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()