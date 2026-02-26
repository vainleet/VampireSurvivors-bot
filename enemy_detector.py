import numpy as np
from ultralytics import YOLO
from config import VIEWPORT_SIZE, CONF_THRESHOLD, CENTER_IGNORE_RADIUS, EXP_IGNORE_RADIUS, PLAYER_CENTER

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