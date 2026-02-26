import time
import cv2
import mss
from enemy_detector import EnemyDetector
from movement_brain import MovementBrain
from smooth_input import SmoothInput
from visual_debug import VisualDebug
from utils import capture_region
from config import VIEWPORT_SIZE, PLAYER_CENTER

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