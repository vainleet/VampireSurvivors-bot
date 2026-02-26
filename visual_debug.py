import cv2
from utils import detect_fires

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