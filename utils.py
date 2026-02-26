import cv2
import numpy as np
import mss

def capture_region(sct, bbox):
    img = np.array(sct.grab(bbox))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

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