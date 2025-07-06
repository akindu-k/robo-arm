import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

grid_values = {
    (0, 0): 1, (0, 1): 2, (0, 2): 3,
    (1, 0): 4, (1, 1): 5, (1, 2): 6,
    (2, 0): 7, (2, 1): 8, (2, 2): 9
}

def detect_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    grid_width = width // 3
    grid_height = height // 3
    detected_values = []

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.5 <= aspect_ratio <= 2.0 and area > 100:
                center_x = x + w // 2
                center_y = y + h // 2
                grid_x = center_x // grid_width
                grid_y = center_y // grid_height
                if 0 <= grid_x <= 2 and 0 <= grid_y <= 2:
                    grid_value = grid_values[(grid_y, grid_x)]
                    detected_values.append(grid_value)

    if detected_values:
        return min(detected_values)
    return 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        val = detect_grid(frame)
        with open("grid_value.txt", "w") as f:
            f.write(str(val))
        time.sleep(0.1)  # delay 100ms
finally:
    cap.release()