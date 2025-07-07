import cv2
import numpy as np
import time
import os

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "grid_value.txt")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Define grid values
grid_values = {
    (0, 0): 1, (0, 1): 2, (0, 2): 3,
    (1, 0): 4, (1, 1): 5, (1, 2): 6,
    (2, 0): 7, (2, 1): 8, (2, 2): 9
}

def detect_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    grid_width = width // 3
    grid_height = height // 3
    detected_values = []

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.8 <= aspect_ratio <= 1.2 and area > 500:
                center_x = x + w // 2
                center_y = y + h // 2
                grid_x = center_x // grid_width
                grid_y = center_y // grid_height
                if 0 <= grid_x <= 2 and 0 <= grid_y <= 2:
                    grid_value = grid_values.get((grid_y, grid_x), 0)
                    detected_values.append(grid_value)

    return min(detected_values) if detected_values else 0

try:
    # Verify the directory is writable
    if not os.access(script_dir, os.W_OK):
        print(f"Error: No write permission in directory {script_dir}")
        exit(1)

    print(f"Attempting to write to file: {file_path}")

    # Record start time
    start_time = time.time()
    duration = 10  # Run for 10 seconds

    last_val = 0  # Store the last detected value
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            continue

        val = detect_grid(frame)
        print(type(frame))
        last_val = val  # Update last detected value

        # Write the value to the file
        try:
            with open(file_path, "w") as f:
                f.write(str(val))
                f.flush()  # Ensure the write is committed
            print(f"Written value {val} to {file_path}")
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")

        time.sleep(0.01)  # Minimal delay to prevent overloading LabVIEW

finally:
    # Ensure the final value is written before exiting
    try:
        with open(file_path, "w") as f:
            f.write(str(last_val))
            f.flush()
        print(f"Final value {last_val} written to {file_path}")
    except Exception as e:
        print(f"Error writing final value to {file_path}: {e}")

    # Cleanup
    cap.release()
    if os.path.exists(file_path):
        print(f"File created at {file_path} with final value: {last_val}")
    else:
        print(f"File was not created at {file_path}")