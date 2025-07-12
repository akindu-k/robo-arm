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
    (0, 0): 9, (0, 1): 8, (0, 2): 7,
    (1, 0): 6, (1, 1): 5, (1, 2): 4,
    (2, 0): 3, (2, 1): 2, (2, 2): 1
}

# Define color ranges for solid color detection in HSV (example ranges for common colors)
color_ranges = [
    # Red
    (np.array([0, 120, 70]), np.array([10, 255, 255])),
    (np.array([170, 120, 70]), np.array([180, 255, 255])),
    # Blue
    (np.array([100, 120, 70]), np.array([130, 255, 255])),
    # Green
    (np.array([40, 120, 70]), np.array([80, 255, 255])),
    # Yellow
    (np.array([20, 120, 70]), np.array([40, 255, 255])),
    # Orange
    (np.array([10, 120, 70]), np.array([20, 255, 255])),
]

def draw_grid(frame, grid_width, grid_height):
    """Draw a 3x3 grid with labels on the frame."""
    height, width = frame.shape[:2]
    # Draw vertical lines
    for i in range(1, 3):
        cv2.line(frame, (i * grid_width, 0), (i * grid_width, height), (255, 255, 255), 1)
    # Draw horizontal lines
    for i in range(1, 3):
        cv2.line(frame, (0, i * grid_height), (width, i * grid_height), (255, 255, 255), 1)
    # Label grid cells with values
    for (row, col), value in grid_values.items():
        x = col * grid_width + grid_width // 2
        y = row * grid_height + grid_height // 2
        cv2.putText(frame, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def detect_grid():
    """Detect solid color objects and determine grid value."""
    ret, frame = cap.read()
    if not ret:
        return [], None  # Return empty list and None if frame capture fails

    # Convert frame to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    grid_width = width // 3
    grid_height = height // 3
    detected_values = []

    # Create a combined mask for all colors
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter small contours to reduce noise
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            # Relaxed aspect ratio to allow various shapes
            if 0.5 <= aspect_ratio <= 2.0:
                center_x = x + w // 2
                center_y = y + h // 2
                grid_x = center_x // grid_width
                grid_y = center_y // grid_height
                if 0 <= grid_x <= 2 and 0 <= grid_y <= 2:
                    grid_value = grid_values.get((grid_y, grid_x), 0)
                    detected_values.append(grid_value)
                    # Draw contour and grid cell highlight
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    cell_x = grid_x * grid_width
                    cell_y = grid_y * grid_height
                    cv2.rectangle(frame, (cell_x, cell_y), (cell_x + grid_width, cell_y + grid_height), (0, 0, 255), 1)

    return detected_values, frame

def display_gui(frame, values):
    """Handle GUI display: draw grid and show detected values."""
    if frame is None:
        return  # Do not attempt to display if frame is None

    height, width = frame.shape[:2]
    grid_width = width // 3
    grid_height = height // 3

    # Draw grid and labels
    draw_grid(frame, grid_width, grid_height)

    # Display detected value on frame (show minimum of current frame for consistency)
    current_val = min(values) if values else 0
    cv2.putText(frame, f"Detected Value: {current_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Grid Detection", frame)

try:
    if not os.access(script_dir, os.W_OK):
        print(f"Error: No write permission in directory {script_dir}")
        exit(1)

    print(f"Attempting to process and write to file: {file_path}")

    start_time = time.time()
    duration = 10  # Run for 10 seconds
    all_values = []  # Array to store all detected values

    # Initialize GUI window
    cv2.namedWindow("Grid Detection", cv2.WINDOW_NORMAL)

    while time.time() - start_time < duration:
        # Detect grid and get frame
        values, frame = detect_grid()
        if frame is None:
            print("Error: Failed to capture frame apart from white background.")
            break

        if values:
            all_values.extend(values)  # Add detected values to array

        # Display GUI
        display_gui(frame, values)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

finally:
    try:
        max_val = max(all_values) if all_values else 0
        with open(file_path, "w") as f:
            f.write(str(max_val))
            f.flush()
        print(f"Maximum value {max_val} written to {file_path}")
    except Exception as e:
        print(f"Error writing final value to {file_path}: {e}")

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(file_path):
        print(f"File created at {file_path} with maximum value: {max_val}")
    else:
        print(f"File was not created at {file_path}")