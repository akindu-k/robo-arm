#!/usr/bin/env python3
"""
Combined Ball and Color Detection System
Integrates advanced ball detection with real-time color detection
Uses OpenCV DNN, MediaPipe, and color-based detection methods
Includes 2D grid array overlay for position tracking
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import urllib.request
import os

class CombinedDetector:
    def __init__(self):
        # Initialize MediaPipe Object Detection
        self.mp_objectron = mp.solutions.objectron
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe hands for additional context
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Ball tracking parameters
        self.ball_trail = deque(maxlen=64)
        
        # Grid parameters for 2D array coordinates
        self.grid_rows = 10
        self.grid_cols = 10
        self.show_grid = True
        self.current_frame_shape = (0, 0)
        
        # Detection mode flags
        self.enable_ball_detection = True
        self.enable_color_detection = True
        self.enable_shape_detection = True
        
        # Try to load OpenCV DNN models
        self.opencv_net = None
        self.load_opencv_dnn()
        
        # COCO class names (for OpenCV DNN)
        self.COCO_CLASSES = [
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant", "", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # Ball-related class indices
        self.ball_class_ids = [37]  # sports ball
        
        # Detection parameters
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        
        # Color detection parameters
        self.color_ranges = {
            'red': {
                'lower': np.array([136, 87, 111], np.uint8),
                'upper': np.array([180, 255, 255], np.uint8),
                'color': (0, 0, 255)
            },
            'green': {
                'lower': np.array([25, 52, 72], np.uint8),
                'upper': np.array([102, 255, 255], np.uint8),
                'color': (0, 255, 0)
            },
            'blue': {
                'lower': np.array([94, 80, 2], np.uint8),
                'upper': np.array([120, 255, 255], np.uint8),
                'color': (255, 0, 0)
            },
            'yellow': {
                'lower': np.array([20, 100, 100], np.uint8),
                'upper': np.array([30, 255, 255], np.uint8),
                'color': (0, 255, 255)
            },
            'orange': {
                'lower': np.array([5, 50, 50], np.uint8),
                'upper': np.array([15, 255, 255], np.uint8),
                'color': (0, 165, 255)
            }
        }
        
        # Morphological kernel for color detection
        self.kernel = np.ones((5, 5), "uint8")
        self.min_area = 300
    
    def load_opencv_dnn(self):
        """Load OpenCV DNN model (MobileNet-SSD)"""
        try:
            print("Loading OpenCV DNN model...")
            
            # URLs for pre-trained MobileNet-SSD model
            prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
            
            prototxt_path = "MobileNetSSD_deploy.prototxt"
            model_path = "MobileNetSSD_deploy.caffemodel"
            
            # Download prototxt file if it doesn't exist
            if not os.path.exists(prototxt_path):
                print("Downloading prototxt file...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            if not os.path.exists(model_path):
                print("Note: MobileNet model requires manual download")
                print("Please download MobileNetSSD_deploy.caffemodel manually")
                print("Using alternative detection methods...")
                return
            
            # Load the network
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.opencv_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("OpenCV DNN model loaded successfully!")
            else:
                print("Model files not found, using alternative detection methods")
                
        except Exception as e:
            print(f"Error loading OpenCV DNN model: {e}")
            print("Continuing with alternative detection methods...")
    
    def detect_balls_opencv_dnn(self, frame):
        """Detect balls using OpenCV DNN"""
        if self.opencv_net is None:
            return []
        
        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.opencv_net.setInput(blob)
            detections = self.opencv_net.forward()
            
            detected_balls = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                if class_id in self.ball_class_ids and confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max((x2 - x1), (y2 - y1)) // 2
                    
                    detected_balls.append({
                        'center': (center_x, center_y),
                        'radius': radius,
                        'confidence': confidence,
                        'class_name': self.COCO_CLASSES[class_id],
                        'bbox': (x1, y1, x2, y2),
                        'method': 'OpenCV DNN',
                        'color': None
                    })
            
            return detected_balls
            
        except Exception as e:
            print(f"Error in OpenCV DNN detection: {e}")
            return []
    
    def detect_balls_hough_enhanced(self, frame):
        """Enhanced Hough circle detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred = cv2.GaussianBlur(filtered, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(frame.shape[0] / 8),
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=min(frame.shape[0], frame.shape[1]) // 4
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                roi = cv2.bitwise_and(gray, mask)
                mean_intensity = cv2.mean(roi, mask)[0]
                
                if mean_intensity > 30:
                    detected_balls.append({
                        'center': (x, y),
                        'radius': r,
                        'confidence': 0.6,
                        'class_name': 'circular object',
                        'bbox': (x-r, y-r, x+r, y+r),
                        'method': 'Enhanced Hough',
                        'color': None
                    })
        
        return detected_balls
    
    def detect_colored_objects(self, frame):
        """Detect colored objects using HSV color space"""
        if not self.enable_color_detection:
            return []
        
        # Convert to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects = []
        
        for color_name, color_info in self.color_ranges.items():
            # Create mask for the color
            mask = cv2.inRange(hsv_frame, color_info['lower'], color_info['upper'])
            
            # Apply morphological operations
            mask = cv2.dilate(mask, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Check if it's roughly circular (could be a colored ball)
                    aspect_ratio = w / h if h > 0 else 1
                    is_circular = 0.7 <= aspect_ratio <= 1.3
                    
                    detected_objects.append({
                        'center': (center_x, center_y),
                        'radius': max(w, h) // 2,
                        'confidence': 0.8 if is_circular else 0.5,
                        'class_name': f'{color_name} {"ball" if is_circular else "object"}',
                        'bbox': (x, y, x + w, y + h),
                        'method': 'Color Detection',
                        'color': color_name,
                        'area': area,
                        'is_circular': is_circular
                    })
        
        return detected_objects
    
    def detect_all_objects(self, frame):
        """Main detection function - combines all detection methods"""
        all_detections = []
        
        # Ball detection using DNN
        if self.enable_ball_detection and self.opencv_net is not None:
            balls_dnn = self.detect_balls_opencv_dnn(frame)
            all_detections.extend(balls_dnn)
        
        # Enhanced Hough circle detection
        if self.enable_shape_detection:
            balls_hough = self.detect_balls_hough_enhanced(frame)
            all_detections.extend(balls_hough)
        
        # Color-based detection
        colored_objects = self.detect_colored_objects(frame)
        all_detections.extend(colored_objects)
        
        # Remove duplicate detections (objects detected by multiple methods)
        filtered_detections = self.filter_duplicate_detections(all_detections)
        
        return filtered_detections
    
    def filter_duplicate_detections(self, detections):
        """Remove duplicate detections based on proximity"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        threshold_distance = 50  # pixels
        
        for detection in detections:
            is_duplicate = False
            center1 = detection['center']
            
            for existing in filtered:
                center2 = existing['center']
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < threshold_distance:
                    # Keep the detection with higher confidence
                    if detection['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def draw_grid(self, frame):
        """Draw a grid on the frame"""
        h, w = frame.shape[:2]
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        
        # Draw vertical lines
        for i in range(1, self.grid_cols):
            x = i * cell_w
            cv2.line(frame, (x, 0), (x, h), (70, 70, 70), 1)
            
        # Draw horizontal lines
        for i in range(1, self.grid_rows):
            y = i * cell_h
            cv2.line(frame, (0, y), (w, y), (70, 70, 70), 1)
            
        # Add grid labels
        for i in range(self.grid_cols):
            cv2.putText(frame, str(i), (i * cell_w + 5, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                      
        for i in range(self.grid_rows):
            cv2.putText(frame, str(i), (5, i * cell_h + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def get_grid_position(self, center):
        """Convert pixel coordinates to grid coordinates"""
        if center is None:
            return None
        
        h, w = self.current_frame_shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        
        grid_x = min(center[0] // cell_w, self.grid_cols - 1)
        grid_y = min(center[1] // cell_h, self.grid_rows - 1)
        
        return (int(grid_x), int(grid_y))
    
    def draw_detection_info(self, frame, detections):
        """Draw detection information on the frame with grid coordinates"""
        self.current_frame_shape = frame.shape[:2]
        
        # Draw grid if enabled
        if self.show_grid:
            frame = self.draw_grid(frame)
        
        for detection in detections:
            center = detection['center']
            radius = int(detection['radius'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            bbox = detection['bbox']
            method = detection.get('method', 'Unknown')
            color_name = detection.get('color', None)
            
            # Get grid position
            grid_pos = self.get_grid_position(center)
            
            # Choose color based on detection method and color
            if color_name and color_name in self.color_ranges:
                draw_color = self.color_ranges[color_name]['color']
            elif method == 'OpenCV DNN':
                draw_color = (0, 255, 0)  # Green for DNN
            elif method == 'Enhanced Hough':
                draw_color = (255, 0, 0)  # Blue for Hough
            else:
                draw_color = (0, 255, 255)  # Yellow for others
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), draw_color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, draw_color, -1)
            
            # Add labels with grid coordinates
            label = f"{class_name}: {confidence:.2f}"
            method_label = f"Method: {method}"
            grid_label = f"Grid: ({grid_pos[0]}, {grid_pos[1]})"
            
            # Calculate label positions
            label_y_offset = 0
            for text in [label, method_label, grid_label]:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Draw background for text
                cv2.rectangle(frame, 
                             (bbox[0], bbox[1] - 25 - label_y_offset),
                             (bbox[0] + text_size[0], bbox[1] - 5 - label_y_offset),
                             draw_color, -1)
                
                # Draw text
                cv2.putText(frame, text, 
                           (bbox[0], bbox[1] - 10 - label_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                label_y_offset += 20
        
        return frame
    
    def draw_trail(self, frame, center):
        """Draw trailing path of the primary detected object"""
        if center is not None:
            self.ball_trail.appendleft(center)
        
        # Draw the trail
        for i in range(1, len(self.ball_trail)):
            if self.ball_trail[i - 1] is None or self.ball_trail[i] is None:
                continue
            
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, self.ball_trail[i - 1], self.ball_trail[i], (0, 255, 255), thickness)
    
    def process_hands(self, frame):
        """Process hand detection using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame

def main():
    # Initialize combined detector
    detector = CombinedDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Combined Ball and Color Detection Started")
    print("Controls:")
    print("q - Quit")
    print("c - Clear object trail")
    print("g - Toggle grid")
    print("b - Toggle ball detection")
    print("o - Toggle color detection")
    print("s - Toggle shape detection")
    print("+ / - - Adjust grid resolution")
    print("h - Show help")
    
    # Performance monitoring
    fps_counter = 0
    fps_timer = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect all objects
            detected_objects = detector.detect_all_objects(frame)
            
            # Draw detection information
            frame = detector.draw_detection_info(frame, detected_objects)
            
            # Draw trail for the first detected object (if any)
            if detected_objects:
                detector.draw_trail(frame, detected_objects[0]['center'])
                
                # Print object grid position to console
                grid_pos = detector.get_grid_position(detected_objects[0]['center'])
                obj_name = detected_objects[0]['class_name']
                print(f"{obj_name} at grid position: ({grid_pos[0]}, {grid_pos[1]})", end="\r")
            
            # Process hands
            frame = detector.process_hands(frame)
            
            # Add performance and status info
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            
            # Status information
            status_y = 30
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status_y += 25
            cv2.putText(frame, f"Objects detected: {len(detected_objects)}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detection modes status
            status_y += 25
            modes = []
            if detector.enable_ball_detection: modes.append("Ball")
            if detector.enable_color_detection: modes.append("Color")
            if detector.enable_shape_detection: modes.append("Shape")
            cv2.putText(frame, f"Modes: {', '.join(modes)}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Grid info
            if detector.show_grid:
                status_y += 25
                cv2.putText(frame, f"Grid: {detector.grid_rows}x{detector.grid_cols}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Combined Ball and Color Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.ball_trail.clear()
                print("Object trail cleared")
            elif key == ord('g'):
                detector.show_grid = not detector.show_grid
                print(f"Grid display: {'ON' if detector.show_grid else 'OFF'}")
            elif key == ord('b'):
                detector.enable_ball_detection = not detector.enable_ball_detection
                print(f"Ball detection: {'ON' if detector.enable_ball_detection else 'OFF'}")
            elif key == ord('o'):
                detector.enable_color_detection = not detector.enable_color_detection
                print(f"Color detection: {'ON' if detector.enable_color_detection else 'OFF'}")
            elif key == ord('s'):
                detector.enable_shape_detection = not detector.enable_shape_detection
                print(f"Shape detection: {'ON' if detector.enable_shape_detection else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                detector.grid_rows = min(detector.grid_rows + 2, 30)
                detector.grid_cols = min(detector.grid_cols + 2, 30)
                print(f"Grid resolution: {detector.grid_rows}x{detector.grid_cols}")
            elif key == ord('-'):
                detector.grid_rows = max(detector.grid_rows - 2, 4)
                detector.grid_cols = max(detector.grid_cols - 2, 4)
                print(f"Grid resolution: {detector.grid_rows}x{detector.grid_cols}")
            elif key == ord('h'):
                print("\nControls:")
                print("q - Quit")
                print("c - Clear object trail")
                print("g - Toggle grid")
                print("b - Toggle ball detection")
                print("o - Toggle color detection")
                print("s - Toggle shape detection")
                print("+ / - - Adjust grid resolution")
                print("h - Show help")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        detector.hands.close()
        print("\nCamera and windows closed")

if __name__ == "__main__":
    main()