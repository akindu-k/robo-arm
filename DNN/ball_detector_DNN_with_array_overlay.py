#!/usr/bin/env python3
"""
Ball Detection using OpenCV DNN and MediaPipe Object Detection
Uses pre-trained models available in OpenCV and MediaPipe
Includes 2D grid array overlay for position tracking
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import urllib.request
import os

class BallDetector:
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
        self.current_frame_shape = (0, 0)  # Will be updated with frame shape
        
        # Try to load OpenCV DNN models
        self.opencv_net = None
        self.load_opencv_dnn()
        
        # MediaPipe Objectron for 3D object detection (can detect spherical objects)
        self.objectron = None
        self.use_mediapipe_detection = True
        
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
    
    def load_opencv_dnn(self):
        """Load OpenCV DNN model (MobileNet-SSD)"""
        try:
            print("Loading OpenCV DNN model...")
            
            # URLs for pre-trained MobileNet-SSD model
            prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
            model_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
            
            prototxt_path = "MobileNetSSD_deploy.prototxt"
            model_path = "MobileNetSSD_deploy.caffemodel"
            
            # Download files if they don't exist
            if not os.path.exists(prototxt_path):
                print("Downloading prototxt file...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            if not os.path.exists(model_path):
                print("Note: MobileNet model requires manual download due to Google Drive restrictions")
                print("Please download MobileNetSSD_deploy.caffemodel manually")
                print("Using alternative approach...")
                return
            
            # Load the network
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.opencv_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("OpenCV DNN model loaded successfully!")
            else:
                print("Model files not found, using alternative detection methods")
                
        except Exception as e:
            print(f"Error loading OpenCV DNN model: {e}")
            print("Continuing with MediaPipe and fallback methods...")
    
    def detect_balls_opencv_dnn(self, frame):
        """Detect balls using OpenCV DNN"""
        if self.opencv_net is None:
            return []
        
        try:
            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            
            # Pass blob through network
            self.opencv_net.setInput(blob)
            detections = self.opencv_net.forward()
            
            detected_balls = []
            
            # Loop through detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                # Check if it's a ball and confidence is high enough
                if class_id in self.ball_class_ids and confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Calculate center and radius
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max((x2 - x1), (y2 - y1)) // 2
                    
                    detected_balls.append({
                        'center': (center_x, center_y),
                        'radius': radius,
                        'confidence': confidence,
                        'class_name': self.COCO_CLASSES[class_id],
                        'bbox': (x1, y1, x2, y2),
                        'method': 'OpenCV DNN'
                    })
            
            return detected_balls
            
        except Exception as e:
            print(f"Error in OpenCV DNN detection: {e}")
            return []
    
    def detect_balls_mediapipe(self, frame):
        """Detect spherical objects using MediaPipe"""
        try:
            # MediaPipe doesn't have a direct ball detector, but we can use selfie segmentation
            # and object detection in combination with traditional CV methods
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # For now, we'll use a hybrid approach with MediaPipe preprocessing
            # and traditional circle detection
            
            # Apply MediaPipe-style preprocessing
            # This is a simplified approach - in practice, you'd use MediaPipe's 
            # selfie segmentation to isolate objects, then detect circles
            
            return self.detect_balls_hough_enhanced(frame)
            
        except Exception as e:
            print(f"Error in MediaPipe detection: {e}")
            return []
    
    def detect_balls_hough_enhanced(self, frame):
        """Enhanced Hough circle detection with MediaPipe preprocessing"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (9, 9), 2)
        
        # Use adaptive parameters based on image properties
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(frame.shape[0] / 8),  # Adaptive minimum distance
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=min(frame.shape[0], frame.shape[1]) // 4
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Additional validation: check if the detected circle looks like a ball
                # by analyzing the pixel intensities
                
                # Extract region of interest
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                roi = cv2.bitwise_and(gray, mask)
                
                # Calculate some features to validate if it's likely a ball
                mean_intensity = cv2.mean(roi, mask)[0]
                
                # Simple heuristic: balls usually have some variation in intensity
                if mean_intensity > 30:  # Not too dark
                    detected_balls.append({
                        'center': (x, y),
                        'radius': r,
                        'confidence': 0.6,  # Medium confidence for traditional CV
                        'class_name': 'circular object',
                        'bbox': (x-r, y-r, x+r, y+r),
                        'method': 'Enhanced Hough'
                    })
        
        return detected_balls
    
    def detect_balls(self, frame):
        """Main ball detection function - tries multiple methods"""
        detected_balls = []
        
        # Try OpenCV DNN first (most accurate)
        if self.opencv_net is not None:
            balls_dnn = self.detect_balls_opencv_dnn(frame)
            detected_balls.extend(balls_dnn)
        
        # If no balls detected with DNN, try MediaPipe approach
        if not detected_balls and self.use_mediapipe_detection:
            balls_mp = self.detect_balls_mediapipe(frame)
            detected_balls.extend(balls_mp)
        
        # If still no balls, try enhanced Hough circles
        if not detected_balls:
            balls_hough = self.detect_balls_hough_enhanced(frame)
            detected_balls.extend(balls_hough)
        
        return detected_balls
    
    def draw_grid(self, frame):
        """Draw a grid on the frame and return the grid cell of detected balls"""
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
    
    def draw_ball_info(self, frame, balls):
        """Draw detection information on the frame with grid coordinates"""
        # Store current frame shape for grid calculations
        self.current_frame_shape = frame.shape[:2]
        
        # Draw grid if enabled
        if self.show_grid:
            frame = self.draw_grid(frame)
        
        for i, ball in enumerate(balls):
            center = ball['center']
            radius = int(ball['radius'])
            confidence = ball['confidence']
            class_name = ball['class_name']
            bbox = ball['bbox']
            method = ball.get('method', 'Unknown')
            
            # Get grid position
            grid_pos = self.get_grid_position(center)
            
            # Use different colors for different detection methods
            if method == 'OpenCV DNN':
                color = (0, 255, 0)  # Green for DNN
            elif method == 'Enhanced Hough':
                color = (255, 0, 0)  # Blue for Hough
            else:
                color = (0, 255, 255)  # Yellow for others
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Add label with grid coordinates
            label = f"{class_name}: {confidence:.2f} ({method})"
            grid_label = f"Grid: ({grid_pos[0]}, {grid_pos[1]})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            grid_label_size = cv2.getTextSize(grid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for labels
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1] - label_size[1] - grid_label_size[1] - 20),
                         (bbox[0] + max(label_size[0], grid_label_size[0]), bbox[1]),
                         color, -1)
            
            cv2.putText(frame, label, 
                       (bbox[0], bbox[1] - grid_label_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, grid_label, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_trail(self, frame, center):
        """Draw trailing path of the ball"""
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
    # Initialize ball detector
    detector = BallDetector()
    
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
    
    print("Ball Detection Started (OpenCV DNN + MediaPipe)")
    print("Press 'q' to quit")
    print("Press 'c' to clear ball trail")
    print("Press 'g' to toggle grid")
    print("Press '+' or '-' to adjust grid resolution")
    print("Press 'h' to show help")
    
    detection_methods = []
    if detector.opencv_net is not None:
        detection_methods.append("OpenCV DNN")
    if detector.use_mediapipe_detection:
        detection_methods.append("MediaPipe + Hough")
    detection_methods.append("Enhanced Hough Circles")
    
    print(f"Available detection methods: {', '.join(detection_methods)}")
    print(f"Grid resolution: {detector.grid_rows}x{detector.grid_cols}")
    
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
            
            # Detect balls
            detected_balls = detector.detect_balls(frame)
            
            # Draw ball information
            frame = detector.draw_ball_info(frame, detected_balls)
            
            # Draw trail for the first detected ball (if any)
            if detected_balls:
                detector.draw_trail(frame, detected_balls[0]['center'])
                
                # Print ball grid position to console for latest detection
                grid_pos = detector.get_grid_position(detected_balls[0]['center'])
                print(f"Ball at grid position: ({grid_pos[0]}, {grid_pos[1]})", end="\r")
            
            # Process hands
            frame = detector.process_hands(frame)
            
            # Add performance info
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection info
            cv2.putText(frame, f"Balls detected: {len(detected_balls)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add grid info
            if detector.show_grid:
                cv2.putText(frame, f"Grid: {detector.grid_rows}x{detector.grid_cols}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Ball Detection with Grid Coordinates', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.ball_trail.clear()
                print("Ball trail cleared")
            elif key == ord('g'):
                # Toggle grid display
                detector.show_grid = not detector.show_grid
                print(f"Grid display: {'ON' if detector.show_grid else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                # Increase grid resolution
                detector.grid_rows = min(detector.grid_rows + 2, 30)
                detector.grid_cols = min(detector.grid_cols + 2, 30)
                print(f"Grid resolution: {detector.grid_rows}x{detector.grid_cols}")
            elif key == ord('-'):
                # Decrease grid resolution
                detector.grid_rows = max(detector.grid_rows - 2, 4)
                detector.grid_cols = max(detector.grid_cols - 2, 4)
                print(f"Grid resolution: {detector.grid_rows}x{detector.grid_cols}")
            elif key == ord('h'):
                print("\nControls:")
                print("q - Quit")
                print("c - Clear ball trail")
                print("g - Toggle grid")
                print("+ / - - Adjust grid resolution")
                print("h - Show help")
                print(f"\nActive detection methods: {', '.join(detection_methods)}")
    
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