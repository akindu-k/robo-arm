#!/usr/bin/env python3
"""
Ball Detection using Pre-trained YOLO Model and MediaPipe
Uses YOLOv5 for accurate ball detection from trained datasets
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import torch
from collections import deque
import urllib.request
import os

class BallDetector:
    def __init__(self):
        # Initialize MediaPipe hands for additional context (optional)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Ball tracking parameters
        self.ball_trail = deque(maxlen=64)
        
        # Load YOLO model
        self.model = None
        self.load_yolo_model()
        
        # Ball-related class IDs from COCO dataset
        self.ball_classes = {
            32: 'sports ball',  # Generic sports ball
            # You can add more specific ball classes if available
        }
        
        # Detection parameters
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        
    def load_yolo_model(self):
        """Load pre-trained YOLO model"""
        try:
            print("Loading YOLOv5 model...")
            # Load YOLOv5 model (will download automatically on first run)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = self.confidence_threshold  # confidence threshold
            self.model.iou = self.nms_threshold  # NMS IoU threshold
            print("YOLOv5 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to alternative detection method...")
            self.model = None
    
    def detect_balls_yolo(self, frame):
        """Detect balls using YOLO model"""
        if self.model is None:
            return []
            
        try:
            # Run inference
            results = self.model(frame)
            
            # Parse results
            detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
            
            detected_balls = []
            
            for _, detection in detections.iterrows():
                class_id = int(detection['class'])
                confidence = float(detection['confidence'])
                
                # Check if detected object is a ball or sports-related object
                class_name = detection['name'].lower()
                
                # Look for ball-related objects
                if (class_id in self.ball_classes or 
                    'ball' in class_name or 
                    'soccer' in class_name or 
                    'football' in class_name or 
                    'basketball' in class_name or 
                    'tennis' in class_name or 
                    'baseball' in class_name):
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    
                    # Calculate center and radius
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = max((x2 - x1), (y2 - y1)) // 2
                    
                    detected_balls.append({
                        'center': (center_x, center_y),
                        'radius': radius,
                        'confidence': confidence,
                        'class_name': class_name,
                        'bbox': (x1, y1, x2, y2)
                    })
            
            return detected_balls
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def detect_balls_fallback(self, frame):
        """Fallback detection method using traditional CV"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use HoughCircles as fallback
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=150
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                detected_balls.append({
                    'center': (x, y),
                    'radius': r,
                    'confidence': 0.5,
                    'class_name': 'circle',
                    'bbox': (x-r, y-r, x+r, y+r)
                })
        
        return detected_balls
    
    def detect_balls(self, frame):
        """Main ball detection function"""
        if self.model is not None:
            return self.detect_balls_yolo(frame)
        else:
            return self.detect_balls_fallback(frame)
    
    def draw_ball_info(self, frame, balls):
        """Draw detection information on the frame"""
        for i, ball in enumerate(balls):
            center = ball['center']
            radius = int(ball['radius'])
            confidence = ball['confidence']
            class_name = ball['class_name']
            bbox = ball['bbox']
            
            # Use different colors for different balls
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Add label with confidence and class
            label = f"{class_name.title()}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background for label
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            cv2.putText(frame, label, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add coordinates
            coord_text = f"({center[0]}, {center[1]})"
            cv2.putText(frame, coord_text,
                       (center[0] - 30, center[1] + radius + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
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
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
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
    
    print("Ball Detection Started (YOLO + MediaPipe)")
    print("Press 'q' to quit")
    print("Press 'c' to clear ball trail")
    print("Press 'h' to show help")
    
    if detector.model is not None:
        print("Using YOLOv5 for ball detection")
    else:
        print("Using fallback circle detection")
    
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
            
            # Optional: Process hands
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
            
            # Add model info
            model_text = "YOLO" if detector.model else "Fallback"
            cv2.putText(frame, f"Model: {model_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Ball Detection - YOLO + Webcam', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.ball_trail.clear()
                print("Ball trail cleared")
            elif key == ord('h'):
                print("\nControls:")
                print("q - Quit")
                print("c - Clear trail")
                print("h - Show help")
                print("\nDetected ball types:")
                for class_id, class_name in detector.ball_classes.items():
                    print(f"  - {class_name}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        detector.hands.close()
        print("Camera and windows closed")

if __name__ == "__main__":
    main()