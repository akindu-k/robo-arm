#!/usr/bin/env python3
"""
Ball Detection using OpenCV and MediaPipe
Detects circular objects (balls) using HoughCircles and contour analysis
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

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
        self.ball_trail = deque(maxlen=64)  # Store ball positions for trail
        
        # Detection parameters
        self.min_radius = 15
        self.max_radius = 150
        self.min_area = 300
        
        # HoughCircles parameters
        self.dp = 1
        self.min_dist = 50
        self.param1 = 50
        self.param2 = 30
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better circle detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        return blurred
    
    def detect_balls_hough(self, frame):
        """Detect balls using HoughCircles"""
        gray = self.preprocess_frame(frame)
        
        # Use HoughCircles to detect circular objects
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_balls = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                detected_balls.append({
                    'center': (x, y),
                    'radius': r,
                    'confidence': 1.0
                })
        
        return detected_balls
    
    def detect_balls_contour(self, frame):
        """Alternative ball detection using contours"""
        gray = self.preprocess_frame(frame)
        
        # Apply threshold to create binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_balls = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            if area > self.min_area:
                # Get minimum enclosing circle
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                
                # Check if radius is within range
                if self.min_radius < radius < self.max_radius:
                    # Calculate circularity to filter out non-circular objects
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Only accept relatively circular objects
                        if circularity > 0.4:
                            detected_balls.append({
                                'center': (int(x), int(y)),
                                'radius': int(radius),
                                'confidence': circularity
                            })
        
        return detected_balls
    
    def detect_balls(self, frame):
        """Main ball detection function combining multiple methods"""
        # Try HoughCircles first
        balls_hough = self.detect_balls_hough(frame)
        
        # If HoughCircles doesn't find anything, try contour method
        if not balls_hough:
            balls_contour = self.detect_balls_contour(frame)
            return balls_contour
        
        return balls_hough
    
    def draw_ball_info(self, frame, balls):
        """Draw detection information on the frame"""
        for i, ball in enumerate(balls):
            center = ball['center']
            radius = int(ball['radius'])
            confidence = ball.get('confidence', 1.0)
            
            # Use different colors for different balls
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]
            
            # Draw circle around the ball
            cv2.circle(frame, center, radius, color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Add label with confidence
            label = f"Ball {i+1}"
            if confidence < 1.0:
                label += f" ({confidence:.2f})"
            
            cv2.putText(frame, label, 
                       (center[0] - 40, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
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
            cv2.line(frame, self.ball_trail[i - 1], self.ball_trail[i], (0, 255, 0), thickness)
    
    def process_hands(self, frame):
        """Process hand detection using MediaPipe (optional feature)"""
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
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Ball Detection Started. Press 'q' to quit.")
    print("Press 'c' to clear ball trail")
    print("Press 's' to switch detection method")
    print("Detecting circular objects (balls)")
    
    # Detection method toggle
    use_hough = True
    
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
            
            # Detect balls using selected method
            if use_hough:
                detected_balls = detector.detect_balls_hough(frame)
            else:
                detected_balls = detector.detect_balls_contour(frame)
            
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
            
            # Add detection count
            cv2.putText(frame, f"Balls detected: {len(detected_balls)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection method info
            method_text = "Hough Circles" if use_hough else "Contour Analysis"
            cv2.putText(frame, f"Method: {method_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Ball Detection - Webcam', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                use_hough = not use_hough
                method = "Hough Circles" if use_hough else "Contour Analysis"
                print(f"Switched to: {method}")
            elif key == ord('h'):
                print("\nControls:")
                print("q - Quit")
                print("c - Clear trail")
                print("s - Switch detection method")
                print("h - Show help")
    
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