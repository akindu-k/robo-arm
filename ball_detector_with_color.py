#!/usr/bin/env python3
"""
Ball Detection using OpenCV and MediaPipe on Raspberry Pi
Detects colored balls using HSV color filtering and contour analysis
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
        
        # HSV color ranges for different colored balls
        # You can adjust these ranges based on your ball colors
        self.color_ranges = {
            'orange': {
                'lower': np.array([5, 50, 50]),
                'upper': np.array([15, 255, 255])
            },
            'green': {
                'lower': np.array([40, 50, 50]),
                'upper': np.array([80, 255, 255])
            },
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'red': {
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255])
            },
            'yellow': {
                'lower': np.array([20, 50, 50]),
                'upper': np.array([30, 255, 255])
            }
        }
        
        # Detection parameters
        self.min_radius = 10
        self.max_radius = 100
        self.min_area = 500
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def detect_ball_by_color(self, hsv_frame, color_name):
        """Detect ball using color filtering"""
        if color_name not in self.color_ranges:
            return None, None
            
        color_range = self.color_ranges[color_name]
        
        # Create mask for the specified color
        mask = cv2.inRange(hsv_frame, color_range['lower'], color_range['upper'])
        
        # Handle red color (wraps around HSV hue)
        if color_name == 'red':
            mask2 = cv2.inRange(hsv_frame, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        
        # Clean up the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        radius = 0
        
        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if contour is large enough
            if cv2.contourArea(largest_contour) > self.min_area:
                # Get minimum enclosing circle
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                
                # Check if radius is within acceptable range
                if self.min_radius < radius < self.max_radius:
                    # Calculate moments to find center
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        center = (int(x), int(y))
        
        return center, radius
    
    def detect_balls(self, frame):
        """Detect balls of different colors in the frame"""
        hsv_frame = self.preprocess_frame(frame)
        detected_balls = []
        
        for color_name in self.color_ranges.keys():
            center, radius = self.detect_ball_by_color(hsv_frame, color_name)
            
            if center is not None and radius > 0:
                detected_balls.append({
                    'center': center,
                    'radius': radius,
                    'color': color_name
                })
        
        return detected_balls
    
    def draw_ball_info(self, frame, balls):
        """Draw detection information on the frame"""
        for ball in balls:
            center = ball['center']
            radius = int(ball['radius'])
            color_name = ball['color']
            
            # Define colors for drawing (BGR format)
            draw_colors = {
                'orange': (0, 165, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'red': (0, 0, 255),
                'yellow': (0, 255, 255)
            }
            
            draw_color = draw_colors.get(color_name, (255, 255, 255))
            
            # Draw circle around the ball
            cv2.circle(frame, center, radius, draw_color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, draw_color, -1)
            
            # Add label
            label = f"{color_name.capitalize()} Ball"
            cv2.putText(frame, label, 
                       (center[0] - 50, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
            
            # Add coordinates
            coord_text = f"({center[0]}, {center[1]})"
            cv2.putText(frame, coord_text,
                       (center[0] - 30, center[1] + radius + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
        
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
    
    # Initialize camera (adjust camera index if needed)
    # For Raspberry Pi camera module, you might need to use different parameters
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for Raspberry Pi optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Ball Detection Started. Press 'q' to quit.")
    print("Press 'c' to clear ball trail")
    print("Detecting colors: orange, green, blue, red, yellow")
    
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
            
            # Add detection count
            cv2.putText(frame, f"Balls detected: {len(detected_balls)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Ball Detection - Raspberry Pi', frame)
            
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