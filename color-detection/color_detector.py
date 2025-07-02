import cv2
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import Counter
import matplotlib.pyplot as plt

class BallColorDetector:
    def __init__(self):
        self.model = None
        self.label_encoder = {'red': 0, 'green': 1, 'blue': 2}
        self.label_decoder = {0: 'red', 1: 'green', 2: 'blue'}
        
        # HSV color ranges for traditional CV approach (backup method)
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 50]), np.array([10, 255, 255])),
                (np.array([170, 120, 50]), np.array([180, 255, 255]))
            ],
            'green': [(np.array([40, 120, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 120, 50]), np.array([140, 255, 255]))]
        }
    
    def extract_features(self, image):
        """Extract color features from image for ML model"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Find the ball (largest contour that's roughly circular)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=10, maxRadius=200)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Use the largest circle
            circle = max(circles, key=lambda c: c[2])
            x, y, r = circle
            
            # Create mask for the ball
            mask = np.zeros(gray.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Extract features from the masked region
            ball_bgr = cv2.bitwise_and(image, image, mask=mask)
            ball_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
            ball_lab = cv2.bitwise_and(lab, lab, mask=mask)
            
        else:
            # Fallback: use center region
            h, w = image.shape[:2]
            center_h, center_w = h//4, w//4
            ball_bgr = image[center_h:3*center_h, center_w:3*center_w]
            ball_hsv = hsv[center_h:3*center_h, center_w:3*center_w]
            ball_lab = lab[center_h:3*center_h, center_w:3*center_w]
        
        # Calculate color statistics
        features = []
        
        # BGR statistics
        for channel in range(3):
            channel_data = ball_bgr[:,:,channel][ball_bgr[:,:,channel] > 0]
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])
            else:
                features.extend([0, 0, 0])
        
        # HSV statistics
        for channel in range(3):
            channel_data = ball_hsv[:,:,channel][ball_hsv[:,:,channel] > 0]
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])
            else:
                features.extend([0, 0])
        
        # LAB statistics
        for channel in range(3):
            channel_data = ball_lab[:,:,channel][ball_lab[:,:,channel] > 0]
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])
            else:
                features.extend([0, 0])
        
        # Color dominance features
        hist_b = cv2.calcHist([ball_bgr], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([ball_bgr], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([ball_bgr], [2], None, [32], [0, 256])
        
        features.extend([
            np.argmax(hist_b), np.max(hist_b),
            np.argmax(hist_g), np.max(hist_g),
            np.argmax(hist_r), np.max(hist_r)
        ])
        
        return np.array(features)
    
    def load_data_from_folder(self, folder_path):
        """Load images and labels from a folder containing images and label files"""
        features = []
        labels = []
        
        # Check for bounding box labels file in the folder
        bbox_file = os.path.join(folder_path, 'bounding_boxes.labels')
        info_file = os.path.join(folder_path, 'info.labels')
        
        if not os.path.exists(bbox_file):
            print(f"Warning: {bbox_file} not found")
            return np.array([]), np.array([])
        
        # Load bounding box annotations
        with open(bbox_file, 'r') as f:
            annotation_data = json.load(f)
        
        # Optionally load info labels for additional context
        info_data = None
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info_data = json.load(f)
        
        processed_count = 0
        skipped_count = 0
        
        for file_info in annotation_data.get('files', []):
            filename = file_info['path']
            image_path = os.path.join(folder_path, filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {filename} not found in {folder_path}")
                skipped_count += 1
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {filename}")
                skipped_count += 1
                continue
            
            # Process each bounding box in the image
            bounding_boxes = file_info.get('boundingBoxes', [])
            
            if not bounding_boxes:
                # Skip images without bounding boxes
                continue
            
            for bbox in bounding_boxes:
                bbox_label = bbox['label'].lower()
                
                # Extract color from label (e.g., "Blue Ball" -> "blue")
                color = None
                for c in ['red', 'green', 'blue']:
                    if c in bbox_label:
                        color = c
                        break
                
                if color is None:
                    print(f"Warning: Could not extract color from label '{bbox_label}' in {filename}")
                    skipped_count += 1
                    continue
                
                # Extract ball region using bounding box
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Ensure bounding box is within image bounds
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                if w <= 0 or h <= 0:
                    print(f"Warning: Invalid bounding box in {filename}")
                    skipped_count += 1
                    continue
                
                ball_region = image[y:y+h, x:x+w]
                
                if ball_region.size == 0:
                    print(f"Warning: Empty bounding box in {filename}")
                    skipped_count += 1
                    continue
                
                # Extract features from the ball region
                feature_vector = self.extract_features_from_region(ball_region)
                features.append(feature_vector)
                labels.append(self.label_encoder[color])
                processed_count += 1
        
        print(f"From {folder_path}: Processed {processed_count} ball annotations, skipped {skipped_count}")
        
        return np.array(features), np.array(labels)
    
    def extract_features_from_region(self, region):
        """Extract features from a cropped ball region (more accurate than full image)"""
        if region.size == 0:
            return np.zeros(33)  # Return zero features for empty regions
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # BGR statistics (mean, std, median for each channel)
        for channel in range(3):
            channel_data = region[:,:,channel].flatten()
            channel_data = channel_data[channel_data > 0]  # Remove black pixels
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])
            else:
                features.extend([0, 0, 0])
        
        # HSV statistics (mean, std for each channel)
        for channel in range(3):
            channel_data = hsv[:,:,channel].flatten()
            if channel == 0:  # Hue channel, handle circular nature
                channel_data = channel_data[channel_data > 0]
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])
            else:
                features.extend([0, 0])
        
        # LAB statistics (mean, std for each channel)
        for channel in range(3):
            channel_data = lab[:,:,channel].flatten()
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])
            else:
                features.extend([0, 0])
        
        # Color histogram features
        hist_b = cv2.calcHist([region], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([region], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([region], [2], None, [16], [0, 256])
        
        features.extend([
            np.argmax(hist_b), np.max(hist_b),
            np.argmax(hist_g), np.max(hist_g),
            np.argmax(hist_r), np.max(hist_r)
        ])
        
        # Dominant color ratios
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            b_mean, g_mean, r_mean = np.mean(region, axis=(0,1))
            total_intensity = b_mean + g_mean + r_mean
            if total_intensity > 0:
                features.extend([
                    r_mean / total_intensity,  # Red ratio
                    g_mean / total_intensity,  # Green ratio
                    b_mean / total_intensity   # Blue ratio
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def train_model(self, training_folder, testing_folder=None, use_separate_test=True):
        """Train the color detection model using separate training and testing folders"""
        print("Loading training data...")
        X_train, y_train = self.load_data_from_folder(training_folder)
        
        if len(X_train) == 0:
            print("No training data loaded! Please check your training folder and label files.")
            return 0
        
        print(f"Training samples: {len(X_train)}")
        train_distribution = Counter([self.label_decoder[label] for label in y_train])
        print(f"Training distribution: {dict(train_distribution)}")
        
        if testing_folder and use_separate_test and os.path.exists(testing_folder):
            print("\nLoading separate testing data...")
            X_test, y_test = self.load_data_from_folder(testing_folder)
            
            if len(X_test) > 0:
                print(f"Testing samples: {len(X_test)}")
                test_distribution = Counter([self.label_decoder[label] for label in y_test])
                print(f"Testing distribution: {dict(test_distribution)}")
            else:
                print("No testing data found, using train-test split instead")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
        else:
            print("\nUsing train-test split (80-20)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Train Random Forest model
        print("\nTraining model...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['red', 'green', 'blue']))
        
        # Feature importance
        if len(X_train[0]) > 0:
            feature_importance = self.model.feature_importances_
            print(f"\nTop 5 most important features:")
            top_features = np.argsort(feature_importance)[-5:][::-1]
            for i, idx in enumerate(top_features):
                print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.3f}")
        
        return accuracy
    
    def analyze_dataset(self, folder_path, dataset_name="Dataset"):
        """Analyze the dataset in a given folder"""
        print(f"\n=== Analyzing {dataset_name} in {folder_path} ===")
        
        bbox_file = os.path.join(folder_path, 'bounding_boxes.labels')
        info_file = os.path.join(folder_path, 'info.labels')
        
        if not os.path.exists(bbox_file):
            print(f"No bounding_boxes.labels found in {folder_path}")
            return
        
        with open(bbox_file, 'r') as f:
            annotations = json.load(f)
        
        total_files = len(annotations.get('files', []))
        files_with_boxes = 0
        ball_counts = {'red': 0, 'green': 0, 'blue': 0, 'other': 0}
        total_boxes = 0
        
        for file_info in annotations.get('files', []):
            boxes = file_info.get('boundingBoxes', [])
            if boxes:
                files_with_boxes += 1
                for bbox in boxes:
                    total_boxes += 1
                    label = bbox['label'].lower()
                    if 'red' in label:
                        ball_counts['red'] += 1
                    elif 'green' in label:
                        ball_counts['green'] += 1
                    elif 'blue' in label:
                        ball_counts['blue'] += 1
                    else:
                        ball_counts['other'] += 1
        
        print(f"Total files: {total_files}")
        print(f"Files with bounding boxes: {files_with_boxes}")
        print(f"Total bounding boxes: {total_boxes}")
        print(f"Ball color distribution: {dict(ball_counts)}")
        
        # Check if info.labels exists
        if os.path.exists(info_file):
            print(f"Info labels file found: {info_file}")
        else:
            print("No info.labels file found")
        
        # Check for actual image files
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Image files found: {len(image_files)}")
        
        return ball_counts
    
    def detect_color_cv(self, image):
        """Traditional CV approach using HSV color ranges"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_votes = {}
        
        for color_name, ranges in self.color_ranges.items():
            total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_mask = cv2.bitwise_or(total_mask, mask)
            
            # Count pixels in color range
            pixel_count = cv2.countNonZero(total_mask)
            color_votes[color_name] = pixel_count
        
        if max(color_votes.values()) > 100:  # Minimum threshold
            return max(color_votes, key=color_votes.get)
        else:
            return "unknown"
    
    def detect_color_ml(self, image):
        """ML approach using trained model"""
        if self.model is None:
            return "Model not trained"
        
        features = self.extract_features(image).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features))
        
        return self.label_decoder[prediction], confidence
    
    def detect_color(self, image, use_ensemble=True):
        """Combined detection using both approaches"""
        if use_ensemble and self.model is not None:
            # Get predictions from both methods
            cv_result = self.detect_color_cv(image)
            ml_result, ml_confidence = self.detect_color_ml(image)
            
            # Ensemble decision
            if ml_confidence > 0.8:
                return ml_result, ml_confidence, "ML"
            elif cv_result != "unknown":
                return cv_result, 0.7, "CV"  # Assume moderate confidence for CV
            else:
                return ml_result, ml_confidence, "ML"
        else:
            # Use CV method only
            return self.detect_color_cv(image), 0.6, "CV"
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return True
        return False

class RealTimeDetector:
    def __init__(self, detector):
        self.detector = detector
        self.cap = None
    
    def start_camera(self, camera_id=0):
        """Start camera for real-time detection"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        return True
    
    def run_detection(self):
        """Run real-time color detection"""
        if self.cap is None:
            print("Camera not initialized")
            return
        
        print("Starting real-time detection. Press 'q' to quit, 's' to save frame")
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect color every 5 frames to reduce processing load
            if frame_count % 5 == 0:
                result = self.detector.detect_color(frame)
                
                if isinstance(result, tuple):
                    color, confidence, method = result
                    text = f"{color.upper()} ({confidence:.2f}) [{method}]"
                    color_bgr = {'red': (0, 0, 255), 'green': (0, 255, 0), 
                               'blue': (255, 0, 0)}.get(color, (255, 255, 255))
                else:
                    text = f"{result.upper()}"
                    color_bgr = (255, 255, 255)
                
                # Draw text on frame
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color_bgr, 2)
                
                # Draw detection area (center circle)
                h, w = frame.shape[:2]
                cv2.circle(frame, (w//2, h//2), 50, (255, 255, 255), 2)
            
            cv2.imshow('Ball Color Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(f'detected_frame_{frame_count}.jpg', frame)
                print(f"Frame saved as detected_frame_{frame_count}.jpg")
        
        self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = BallColorDetector()
    
    # Initialize detector
    detector = BallColorDetector()
    
    # Define your folder paths
    training_folder = "training"    # Folder containing training images + labels
    testing_folder = "testing"      # Folder containing testing images + labels
    
    # Analyze datasets first
    if os.path.exists(training_folder):
        train_stats = detector.analyze_dataset(training_folder, "Training Dataset")
    else:
        print(f"Training folder '{training_folder}' not found!")
        print("Please create the training folder with your images and label files.")
        train_stats = None
    
    if os.path.exists(testing_folder):
        test_stats = detector.analyze_dataset(testing_folder, "Testing Dataset")
    else:
        print(f"Testing folder '{testing_folder}' not found!")
        print("Will use train-test split instead.")
        test_stats = None
    
    # Training phase
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    if train_stats:
        accuracy = detector.train_model(training_folder, testing_folder)
        if accuracy > 0:
            detector.save_model("ball_color_model.pkl")
            print(f"\nModel saved! Accuracy: {accuracy:.1%}")
        else:
            print("Training failed!")
    else:
        print("Cannot train without training data!")
    
    # Test on a single image from testing folder
    if os.path.exists(testing_folder):
        test_images = [f for f in os.listdir(testing_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_images:
            test_image_path = os.path.join(testing_folder, test_images[0])
            print(f"\n=== Testing on {test_images[0]} ===")
            test_image = cv2.imread(test_image_path)
            if test_image is not None:
                result = detector.detect_color(test_image)
                if isinstance(result, tuple):
                    color, confidence, method = result
                    print(f"Detected color: {color.upper()} (confidence: {confidence:.2f}, method: {method})")
                else:
                    print(f"Detected color: {result.upper()}")
    
    # Show current directory structure for debugging
    print(f"\n=== Current Directory Structure ===")
    current_files = os.listdir(".")
    for item in current_files:
        if os.path.isdir(item):
            print(f"📁 {item}/")
            # Show contents of training/testing folders
            if item in ['training', 'testing']:
                sub_files = os.listdir(item)
                for sub_file in sub_files[:5]:  # Show first 5 files
                    print(f"   - {sub_file}")
                if len(sub_files) > 5:
                    print(f"   ... and {len(sub_files) - 5} more files")
        else:
            print(f"📄 {item}")
    
    # Real-time detection
    print(f"\n{'='*50}")
    print("REAL-TIME DETECTION")
    print("="*50)
    print("Starting camera... Press 'q' to quit")
    
    real_time = RealTimeDetector(detector)
    if real_time.start_camera():
        real_time.run_detection()
    else:
        print("Could not start camera for real-time detection")