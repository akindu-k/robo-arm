# Robotic Arm Ball Tracking System

This project implements a real-time ball detection and tracking system that overlays a customizable 2D grid array on the camera feed. The system maps the ball's physical position to discrete grid coordinates, making it suitable for robotic arm positioning and control applications.

---

## 🔧 Features

### 🎯 Multi-method Ball Detection
- **OpenCV Deep Neural Networks (MobileNet-SSD)**
- **Enhanced Hough Circle Detection**
- **MediaPipe Object Detection Pipeline**

### 🧮 2D Grid Array Overlay
- Customizable grid resolution (rows and columns)
- Real-time positioning of detected balls within grid coordinates
- Visual display of grid coordinates

### 🚀 Advanced Tracking
- Ball movement trail visualization
- Position history tracking
- Hand detection for context awareness

---

## 📦 Requirements

- Python 3.6 or higher
- Webcam or compatible camera
- Dependencies listed in `requirements.txt`

---

## 📥 Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/robo-arm.git
cd robo-arm/DNN
````

Install dependencies:

```bash
pip install -r requirements.txt
```

### Download Model Files (Optional but Recommended)

The script will try to auto-download the `.prototxt` file.
You **may need to manually download** the `.caffemodel` due to Google Drive restrictions.

Place the following in the same directory as the script:

* `MobileNetSSD_deploy.prototxt`
* `MobileNetSSD_deploy.caffemodel`

---

## ▶️ Usage

Run the ball detection script with grid overlay:

```bash
python ball_detector_DNN_with_array_overlay.py
```

---

## 🎛️ Controls

* `q`: Quit the application
* `c`: Clear the ball movement trail
* `g`: Toggle grid display on/off
* `+` / `-`: Increase / Decrease grid resolution
* `h`: Show help information

---

## ⚙️ How It Works

The system uses multiple detection methods to locate balls in the video feed:

1. Attempts detection using pre-trained OpenCV DNN (MobileNet-SSD)
2. Falls back to MediaPipe-enhanced detection
3. Finally, uses enhanced Hough circle detection if needed

When a ball is detected:

* Maps pixel position to grid coordinates
* Displays the position on-screen
* Tracks movement with a visual trail
* Prints current grid coordinates to the console

---

## 📐 Grid System

* The 2D grid overlay divides the camera view into a configurable number of rows and columns (default: 10x10)
* Grid coordinates are expressed as `(x, y)` with `(0, 0)` at the top-left
* Grid resolution can be adjusted at runtime

This discretized mapping can be integrated with robotic arm controls for positioning.

---

## 🛠️ Troubleshooting

* **Camera access issues:** Ensure your webcam is connected and not in use by another app
* **Model loading errors:** The system will automatically switch to other detection methods
* **Performance issues:** Lower camera resolution or tweak detection parameters

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).


