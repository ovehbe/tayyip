# 📱 Visual Odometry Demo - Tayyip

A comprehensive Visual Odometry demonstration project featuring both Python OpenCV and Android ARCore implementations for real-time camera tracking and GPS integration.

## 🚀 Features

### Python Implementation
- **Live Monocular Visual Odometry** using OpenCV
- **Real-time trajectory visualization** 
- **GPS control simulation** with keyboard controls
- **CSV trajectory export**
- **Interactive mode switching** (GPS ↔ Visual Odometry)

### Android Implementation  
- **ARCore integration** for mobile visual tracking
- **Live camera preview** with position overlay
- **GPS + Visual Odometry** dual-mode tracking
- **Real-time trajectory visualization** below camera
- **CSV export and sharing** functionality
- **Clean Material Design UI**

## 📱 Screenshots

The Android app features:
- Live camera feed with position overlay
- Real-time trajectory plotting (green = GPS, orange = VO)
- Mode switching controls
- Save & share trajectory data

## 🛠️ Installation

### Python Demo

```bash
cd live_vo
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib

# Run basic demo
python live_vo.py

# Run enhanced demo with GPS controls
python live_vo_with_gps.py

# Interactive launcher
python demo.py
```

### Android App

1. **Download the APK**: [Latest Release](https://github.com/ovehbe/tayyip/releases)
2. **Install on Android device** with ARCore support
3. **Grant permissions** for camera and location
4. **Launch "ARCore GPS Demo"**

Or build from source:
```bash
cd live_vo/android_arcore_gps
./gradlew assembleRelease
# APK will be in app/build/outputs/apk/release/
```

## 🎮 Usage

### Python Controls
- **ESC**: Exit and save trajectory.csv
- **C**: Clear pose and trajectory (enhanced version)
- **S**: Stop GPS (VO-only mode, red trajectory)
- **R**: Restart GPS (green trajectory)

### Android Controls
- **Clear**: Reset AR session and trajectory
- **Stop GPS**: Switch to Visual Odometry mode (orange)
- **Restart GPS**: Switch back to GPS mode (green)
- **Save & Share**: Export CSV and open share menu

## 📊 Data Output

Both implementations export trajectory data in CSV format:

**Python**: `trajectory.csv`
```csv
frame,x,y,z
0,0.00,0.00,0.00
1,0.02,-0.01,0.03
...
```

**Android**: `trajectory_YYYY-MM-DD_HH-mm-ss.csv`
```csv
frame,x,y,z,mode,timestamp
0,0.042,-0.010,-0.037,GPS,1627849200000
1,0.035,0.047,-0.041,VO,1627849200100
...
```

## 🔧 Technical Details

### Visual Odometry Algorithm
1. **ORB Feature Detection** (Oriented FAST and Rotated BRIEF)
2. **Feature Matching** with Brute-force matcher
3. **Essential Matrix Estimation** using RANSAC
4. **Pose Recovery** from essential matrix decomposition
5. **Incremental Pose Integration**

### Android Architecture
- **Camera2 API** for live camera preview
- **ARCore** for pose tracking (with fallback simulation)
- **Custom TrajectoryView** for real-time visualization
- **GPS integration** with LocationServices
- **FileProvider** for secure CSV sharing

## 📁 Project Structure

```
tayyip/
├── live_vo/                          # Python implementation
│   ├── live_vo.py                    # Basic VO demo
│   ├── live_vo_with_gps.py          # Enhanced VO with GPS controls
│   ├── demo.py                       # Interactive launcher
│   ├── test_installation.py          # Dependency verification
│   ├── venv/                         # Python virtual environment
│   └── android_arcore_gps/           # Android implementation
│       ├── app/src/main/
│       │   ├── java/com/example/arcoregps/
│       │   │   ├── MainActivity.kt   # Main activity
│       │   │   ├── CameraManager.kt  # Camera + tracking logic
│       │   │   └── TrajectoryView.kt # Custom trajectory visualization
│       │   └── res/                  # Android resources
│       ├── build.gradle              # Project dependencies
│       ├── CAMERA_FIX_README.md     # Camera implementation notes
│       ├── TRAJECTORY_FIXES.md      # Tracking improvements
│       └── debug_app.sh             # Debug helper script
└── README.md                         # This file
```

## 🎯 Key Improvements

### Camera Integration
- **Fixed blank camera issue** - Switched from GLSurfaceView to SurfaceView
- **Proper Camera2 API** implementation with lifecycle management
- **Real-time preview** with position overlays

### Trajectory Visualization
- **Smart filtering** - 5cm movement threshold to reduce noise
- **Auto-scaling** - Minimum 2×2m view with padding
- **Clean visualization** - Every 3rd point drawn, proper scaling
- **Informative display** - Shows scale, distance, point count

### Mode Differentiation  
- **GPS Mode**: Gradual movement with occasional position jumps
- **VO Mode**: Smoother tracking with exponential smoothing
- **Visual distinction** - Different colors and movement patterns

## 🔬 Camera Calibration

The demo uses dummy camera intrinsics:
```python
K = np.array([[700,   0, 320],
              [  0, 700, 240], 
              [  0,   0,   1]])
```

For better accuracy, calibrate your specific camera using `cv2.calibrateCamera()`.

## 📋 Requirements

### Python
- Python 3.7+
- OpenCV 4.0+
- NumPy
- Matplotlib
- Webcam/Camera

### Android
- Android 7.0+ (API 24+)
- ARCore compatible device
- Camera and GPS permissions
- 50MB storage space

## 🐛 Troubleshooting

### Python Issues
- **Camera not detected**: Check permissions and camera connectivity
- **Poor tracking**: Ensure good lighting and textured environment
- **Installation errors**: Verify virtual environment activation

### Android Issues
- **App crashes**: Check ARCore compatibility and permissions
- **No camera feed**: Restart app, check camera permissions
- **GPS not working**: Enable location services, go outdoors

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenCV** for computer vision algorithms
- **Google ARCore** for mobile AR tracking
- **Android Camera2 API** for camera integration
- **Visual Odometry research community** for algorithmic foundations

---

Built with ❤️ for computer vision and mobile AR enthusiasts!