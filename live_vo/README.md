# Visual Odometry Demo Project

This project contains a complete Visual Odometry (VO) demo with both Python and Android implementations, as specified in the original prompt.

## Project Structure

```
live_vo/
├── live_vo.py                    # Basic Visual Odometry demo
├── live_vo_with_gps.py          # Enhanced VO with GPS control stub
├── venv/                        # Python virtual environment
├── android_arcore_gps/          # Android ARCore + GPS skeleton
│   └── app/
│       ├── build.gradle         # App dependencies
│       └── src/main/
│           ├── AndroidManifest.xml
│           ├── java/com/example/arcoregps/MainActivity.kt
│           └── res/
└── README.md                    # This file
```

## Python Visual Odometry Demo

### Setup

The environment is already configured with:
- Python virtual environment in `venv/`
- Required packages: opencv-python, numpy, matplotlib

### Running the Basic Demo

```bash
# Activate virtual environment
source venv/bin/activate

# Run basic visual odometry
python live_vo.py
```

**Controls:**
- **ESC**: Exit and save trajectory to `trajectory.csv`
- Move your camera around to see the trajectory being tracked

### Running the Enhanced Demo with GPS Controls

```bash
# Run enhanced version with GPS control stub
python live_vo_with_gps.py
```

**Controls:**
- **ESC**: Exit and save trajectory
- **C**: Clear pose and trajectory  
- **S**: Stop GPS (VO-only mode) - trajectory turns red
- **R**: Restart GPS - trajectory turns green

### Output

Both demos generate:
- **Live camera window**: Shows the webcam feed with status overlay
- **Trajectory window**: Real-time visualization of the estimated path
- **trajectory.csv**: Export file with columns: frame, x, y, z

## Android ARCore + GPS Demo

### Prerequisites

- Android Studio with ARCore support
- Android device with ARCore support
- GPS capability

### Setup

1. Open `android_arcore_gps/` in Android Studio
2. Sync Gradle dependencies
3. Connect ARCore-compatible Android device
4. Build and run the app

### Features

The Android skeleton includes:
- **ARCore integration**: Camera pose tracking using AR
- **GPS integration**: Location services for outdoor tracking
- **Combined logging**: Exports both AR poses and GPS coordinates
- **Control buttons**: Clear, Stop GPS, Restart GPS functionality
- **CSV export**: Saves trajectory data to external storage on app exit

### Key Files

- `MainActivity.kt`: Main activity with ARCore and GPS integration
- `AndroidManifest.xml`: Permissions for camera and location
- `build.gradle`: Dependencies for ARCore and Play Services Location

## Technical Details

### Visual Odometry Algorithm

1. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) features
2. **Feature Matching**: Brute-force matcher with Hamming distance
3. **Essential Matrix**: RANSAC-based estimation between frame pairs
4. **Pose Recovery**: Decomposition of essential matrix to R,t
5. **Pose Accumulation**: Incremental pose updates

### Camera Parameters

The demo uses dummy intrinsic parameters:
```python
K = np.array([[700,   0, 320],
              [  0, 700, 240],
              [  0,   0,   1]])
```

For better accuracy, calibrate your specific camera using `cv2.calibrateCamera()`.

### GPS Integration (Android)

- **Location Updates**: 1-second intervals with high accuracy
- **Coordinate System**: WGS84 latitude/longitude/altitude
- **Origin Setting**: First GPS fix sets the reference point
- **Data Fusion**: Combines AR poses with GPS coordinates

## Usage Tips

### Python Demos

1. **Good Lighting**: Ensure adequate lighting for feature detection
2. **Textured Environment**: Point camera at textured surfaces (avoid blank walls)
3. **Smooth Movement**: Move camera slowly for better tracking
4. **Feature Rich Areas**: Areas with corners, edges, and patterns work best

### Android Demo

1. **Device Orientation**: Hold device steady in landscape mode
2. **ARCore Calibration**: Allow ARCore to initialize tracking
3. **GPS Signal**: Ensure good GPS signal for outdoor use
4. **Permissions**: Grant camera and location permissions

## Troubleshooting

### Python Issues

- **"Could not open webcam"**: Check camera permissions and connectivity
- **Poor tracking**: Ensure good lighting and textured environment
- **Installation errors**: Verify virtual environment activation

### Android Issues

- **ARCore not supported**: Check device compatibility
- **Permission denied**: Grant camera and location permissions in settings
- **Build errors**: Ensure latest Android Studio and SDK versions

## Data Analysis

The generated CSV files can be analyzed using:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load trajectory data
df = pd.read_csv('trajectory.csv')

# Plot 2D trajectory
plt.figure(figsize=(10, 8))
plt.plot(df['x'], df['z'])
plt.xlabel('X (meters)')
plt.ylabel('Z (meters)')
plt.title('Camera Trajectory (Top View)')
plt.axis('equal')
plt.grid(True)
plt.show()
```

## Next Steps

1. **Camera Calibration**: Replace dummy intrinsics with calibrated parameters
2. **Loop Closure**: Add SLAM capabilities for drift correction
3. **Sensor Fusion**: Integrate IMU data for improved tracking
4. **Real GPS**: Replace stub with actual GPS integration in Python
5. **AR Visualization**: Add 3D trajectory rendering in Android app

## References

- OpenCV Visual Odometry: https://docs.opencv.org/
- ARCore Documentation: https://developers.google.com/ar
- Android Location Services: https://developer.android.com/training/location