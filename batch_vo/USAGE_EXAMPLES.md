# Batch Visual Odometry - Usage Examples

## Quick Start

### 1. Setup Environment
```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh

# Activate the environment (do this every time you use the tool)
source venv/bin/activate
```

### 2. Prepare Your Images

**Image Requirements:**
- Sequential images from a moving camera
- Good overlap between consecutive frames (70-80%)
- Rich visual texture for feature matching
- Consistent lighting conditions
- Minimal motion blur

**Naming Convention:**
- Files should be sortable in temporal order
- Examples: `IMG_001.jpg`, `IMG_002.jpg`, etc.
- Or: `frame_0001.png`, `frame_0002.png`, etc.

### 3. Basic Usage

#### Interactive Demo (Recommended for beginners)
```bash
python demo_batch_vo.py
```
Follow the prompts to:
- Select your image directory
- Configure camera parameters
- Process and visualize results

#### Command Line Usage
```bash
python batch_visual_odometry.py \
  --input_dir /path/to/your/images \
  --output_dir results \
  --detector SIFT \
  --fx 800 --fy 800 --cx 320 --cy 240
```

## Real-World Examples

### Example 1: Phone Video Frames
If you have a video from your phone, extract frames first:

```bash
# Extract frames from video using ffmpeg
ffmpeg -i your_video.mp4 -vf fps=5 -q:v 2 frames/frame_%04d.jpg

# Then process with VO
python batch_visual_odometry.py \
  --input_dir frames/ \
  --output_dir phone_results/ \
  --detector SIFT \
  --fx 800 --fy 800 --cx 320 --cy 240
```

### Example 2: DSLR Image Sequence
For high-quality DSLR images:

```bash
python batch_visual_odometry.py \
  --input_dir dslr_images/ \
  --output_dir dslr_results/ \
  --detector SIFT \
  --max_features 3000 \
  --fx 1200 --fy 1200 --cx 640 --cy 480
```

### Example 3: Action Camera (GoPro, etc.)
For wide-angle action cameras:

```bash
python batch_visual_odometry.py \
  --input_dir gopro_images/ \
  --output_dir gopro_results/ \
  --detector ORB \
  --max_features 2000 \
  --fx 400 --fy 400 --cx 320 --cy 240
```

## Camera Parameter Guidelines

### Finding Your Camera Parameters

**Option 1: Use defaults (quick start)**
- Most cameras: `fx=800, fy=800, cx=320, cy=240` (for 640x480 images)

**Option 2: Estimate from image size**
```python
# Rule of thumb: focal length ≈ image_width
image_width = 1280  # your image width
fx = fy = image_width
cx = image_width / 2
cy = image_height / 2
```

**Option 3: Camera calibration (most accurate)**
Use OpenCV's camera calibration with a checkerboard pattern.

### Common Camera Types

| Camera Type | Typical Parameters (640x480) |
|-------------|------------------------------|
| Phone camera | fx=800, fy=800, cx=320, cy=240 |
| Webcam | fx=600, fy=600, cx=320, cy=240 |
| DSLR | fx=1200, fy=1200, cx=320, cy=240 |
| Action camera | fx=400, fy=400, cx=320, cy=240 |

## Output Analysis

### Generated Files

1. **`trajectory.csv`** - Raw trajectory data
   - `x, y, z`: Position in meters
   - `matches`: Number of feature matches per frame
   - `scale`: Estimated scale factor

2. **`trajectory_3d.png`** - 3D visualization
   - Professional plot suitable for presentations
   - Shows complete path with start/end markers

3. **`metadata.json`** - Processing details
   - Total distance traveled
   - Processing parameters
   - Trajectory bounds

### Quality Assessment

**Good Results Indicators:**
- Smooth trajectory without sudden jumps
- High number of feature matches (>100 per frame)
- Consistent scale factors (0.5-2.0 range)

**Troubleshooting Poor Results:**
- **Few matches**: Increase max_features, improve lighting
- **Jumpy trajectory**: Lower match_ratio, use SIFT instead of ORB
- **Wrong scale**: Check camera parameters, ensure good image overlap

## Advanced Usage

### Batch Processing Multiple Sequences
```bash
# Process multiple directories
for dir in sequence_*; do
    echo "Processing $dir..."
    python batch_visual_odometry.py \
      --input_dir "$dir" \
      --output_dir "results_$(basename $dir)" \
      --detector SIFT
done
```

### Custom Feature Detector Settings
```bash
# High accuracy (slower)
python batch_visual_odometry.py \
  --input_dir images/ \
  --output_dir results/ \
  --detector SIFT \
  --max_features 5000

# Fast processing (lower accuracy)
python batch_visual_odometry.py \
  --input_dir images/ \
  --output_dir results/ \
  --detector ORB \
  --max_features 1000
```

### Comparing with Ground Truth
If you have GPS or motion capture ground truth:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load trajectories
vo_data = pd.read_csv('results/trajectory.csv')
gt_data = pd.read_csv('ground_truth.csv')

# Plot comparison
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt_data['x'], gt_data['y'], gt_data['z'], 
        'o-', color='orange', label='Ground Truth')
ax.plot(vo_data['x'], vo_data['y'], vo_data['z'], 
        'o-', color='blue', label='Visual Odometry')

ax.legend()
plt.show()
```

## Performance Tips

1. **Image Size**: Resize large images (>1920px) for faster processing
2. **Frame Rate**: Skip frames if you have high frame rate video (use every 2nd or 3rd frame)
3. **Memory**: For long sequences, process in batches of 100-200 frames
4. **Feature Detector**: Use SIFT for accuracy, ORB for speed

## Common Issues and Solutions

### Issue: "No matches found"
**Causes:**
- Images lack texture/features
- Camera moved too fast
- Poor lighting conditions

**Solutions:**
- Increase max_features
- Use slower camera movement
- Improve lighting
- Try different feature detector

### Issue: "Trajectory looks wrong"
**Causes:**
- Incorrect camera parameters
- Poor image quality
- Scale estimation errors

**Solutions:**
- Calibrate camera properly
- Check image overlap
- Adjust triangulation depth limits

### Issue: "Processing is slow"
**Causes:**
- Large images
- Too many features
- High number of frames

**Solutions:**
- Resize images to 640-1280px width
- Reduce max_features to 1000-2000
- Use ORB instead of SIFT