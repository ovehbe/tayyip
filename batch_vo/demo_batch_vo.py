#!/usr/bin/env python3
"""
Demo script for Batch Visual Odometry processing.

This script provides a simple interface to run batch VO on your image sequences.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def check_requirements():
    """Check if required packages are installed."""
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install requirements with: pip install -r requirements.txt")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Sample Camera Configuration
# Adjust these values based on your camera calibration

# Camera focal lengths (pixels)
fx = 800.0
fy = 800.0

# Camera principal point (pixels)
cx = 320.0
cy = 240.0

# Feature detection parameters
detector = SIFT
max_features = 2000

# Matching parameters
match_ratio = 0.7
ransac_threshold = 1.0

# Triangulation depth limits (meters)
min_depth = 0.1
max_depth = 50.0
"""
    
    with open('camera_config.txt', 'w') as f:
        f.write(config_content)
    
    print("📝 Created camera_config.txt - adjust parameters for your camera")

def run_batch_vo():
    """Interactive demo to run batch VO."""
    print("🚀 Batch Visual Odometry Demo")
    print("=" * 40)
    
    if not check_requirements():
        return
    
    # Get input directory
    while True:
        input_dir = input("📁 Enter path to your image sequence directory: ").strip()
        if os.path.isdir(input_dir):
            break
        print("❌ Directory not found. Please try again.")
    
    # Check for images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    import glob
    image_count = 0
    for ext in image_extensions:
        image_count += len(glob.glob(os.path.join(input_dir, ext)))
        image_count += len(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if image_count < 2:
        print(f"❌ Found only {image_count} images. Need at least 2 for VO processing.")
        return
    
    print(f"✅ Found {image_count} images")
    
    # Get output directory
    output_dir = input("📂 Enter output directory (default: results): ").strip()
    if not output_dir:
        output_dir = "results"
    
    # Camera parameters
    print("\n📷 Camera Configuration:")
    fx = float(input("Focal length X (default: 800): ") or "800")
    fy = float(input("Focal length Y (default: 800): ") or "800")
    cx = float(input("Principal point X (default: 320): ") or "320")
    cy = float(input("Principal point Y (default: 240): ") or "240")
    
    # Feature detector
    detector = input("Feature detector (SIFT/ORB, default: SIFT): ").strip().upper()
    if detector not in ['SIFT', 'ORB']:
        detector = 'SIFT'
    
    # Build command
    cmd = [
        sys.executable, 'batch_visual_odometry.py',
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--detector', detector,
        '--fx', str(fx),
        '--fy', str(fy),
        '--cx', str(cx),
        '--cy', str(cy)
    ]
    
    print(f"\n🔄 Running: {' '.join(cmd)}")
    print("Processing...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Processing completed successfully!")
            print(f"📊 Results saved to: {output_dir}")
            print(f"   • trajectory.csv - Raw trajectory data")
            print(f"   • trajectory_3d.png - 3D visualization")
            print(f"   • metadata.json - Processing metadata")
            
            # Show trajectory summary
            try:
                import pandas as pd
                df = pd.read_csv(os.path.join(output_dir, 'trajectory.csv'))
                total_distance = ((df['x'].diff()**2 + df['y'].diff()**2 + df['z'].diff()**2)**0.5).sum()
                print(f"\n📈 Trajectory Summary:")
                print(f"   • Total frames: {len(df)}")
                print(f"   • Total distance: {total_distance:.2f} meters")
                print(f"   • X range: {df['x'].min():.2f} to {df['x'].max():.2f} m")
                print(f"   • Y range: {df['y'].min():.2f} to {df['y'].max():.2f} m")
                print(f"   • Z range: {df['z'].min():.2f} to {df['z'].max():.2f} m")
            except:
                pass
            
        else:
            print("❌ Processing failed!")
            print("Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running processing: {e}")

def main():
    parser = argparse.ArgumentParser(description='Batch VO Demo')
    parser.add_argument('--create-config', action='store_true', help='Create sample config file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
    else:
        run_batch_vo()

if __name__ == '__main__':
    main()