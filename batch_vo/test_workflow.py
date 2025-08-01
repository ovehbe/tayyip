#!/usr/bin/env python3
"""
Complete test workflow for Batch Visual Odometry.

This script demonstrates the full pipeline:
1. Generate synthetic test images
2. Run VO processing
3. Analyze and display results
"""

import os
import sys
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_test_workflow():
    """Run complete test workflow."""
    print("🧪 Batch Visual Odometry - Complete Test Workflow")
    print("=" * 50)
    
    # Step 1: Generate test images
    print("\n📷 Step 1: Generating synthetic test images...")
    test_dir = "test_images"
    
    cmd_generate = [
        sys.executable, "generate_test_images.py",
        "--num_frames", "30",
        "--output_dir", test_dir,
        "--width", "640",
        "--height", "480",
        "--fx", "800",
        "--fy", "800"
    ]
    
    try:
        result = subprocess.run(cmd_generate, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Test image generation failed: {result.stderr}")
            return False
        print("✅ Test images generated successfully")
    except Exception as e:
        print(f"❌ Error generating images: {e}")
        return False
    
    # Step 2: Run VO processing
    print("\n🔄 Step 2: Running Visual Odometry processing...")
    results_dir = "test_results"
    
    cmd_vo = [
        sys.executable, "batch_visual_odometry.py",
        "--input_dir", test_dir,
        "--output_dir", results_dir,
        "--detector", "SIFT",
        "--max_features", "1000",
        "--fx", "800",
        "--fy", "800",
        "--cx", "320",
        "--cy", "240"
    ]
    
    try:
        result = subprocess.run(cmd_vo, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ VO processing failed: {result.stderr}")
            return False
        print("✅ VO processing completed successfully")
    except Exception as e:
        print(f"❌ Error running VO: {e}")
        return False
    
    # Step 3: Analyze results
    print("\n📊 Step 3: Analyzing results...")
    try:
        # Load trajectories
        gt_path = os.path.join(test_dir, "ground_truth.csv")
        vo_path = os.path.join(results_dir, "trajectory.csv")
        
        if not os.path.exists(gt_path) or not os.path.exists(vo_path):
            print("❌ Missing trajectory files")
            return False
        
        gt_data = pd.read_csv(gt_path)
        vo_data = pd.read_csv(vo_path)
        
        # Compute basic statistics
        gt_distance = ((gt_data['x'].diff()**2 + gt_data['y'].diff()**2 + gt_data['z'].diff()**2)**0.5).sum()
        vo_distance = ((vo_data['x'].diff()**2 + vo_data['y'].diff()**2 + vo_data['z'].diff()**2)**0.5).sum()
        
        print(f"   📏 Ground Truth Distance: {gt_distance:.2f} m")
        print(f"   📏 VO Estimated Distance: {vo_distance:.2f} m")
        print(f"   📐 Distance Error: {abs(gt_distance - vo_distance):.2f} m ({abs(gt_distance - vo_distance)/gt_distance*100:.1f}%)")
        
        # Compute trajectory error (simplified)
        if len(gt_data) == len(vo_data):
            position_errors = ((gt_data['x'] - vo_data['x'])**2 + 
                             (gt_data['y'] - vo_data['y'])**2 + 
                             (gt_data['z'] - vo_data['z'])**2)**0.5
            mean_error = position_errors.mean()
            max_error = position_errors.max()
            
            print(f"   📍 Mean Position Error: {mean_error:.3f} m")
            print(f"   📍 Max Position Error: {max_error:.3f} m")
        
        print("✅ Analysis completed")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False
    
    # Step 4: Create comparison plot
    print("\n📈 Step 4: Creating comparison visualization...")
    try:
        create_comparison_plot(gt_path, vo_path, os.path.join(results_dir, "comparison_plot.png"))
        print("✅ Comparison plot created")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        return False
    
    # Step 5: Summary
    print("\n🎉 Test Workflow Complete!")
    print("=" * 30)
    print(f"📁 Test images: {test_dir}/")
    print(f"📊 VO results: {results_dir}/")
    print(f"   • trajectory.csv - VO trajectory data")
    print(f"   • trajectory_3d.png - VO 3D visualization")
    print(f"   • comparison_plot.png - GT vs VO comparison")
    print(f"   • metadata.json - Processing metadata")
    
    return True

def create_comparison_plot(gt_path, vo_path, output_path):
    """Create comparison plot between ground truth and VO."""
    # Load data
    gt_data = pd.read_csv(gt_path)
    vo_data = pd.read_csv(vo_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory comparison
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(gt_data['x'], gt_data['y'], gt_data['z'], 
            color='orange', linewidth=2.5, label='Ground Truth', alpha=0.8)
    ax1.plot(vo_data['x'], vo_data['y'], vo_data['z'], 
            color='blue', linewidth=2.5, label='Visual Odometry', alpha=0.8)
    
    ax1.set_xlabel('X (metres)')
    ax1.set_ylabel('Y (metres)')
    ax1.set_zlabel('Z (metres)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # X-Y top view
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_data['x'], gt_data['y'], 'o-', color='orange', 
            linewidth=2, markersize=3, label='Ground Truth')
    ax2.plot(vo_data['x'], vo_data['y'], 'o-', color='blue', 
            linewidth=2, markersize=3, label='Visual Odometry')
    ax2.set_xlabel('X (metres)')
    ax2.set_ylabel('Y (metres)')
    ax2.set_title('Top View (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Position error over time
    if len(gt_data) == len(vo_data):
        ax3 = fig.add_subplot(223)
        position_error = ((gt_data['x'] - vo_data['x'])**2 + 
                         (gt_data['y'] - vo_data['y'])**2 + 
                         (gt_data['z'] - vo_data['z'])**2)**0.5
        
        ax3.plot(gt_data['timestamp'], position_error, 'r-', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Position Error (metres)')
        ax3.set_title('Position Error Over Time')
        ax3.grid(True, alpha=0.3)
    
    # Distance comparison
    ax4 = fig.add_subplot(224)
    gt_cumulative = ((gt_data['x'].diff()**2 + gt_data['y'].diff()**2 + gt_data['z'].diff()**2)**0.5).fillna(0).cumsum()
    vo_cumulative = ((vo_data['x'].diff()**2 + vo_data['y'].diff()**2 + vo_data['z'].diff()**2)**0.5).fillna(0).cumsum()
    
    ax4.plot(gt_data['timestamp'], gt_cumulative, color='orange', linewidth=2, label='Ground Truth')
    ax4.plot(vo_data['timestamp'], vo_cumulative, color='blue', linewidth=2, label='Visual Odometry')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Cumulative Distance (metres)')
    ax4.set_title('Cumulative Distance Traveled')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main test function."""
    if not run_test_workflow():
        print("\n❌ Test workflow failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")

if __name__ == '__main__':
    main()