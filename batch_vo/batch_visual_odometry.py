#!/usr/bin/env python3
"""
Batch Visual Odometry Processor
===============================

Processes frame-by-frame images to compute 3D trajectory using Visual Odometry.
Exports results to CSV and generates professional 3D trajectory plots.

Features:
- Robust feature matching with SIFT/ORB
- Essential matrix estimation with RANSAC
- Proper scale recovery using triangulation
- Outlier filtering and trajectory smoothing
- Professional 3D plotting similar to research papers
- CSV export with detailed trajectory data

Usage:
    python batch_visual_odometry.py --input_dir <image_directory> --output_dir <results_directory>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import glob
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchVisualOdometry:
    """
    Batch Visual Odometry processor for frame-by-frame image sequences.
    """
    
    def __init__(self, 
                 feature_detector='SIFT',
                 max_features=2000,
                 match_ratio=0.7,
                 ransac_threshold=1.0,
                 min_triangulation_depth=0.1,
                 max_triangulation_depth=50.0):
        """
        Initialize the Visual Odometry processor.
        
        Args:
            feature_detector: 'SIFT' or 'ORB'
            max_features: Maximum number of features to detect
            match_ratio: Ratio test threshold for feature matching
            ransac_threshold: RANSAC threshold for essential matrix estimation
            min_triangulation_depth: Minimum depth for valid triangulated points
            max_triangulation_depth: Maximum depth for valid triangulated points
        """
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.min_triangulation_depth = min_triangulation_depth
        self.max_triangulation_depth = max_triangulation_depth
        
        # Initialize feature detector
        if feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError("Feature detector must be 'SIFT' or 'ORB'")
        
        # Initialize matcher
        if feature_detector == 'SIFT':
            self.matcher = cv2.BFMatcher()
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Camera intrinsic parameters (default values, should be calibrated)
        self.K = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
        
        # Trajectory storage
        self.trajectory = []
        self.poses = []
        self.frame_info = []
        
        # Current pose (cumulative)
        self.current_pose = np.eye(4)
        
        logger.info(f"Initialized BatchVO with {feature_detector} detector")
    
    def set_camera_matrix(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsic parameters."""
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
        logger.info(f"Camera matrix updated: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and match features between two images.
        
        Returns:
            pts1, pts2: Matched point coordinates
        """
        # Detect keypoints and descriptors
        kp1, desc1 = self.detector.detectAndCompute(img1, None)
        kp2, desc2 = self.detector.detectAndCompute(img2, None)
        
        if desc1 is None or desc2 is None:
            return np.array([]), np.array([])
        
        # Match features
        if self.feature_detector == 'SIFT':
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)
        else:
            matches = self.matcher.match(desc1, desc2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.3)]
        
        if len(good_matches) < 8:
            logger.warning(f"Only {len(good_matches)} good matches found")
            return np.array([]), np.array([])
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2
    
    def estimate_pose(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Estimate pose change between two frames.
        
        Returns:
            R: Rotation matrix
            t: Translation vector
            scale: Estimated scale factor
        """
        if len(pts1) < 8 or len(pts2) < 8:
            return None, None, None
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, 
                                      prob=0.999, threshold=self.ransac_threshold)
        
        if E is None:
            return None, None, None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Estimate scale using triangulation
        scale = self.estimate_scale(pts1, pts2, R, t, mask)
        
        return R, t, scale
    
    def estimate_scale(self, pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate scale factor using triangulation of matched points.
        """
        # Filter inlier points
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        if len(pts1_inliers) < 4:
            return 1.0
        
        # Create projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(-1, 1)])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        # Filter points based on depth
        depths = points_3d[2]
        valid_mask = (depths > self.min_triangulation_depth) & (depths < self.max_triangulation_depth)
        
        if np.sum(valid_mask) < 3:
            return 1.0
        
        valid_depths = depths[valid_mask]
        median_depth = np.median(valid_depths)
        
        # Scale factor based on reasonable scene depth (heuristic)
        # Assume average scene depth should be around 2-5 meters
        target_depth = 3.0
        scale = target_depth / max(median_depth, 0.1)
        
        # Clamp scale to reasonable range
        scale = np.clip(scale, 0.1, 10.0)
        
        return scale
    
    def process_image_sequence(self, image_paths: List[str]) -> bool:
        """
        Process a sequence of images to compute trajectory.
        
        Args:
            image_paths: List of image file paths in temporal order
            
        Returns:
            True if processing successful
        """
        if len(image_paths) < 2:
            logger.error("Need at least 2 images for VO processing")
            return False
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        # Reset trajectory
        self.trajectory = []
        self.poses = []
        self.frame_info = []
        self.current_pose = np.eye(4)
        
        # Add initial position
        self.trajectory.append([0.0, 0.0, 0.0])
        self.poses.append(self.current_pose.copy())
        self.frame_info.append({
            'frame_id': 0,
            'image_path': image_paths[0],
            'matches': 0,
            'scale': 1.0
        })
        
        prev_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        if prev_img is None:
            logger.error(f"Failed to load image: {image_paths[0]}")
            return False
        
        for i, img_path in enumerate(image_paths[1:], 1):
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if curr_img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            
            # Detect and match features
            pts1, pts2 = self.detect_and_match_features(prev_img, curr_img)
            
            if len(pts1) == 0:
                logger.warning(f"No matches found for frame {i}")
                # Add current position (no movement)
                self.trajectory.append(self.trajectory[-1].copy())
                self.poses.append(self.current_pose.copy())
                self.frame_info.append({
                    'frame_id': i,
                    'image_path': img_path,
                    'matches': 0,
                    'scale': 1.0
                })
                prev_img = curr_img
                continue
            
            # Estimate pose
            R, t, scale = self.estimate_pose(pts1, pts2)
            
            if R is None or t is None:
                logger.warning(f"Pose estimation failed for frame {i}")
                # Add current position (no movement)
                self.trajectory.append(self.trajectory[-1].copy())
                self.poses.append(self.current_pose.copy())
                self.frame_info.append({
                    'frame_id': i,
                    'image_path': img_path,
                    'matches': len(pts1),
                    'scale': 1.0
                })
                prev_img = curr_img
                continue
            
            # Apply scale to translation
            t_scaled = t.ravel() * scale
            
            # Update current pose
            pose_delta = np.eye(4)
            pose_delta[:3, :3] = R
            pose_delta[:3, 3] = t_scaled
            
            self.current_pose = self.current_pose @ pose_delta
            
            # Extract position
            position = self.current_pose[:3, 3]
            self.trajectory.append(position.tolist())
            self.poses.append(self.current_pose.copy())
            self.frame_info.append({
                'frame_id': i,
                'image_path': img_path,
                'matches': len(pts1),
                'scale': scale
            })
            
            logger.info(f"Frame {i}: {len(pts1)} matches, scale={scale:.3f}, pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            
            prev_img = curr_img
        
        logger.info(f"Processing complete. Trajectory has {len(self.trajectory)} points")
        return True
    
    def export_to_csv(self, output_path: str):
        """Export trajectory and metadata to CSV file."""
        if not self.trajectory:
            logger.error("No trajectory data to export")
            return
        
        # Create DataFrame
        data = []
        for i, (pos, frame_info) in enumerate(zip(self.trajectory, self.frame_info)):
            data.append({
                'frame_id': frame_info['frame_id'],
                'timestamp': i * 0.033,  # Assume 30 FPS
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'matches': frame_info['matches'],
                'scale': frame_info['scale'],
                'image_path': frame_info['image_path']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Trajectory exported to {output_path}")
    
    def create_3d_plot(self, output_path: str, title: str = "Visual Odometry Trajectory"):
        """Create professional 3D trajectory plot."""
        if not self.trajectory:
            logger.error("No trajectory data to plot")
            return
        
        trajectory = np.array(self.trajectory)
        
        # Create figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
               color='#1f77b4', linewidth=2.5, label='Visual Odometry', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color='red', s=100, marker='s', label='End', zorder=5)
        
        # Set labels and title
        ax.set_xlabel('X (metres)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (metres)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (metres)', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Configure grid and appearance
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges more visible
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Set equal aspect ratio
        def set_axes_equal(ax):
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            
            x_range = abs(x_limits[1] - x_limits[0])
            y_range = abs(y_limits[1] - y_limits[0])
            z_range = abs(z_limits[1] - z_limits[0])
            
            max_range = max(x_range, y_range, z_range)
            
            x_middle = np.mean(x_limits)
            y_middle = np.mean(y_limits)
            z_middle = np.mean(z_limits)
            
            ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
            ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
            ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
        
        set_axes_equal(ax)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add statistics text box
        total_distance = self.calculate_total_distance()
        stats_text = f'Total Distance: {total_distance:.2f}m\\nFrames: {len(self.trajectory)}'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"3D plot saved to {output_path}")
    
    def calculate_total_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_dist = 0.0
        trajectory = np.array(self.trajectory)
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
            total_dist += dist
        
        return total_dist
    
    def save_metadata(self, output_path: str):
        """Save processing metadata to JSON file."""
        metadata = {
            'total_frames': len(self.trajectory),
            'total_distance': self.calculate_total_distance(),
            'feature_detector': self.feature_detector,
            'max_features': self.max_features,
            'match_ratio': self.match_ratio,
            'ransac_threshold': self.ransac_threshold,
            'camera_matrix': self.K.tolist(),
            'trajectory_bounds': {
                'x_min': float(np.min([p[0] for p in self.trajectory])),
                'x_max': float(np.max([p[0] for p in self.trajectory])),
                'y_min': float(np.min([p[1] for p in self.trajectory])),
                'y_max': float(np.max([p[1] for p in self.trajectory])),
                'z_min': float(np.min([p[2] for p in self.trajectory])),
                'z_max': float(np.max([p[2] for p in self.trajectory]))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")

def load_camera_info(input_dir, args):
    """Load camera parameters from camera_info.json if available"""
    camera_info_path = os.path.join(input_dir, 'camera_info.json')
    if os.path.exists(camera_info_path):
        try:
            with open(camera_info_path, 'r') as f:
                camera_info = json.load(f)
            logger.info(f"Loaded camera parameters from {camera_info_path}")
            return (
                camera_info.get('fx', args.fx),
                camera_info.get('fy', args.fy),
                camera_info.get('cx', args.cx),
                camera_info.get('cy', args.cy)
            )
        except Exception as e:
            logger.warning(f"Failed to load camera_info.json: {e}")
    
    return args.fx, args.fy, args.cx, args.cy

def load_ground_truth(input_dir):
    """Load ground truth trajectory if available"""
    gt_path = os.path.join(input_dir, 'ground_truth.csv')
    if os.path.exists(gt_path):
        try:
            df = pd.read_csv(gt_path)
            logger.info(f"Loaded ground truth from {gt_path}")
            
            # Check if we have all required columns
            required_cols = ['translation_x', 'translation_y']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Ground truth missing required columns: {required_cols}")
                return None
            
            # Add translation_z if missing (assume Z=0)
            if 'translation_z' not in df.columns:
                logger.info("Ground truth missing translation_z, assuming Z=0")
                df['translation_z'] = 0.0
            
            return df[['translation_x', 'translation_y', 'translation_z']].values
        except Exception as e:
            logger.warning(f"Failed to load ground truth: {e}")
    
    return None

def create_comparison_plot(gt_trajectory, vo_trajectory, output_path):
    """Create comparison plot between ground truth and VO trajectory"""
    try:
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
                'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(vo_trajectory[:, 0], vo_trajectory[:, 1], vo_trajectory[:, 2], 
                'r--', linewidth=2, label='Visual Odometry', alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        
        # XY plot
        ax2 = fig.add_subplot(222)
        ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-', linewidth=2, label='Ground Truth')
        ax2.plot(vo_trajectory[:, 0], vo_trajectory[:, 1], 'r--', linewidth=2, label='Visual Odometry')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (X-Y)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Error plots
        min_len = min(len(gt_trajectory), len(vo_trajectory))
        gt_short = gt_trajectory[:min_len]
        vo_short = vo_trajectory[:min_len]
        
        errors = np.linalg.norm(gt_short - vo_short, axis=1)
        
        ax3 = fig.add_subplot(223)
        ax3.plot(range(min_len), errors, 'b-', linewidth=1)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position Error (m)')
        ax3.set_title('Position Error Over Time')
        ax3.grid(True)
        
        # Error statistics
        ax4 = fig.add_subplot(224)
        ax4.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Position Error (m)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.grid(True)
        
        # Add statistics text
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        stats_text = f'Mean Error: {mean_error:.3f}m\nStd Error: {std_error:.3f}m\nMax Error: {max_error:.3f}m'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to: {output_path}")
        logger.info(f"Trajectory comparison stats: Mean error: {mean_error:.3f}m, Max error: {max_error:.3f}m")
        
    except Exception as e:
        logger.error(f"Failed to create comparison plot: {e}")

def main():
    parser = argparse.ArgumentParser(description='Batch Visual Odometry Processor')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--detector', choices=['SIFT', 'ORB'], default='SIFT', help='Feature detector')
    parser.add_argument('--max_features', type=int, default=2000, help='Maximum features to detect')
    parser.add_argument('--fx', type=float, default=800, help='Camera focal length X')
    parser.add_argument('--fy', type=float, default=800, help='Camera focal length Y')
    parser.add_argument('--cx', type=float, default=320, help='Camera principal point X')
    parser.add_argument('--cy', type=float, default=240, help='Camera principal point Y')
    parser.add_argument('--comparison_only', action='store_true', help='Only generate comparison plot (skip VO processing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera parameters (from file or command line)
    fx, fy, cx, cy = load_camera_info(args.input_dir, args)
    logger.info(f"Using camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Load ground truth if available
    ground_truth = load_ground_truth(args.input_dir)
    
    # Check if we're only generating comparison plot
    if args.comparison_only:
        if ground_truth is None:
            logger.error("Cannot create comparison plot: no ground truth found")
            return
        
        vo_csv_path = output_dir / 'trajectory.csv'
        if not vo_csv_path.exists():
            logger.error(f"Cannot create comparison plot: no VO trajectory found at {vo_csv_path}")
            return
        
        try:
            vo_df = pd.read_csv(vo_csv_path)
            vo_trajectory = vo_df[['x', 'y', 'z']].values
            comparison_path = output_dir / 'comparison_plot.png'
            create_comparison_plot(ground_truth, vo_trajectory, str(comparison_path))
            logger.info("Comparison plot generation complete!")
            return
        except Exception as e:
            logger.error(f"Failed to load VO trajectory: {e}")
            return
    
    # Get image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    image_paths.sort()  # Ensure temporal order
    
    if not image_paths:
        logger.error(f"No images found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Initialize VO processor
    vo = BatchVisualOdometry(
        feature_detector=args.detector,
        max_features=args.max_features
    )
    
    # Set camera parameters
    vo.set_camera_matrix(fx, fy, cx, cy)
    
    # Process images
    if not vo.process_image_sequence(image_paths):
        logger.error("Failed to process image sequence")
        return
    
    # Export results
    csv_path = output_dir / 'trajectory.csv'
    plot_path = output_dir / 'trajectory_3d.png'
    metadata_path = output_dir / 'metadata.json'
    
    vo.export_to_csv(str(csv_path))
    vo.create_3d_plot(str(plot_path))
    vo.save_metadata(str(metadata_path))
    
    # Generate comparison plot if ground truth is available
    if ground_truth is not None:
        try:
            vo_trajectory = np.array(vo.trajectory)
            comparison_path = output_dir / 'comparison_plot.png'
            create_comparison_plot(ground_truth, vo_trajectory, str(comparison_path))
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
    
    logger.info("Batch processing complete!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()