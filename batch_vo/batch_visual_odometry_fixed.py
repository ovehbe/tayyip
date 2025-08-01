#!/usr/bin/env python3
"""
FIXED Batch Visual Odometry with Correct Coordinate System and Relaxed Matching

Key Fixes:
1. Coordinate system transformation (Camera Z->World X for forward motion)
2. Relaxed matching criteria to prevent "insufficient matches"
3. Adaptive scale normalization instead of rigid scaling
4. Better parameter tuning for handheld indoor footage
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import json
import pandas as pd
import argparse
import logging
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedBatchVisualOdometry:
    def __init__(self, feature_detector='SIFT', max_features=2000):
        """
        Initialize FIXED Visual Odometry processor
        """
        self.feature_detector_name = feature_detector
        self.max_features = max_features
        
        # Initialize feature detector
        if feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unsupported detector: {feature_detector}")
        
        # Feature matcher
        if feature_detector == 'SIFT':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera parameters
        self.camera_matrix = None
        
        # Trajectory storage (in WORLD coordinates: X=forward, Y=right, Z=up)
        self.trajectory = []
        self.poses = []
        self.keyframes = []
        self.keyframe_descriptors = []
        
        # RELAXED processing parameters for handheld indoor footage
        self.match_ratio = 0.8  # More lenient ratio test (was 0.75)
        self.ransac_threshold = 1.5  # More lenient RANSAC (was 1.0)
        self.min_matches = 20  # Fewer matches required (was 50)
        self.min_inlier_ratio = 0.2  # Lower inlier requirement (was 0.3)
        
        # ADAPTIVE scale parameters
        self.base_scale = 0.05  # Base scale factor (5cm)
        self.scale_adaptation = True
        self.scale_history = deque(maxlen=10)  # Track recent scales
        self.max_translation = 0.5  # More realistic max per frame (was 2.0)
        
        # Smoothing
        self.smoothing_window = 3  # Smaller window for responsiveness
        
        # Loop closure (less aggressive)
        self.keyframe_interval = 20  # Less frequent keyframes
        self.loop_threshold = 50  # Fewer matches needed for loop
        
        # Current state
        self.current_pose = np.eye(4)
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Statistics
        self.total_matches = []
        self.inlier_ratios = []
        self.translation_magnitudes = []
        self.skipped_frames = 0
        
    def set_camera_matrix(self, fx, fy, cx, cy):
        """Set camera intrinsic parameters"""
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        logger.info(f"Set camera matrix: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    def camera_to_world_coords(self, camera_translation):
        """
        Transform camera coordinates to world coordinates
        
        Camera coordinates (OpenCV):
        - X: right
        - Y: down  
        - Z: forward (into scene)
        
        World coordinates (trajectory):
        - X: forward motion
        - Y: right/left strafe
        - Z: up/down (minimal for ground robot)
        """
        # Rotate camera coordinates to world coordinates
        # Camera Z -> World X (forward)
        # Camera X -> World Y (right)  
        # Camera -Y -> World Z (up, negated because camera Y is down)
        
        world_x = camera_translation[2]   # Camera Z becomes World X (forward)
        world_y = camera_translation[0]   # Camera X becomes World Y (right)
        world_z = -camera_translation[1]  # Camera -Y becomes World Z (up)
        
        return np.array([world_x, world_y, world_z])
    
    def robust_feature_matching(self, desc1, desc2):
        """RELAXED feature matching for challenging indoor footage"""
        if desc1 is None or desc2 is None or len(desc1) < 5 or len(desc2) < 5:
            return []
        
        try:
            # Get matches
            if self.feature_detector_name == 'SIFT':
                raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            else:
                raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            logger.warning("Feature matching failed")
            return []
        
        # Apply RELAXED Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Include single matches if we're desperate
                good_matches.append(match_pair[0])
        
        logger.debug(f"Ratio test: {len(raw_matches)} -> {len(good_matches)} matches")
        return good_matches
    
    def estimate_pose_robust(self, kp1, kp2, matches):
        """RELAXED pose estimation for challenging conditions"""
        if len(matches) < self.min_matches:
            logger.debug(f"Insufficient matches: {len(matches)} < {self.min_matches}")
            return None, None, 0
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix with RELAXED RANSAC
        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, 
                self.camera_matrix, 
                method=cv2.RANSAC,
                prob=0.99,  # Slightly lower confidence for relaxed matching
                threshold=self.ransac_threshold
            )
        except cv2.error:
            logger.debug("Essential matrix estimation failed")
            return None, None, 0
        
        if E is None or mask is None:
            return None, None, 0
        
        inlier_ratio = np.sum(mask) / len(mask)
        
        if inlier_ratio < self.min_inlier_ratio:
            logger.debug(f"Low inlier ratio: {inlier_ratio:.2f}")
            return None, None, inlier_ratio
        
        # Recover pose
        try:
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        except cv2.error:
            logger.debug("Pose recovery failed")
            return None, None, inlier_ratio
        
        return R, t, inlier_ratio
    
    def adaptive_scale_normalization(self, translation):
        """Adaptive scale normalization that preserves motion variability"""
        t_norm = np.linalg.norm(translation)
        
        if t_norm < 1e-6:
            return translation
        
        # Calculate adaptive scale based on recent motion
        if len(self.scale_history) > 0:
            recent_avg_scale = np.mean(self.scale_history)
            adaptive_scale = 0.7 * self.base_scale + 0.3 * recent_avg_scale
        else:
            adaptive_scale = self.base_scale
        
        # Normalize but preserve relative motion variability
        if t_norm > self.max_translation:
            # Cap excessive motion
            normalized_t = (translation / t_norm) * self.max_translation
            logger.debug(f"Capped large translation: {t_norm:.3f} -> {self.max_translation:.3f}")
        else:
            # Scale naturally but preserve relative magnitude
            scale_factor = adaptive_scale * (1.0 + 0.5 * (t_norm - 0.1))  # Slight motion-dependent scaling
            normalized_t = (translation / t_norm) * scale_factor
        
        # Track scale for adaptation
        actual_scale = np.linalg.norm(normalized_t)
        self.scale_history.append(actual_scale)
        
        return normalized_t
    
    def smooth_trajectory(self):
        """Apply LIGHT trajectory smoothing"""
        if len(self.trajectory) < self.smoothing_window:
            return
        
        # Light smoothing of last few positions
        start_idx = max(0, len(self.trajectory) - self.smoothing_window)
        positions = np.array(self.trajectory[start_idx:])
        
        if len(positions) >= 2:
            # Weighted average favoring recent positions
            weights = np.linspace(0.5, 1.0, len(positions))
            smoothed = np.average(positions, axis=0, weights=weights)
            self.trajectory[-1] = smoothed
    
    def detect_loop_closure(self, current_descriptors):
        """RELAXED loop closure detection"""
        if len(self.keyframe_descriptors) < 3:
            return None
        
        best_match_idx = -1
        best_match_count = 0
        
        # Only check distant keyframes
        for i, kf_desc in enumerate(self.keyframe_descriptors[:-1]):
            if kf_desc is None:
                continue
                
            matches = self.robust_feature_matching(current_descriptors, kf_desc)
            
            if len(matches) > best_match_count and len(matches) > self.loop_threshold:
                best_match_count = len(matches)
                best_match_idx = i
        
        if best_match_idx >= 0:
            logger.info(f"Loop closure detected with keyframe {best_match_idx} ({best_match_count} matches)")
            return best_match_idx
        
        return None
    
    def correct_loop_closure(self, loop_keyframe_idx):
        """Apply GENTLE loop closure correction"""
        if loop_keyframe_idx < 0 or loop_keyframe_idx >= len(self.keyframes):
            return
        
        # Gentle correction
        loop_position = self.keyframes[loop_keyframe_idx]
        current_position = self.trajectory[-1]
        
        # Light weighted correction
        correction_weight = 0.1  # Very gentle (was 0.3)
        corrected_position = (1 - correction_weight) * current_position + correction_weight * loop_position
        
        self.trajectory[-1] = corrected_position
        logger.info(f"Applied gentle loop closure: {np.linalg.norm(current_position - corrected_position):.3f}m")
    
    def process_image_sequence(self, image_paths):
        """Process sequence with RELAXED parameters for indoor handheld footage"""
        logger.info(f"Processing {len(image_paths)} images with RELAXED parameters...")
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                logger.info(f"Processing frame {i+1}/{len(image_paths)} (skipped: {self.skipped_frames})")
            
            # Load image
            frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            
            # Detect features
            try:
                keypoints, descriptors = self.detector.detectAndCompute(frame, None)
            except cv2.error:
                logger.warning(f"Feature detection failed for frame {i}")
                continue
            
            if len(keypoints) < 5:  # Very low threshold
                logger.warning(f"Very few features in frame {i}: {len(keypoints)}")
                self.skipped_frames += 1
                continue
            
            # First frame
            if self.prev_frame is None:
                self.prev_frame = frame
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                
                # Initialize trajectory at origin
                initial_position = np.array([0.0, 0.0, 0.0])
                self.trajectory.append(initial_position)
                self.poses.append(self.current_pose.copy())
                
                # First keyframe
                self.keyframes.append(initial_position)
                self.keyframe_descriptors.append(descriptors)
                continue
            
            # Match features
            matches = self.robust_feature_matching(self.prev_descriptors, descriptors)
            self.total_matches.append(len(matches))
            
            if len(matches) < self.min_matches:
                logger.debug(f"Insufficient matches in frame {i}: {len(matches)}")
                # Don't skip - use previous position instead
                self.trajectory.append(self.trajectory[-1].copy())
                self.poses.append(self.poses[-1].copy())
                self.skipped_frames += 1
                
                # Still update for next iteration
                self.prev_frame = frame
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                continue
            
            # Estimate pose
            R, t, inlier_ratio = self.estimate_pose_robust(
                self.prev_keypoints, keypoints, matches
            )
            
            self.inlier_ratios.append(inlier_ratio)
            
            if R is None or t is None:
                logger.debug(f"Pose estimation failed for frame {i}")
                # Use previous pose
                self.trajectory.append(self.trajectory[-1].copy())
                self.poses.append(self.poses[-1].copy())
                self.skipped_frames += 1
            else:
                # COORDINATE SYSTEM FIX: Transform camera coords to world coords
                camera_translation = t.flatten()
                world_translation = self.camera_to_world_coords(camera_translation)
                
                # Apply adaptive scale normalization
                world_translation_scaled = self.adaptive_scale_normalization(world_translation)
                self.translation_magnitudes.append(np.linalg.norm(world_translation_scaled))
                
                # Update pose in WORLD coordinates
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = world_translation_scaled
                
                self.current_pose = self.current_pose @ transform
                
                # Extract position
                position = self.current_pose[:3, 3]
                self.trajectory.append(position)
                self.poses.append(self.current_pose.copy())
                
                # Apply light smoothing
                self.smooth_trajectory()
                
                # Keyframes for loop closure
                if i % self.keyframe_interval == 0 and len(self.keyframes) > 0:
                    self.keyframes.append(position.copy())
                    self.keyframe_descriptors.append(descriptors)
                    
                    # Check for loop closure
                    loop_idx = self.detect_loop_closure(descriptors)
                    if loop_idx is not None:
                        self.correct_loop_closure(loop_idx)
            
            # Update for next iteration
            self.prev_frame = frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
        
        logger.info(f"Processing complete! Generated {len(self.trajectory)} poses (skipped {self.skipped_frames} frames)")
        return True
    
    def export_to_csv(self, output_path):
        """Export trajectory to CSV file"""
        df = pd.DataFrame(self.trajectory, columns=['x', 'y', 'z'])
        df['frame'] = range(len(self.trajectory))
        df.to_csv(output_path, index=False)
        logger.info(f"Trajectory exported to: {output_path}")
    
    def create_3d_plot(self, output_path):
        """Create 3D trajectory visualization with CORRECT axes"""
        if len(self.trajectory) < 2:
            logger.warning("Insufficient trajectory data for plotting")
            return
        
        trajectory = np.array(self.trajectory)
        total_distance = np.sum([
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory)-1)
        ])
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory with CORRECTED coordinate labels
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.8, label='Visual Odometry')
        
        # Mark start and end
        ax.scatter(*trajectory[0], color='green', s=100, label='Start')
        ax.scatter(*trajectory[-1], color='red', s=100, label='End')
        
        # Mark keyframes
        if self.keyframes:
            keyframes = np.array(self.keyframes)
            ax.scatter(keyframes[:, 0], keyframes[:, 1], keyframes[:, 2], 
                      color='orange', s=50, alpha=0.6, label='Keyframes')
        
        # CORRECT axis labels for world coordinates
        ax.set_xlabel('X (metres) - Forward')
        ax.set_ylabel('Y (metres) - Right')
        ax.set_zlabel('Z (metres) - Up')
        ax.set_title(f'FIXED Visual Odometry Trajectory\nTotal Distance: {total_distance:.2f}m, Frames: {len(trajectory)}')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D plot saved to: {output_path}")
    
    def save_metadata(self, output_path):
        """Save processing metadata"""
        trajectory = np.array(self.trajectory)
        total_distance = np.sum([
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory)-1)
        ]) if len(trajectory) > 1 else 0.0
        
        metadata = {
            'total_frames': len(self.trajectory),
            'skipped_frames': self.skipped_frames,
            'total_distance': total_distance,
            'feature_detector': self.feature_detector_name,
            'max_features': self.max_features,
            'match_ratio': self.match_ratio,
            'ransac_threshold': self.ransac_threshold,
            'min_matches': self.min_matches,
            'base_scale': self.base_scale,
            'coordinate_system': 'WORLD (X=forward, Y=right, Z=up)',
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'trajectory_bounds': {
                'x_min': float(np.min(trajectory[:, 0])) if len(trajectory) > 0 else 0,
                'x_max': float(np.max(trajectory[:, 0])) if len(trajectory) > 0 else 0,
                'y_min': float(np.min(trajectory[:, 1])) if len(trajectory) > 0 else 0,
                'y_max': float(np.max(trajectory[:, 1])) if len(trajectory) > 0 else 0,
                'z_min': float(np.min(trajectory[:, 2])) if len(trajectory) > 0 else 0,
                'z_max': float(np.max(trajectory[:, 2])) if len(trajectory) > 0 else 0,
            },
            'statistics': {
                'avg_matches': float(np.mean(self.total_matches)) if self.total_matches else 0,
                'avg_inlier_ratio': float(np.mean(self.inlier_ratios)) if self.inlier_ratios else 0,
                'avg_translation': float(np.mean(self.translation_magnitudes)) if self.translation_magnitudes else 0,
                'max_translation': float(np.max(self.translation_magnitudes)) if self.translation_magnitudes else 0,
                'total_keyframes': len(self.keyframes),
                'success_rate': (len(self.trajectory) - self.skipped_frames) / len(self.trajectory) if len(self.trajectory) > 0 else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {output_path}")

# Load functions remain the same
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
            
            required_cols = ['translation_x', 'translation_y']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Ground truth missing required columns: {required_cols}")
                return None
            
            if 'translation_z' not in df.columns:
                logger.info("Ground truth missing translation_z, assuming Z=0")
                df['translation_z'] = 0.0
            
            return df[['translation_x', 'translation_y', 'translation_z']].values
        except Exception as e:
            logger.warning(f"Failed to load ground truth: {e}")
    
    return None

def create_comparison_plot(gt_trajectory, vo_trajectory, output_path):
    """Create comparison plot with CORRECT coordinate system"""
    try:
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
                'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(vo_trajectory[:, 0], vo_trajectory[:, 1], vo_trajectory[:, 2], 
                'r--', linewidth=2, label='Visual Odometry', alpha=0.8)
        ax1.set_xlabel('X (m) - Forward')
        ax1.set_ylabel('Y (m) - Right')
        ax1.set_zlabel('Z (m) - Up')
        ax1.set_title('3D Trajectory Comparison (FIXED Coordinates)')
        ax1.legend()
        
        # XY plot (forward vs right)
        ax2 = fig.add_subplot(222)
        ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-', linewidth=2, label='Ground Truth')
        ax2.plot(vo_trajectory[:, 0], vo_trajectory[:, 1], 'r--', linewidth=2, label='Visual Odometry')
        ax2.set_xlabel('X (m) - Forward')
        ax2.set_ylabel('Y (m) - Right')
        ax2.set_title('Top View (Forward vs Right)')
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
        
        # Stats
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        stats_text = f'Mean Error: {mean_error:.3f}m\nStd Error: {std_error:.3f}m\nMax Error: {max_error:.3f}m'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"FIXED comparison plot saved to: {output_path}")
        logger.info(f"Error stats: Mean: {mean_error:.3f}m, Max: {max_error:.3f}m")
        
    except Exception as e:
        logger.error(f"Failed to create comparison plot: {e}")

def main():
    parser = argparse.ArgumentParser(description='FIXED Batch Visual Odometry Processor')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--detector', choices=['SIFT', 'ORB'], default='SIFT', help='Feature detector')
    parser.add_argument('--max_features', type=int, default=2000, help='Maximum features to detect')
    parser.add_argument('--fx', type=float, default=800, help='Camera focal length X')
    parser.add_argument('--fy', type=float, default=800, help='Camera focal length Y')
    parser.add_argument('--cx', type=float, default=320, help='Camera principal point X')
    parser.add_argument('--cy', type=float, default=240, help='Camera principal point Y')
    parser.add_argument('--scale', type=float, default=0.05, help='Base scale factor (meters per unit)')
    parser.add_argument('--comparison_only', action='store_true', help='Only generate comparison plot')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera parameters and ground truth
    fx, fy, cx, cy = load_camera_info(args.input_dir, args)
    logger.info(f"Using camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    ground_truth = load_ground_truth(args.input_dir)
    
    # Comparison only mode
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
    
    image_paths.sort()
    
    if not image_paths:
        logger.error(f"No images found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Initialize FIXED VO processor
    vo = FixedBatchVisualOdometry(
        feature_detector=args.detector,
        max_features=args.max_features
    )
    
    # Set parameters
    vo.set_camera_matrix(fx, fy, cx, cy)
    vo.base_scale = args.scale
    
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
    
    # Generate comparison plot if available
    if ground_truth is not None:
        try:
            vo_trajectory = np.array(vo.trajectory)
            comparison_path = output_dir / 'comparison_plot.png'
            create_comparison_plot(ground_truth, vo_trajectory, str(comparison_path))
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
    
    logger.info("FIXED batch processing complete!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()