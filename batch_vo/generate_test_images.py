#!/usr/bin/env python3
"""
Generate synthetic test images for Visual Odometry testing.

Creates a sequence of images simulating camera movement through a textured environment.
Useful for testing the batch VO pipeline without real camera data.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import math

class TestImageGenerator:
    """Generate synthetic images for VO testing."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.pattern = self.create_base_pattern()
    
    def create_base_pattern(self):
        """Create a rich textured pattern for feature detection."""
        pattern = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        # Create checkerboard base
        for i in range(0, 1000, 50):
            for j in range(0, 1000, 50):
                if (i//50 + j//50) % 2 == 0:
                    pattern[i:i+50, j:j+50] = [200, 200, 200]
                else:
                    pattern[i:i+50, j:j+50] = [50, 50, 50]
        
        # Add random circles for features
        for _ in range(100):
            center = (np.random.randint(50, 950), np.random.randint(50, 950))
            radius = np.random.randint(10, 30)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(pattern, center, radius, color, -1)
        
        # Add random rectangles
        for _ in range(50):
            pt1 = (np.random.randint(0, 900), np.random.randint(0, 900))
            pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(pattern, pt1, pt2, color, -1)
        
        # Add some text features
        for i in range(20):
            text = f"T{i}"
            position = (np.random.randint(50, 900), np.random.randint(50, 900))
            font_scale = np.random.uniform(0.5, 2.0)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.putText(pattern, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
        
        return pattern
    
    def generate_trajectory(self, num_frames):
        """Generate camera trajectory."""
        # Create a smooth trajectory (figure-8 pattern)
        t = np.linspace(0, 4*np.pi, num_frames)
        
        # Position trajectory
        x = 2 * np.sin(t) * 100  # Scale for pixel coordinates
        y = np.sin(2*t) * 100
        z = np.cos(t/2) * 50 + 300  # Distance from pattern
        
        # Orientation (looking at pattern)
        yaw = np.sin(t/4) * 0.3  # Small yaw oscillation
        pitch = np.sin(t/3) * 0.2  # Small pitch oscillation
        roll = np.zeros_like(t)  # No roll
        
        trajectory = []
        for i in range(num_frames):
            pose = {
                'position': [x[i], y[i], z[i]],
                'orientation': [roll[i], pitch[i], yaw[i]]  # Roll, Pitch, Yaw
            }
            trajectory.append(pose)
        
        return trajectory
    
    def create_camera_matrix(self, fx=800, fy=800):
        """Create camera intrinsic matrix."""
        cx = self.width / 2
        cy = self.height / 2
        return np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
    
    def pose_to_transform(self, pose):
        """Convert pose to transformation matrix."""
        pos = pose['position']
        rot = pose['orientation']  # roll, pitch, yaw
        
        # Create rotation matrix
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(rot[0]), -np.sin(rot[0])],
                       [0, np.sin(rot[0]), np.cos(rot[0])]])
        
        R_y = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])],
                       [0, 1, 0],
                       [-np.sin(rot[1]), 0, np.cos(rot[1])]])
        
        R_z = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0],
                       [np.sin(rot[2]), np.cos(rot[2]), 0],
                       [0, 0, 1]])
        
        R = R_z @ R_y @ R_x
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        return T
    
    def render_image(self, pose, camera_matrix):
        """Render image from given pose."""
        # Create projection matrix
        T = self.pose_to_transform(pose)
        
        # Define 3D points on the pattern plane (z=0)
        pattern_h, pattern_w = self.pattern.shape[:2]
        
        # Create grid of 3D points
        step = 2  # Sample every 2 pixels for performance
        x_coords = np.arange(0, pattern_w, step)
        y_coords = np.arange(0, pattern_h, step)
        
        # Get corresponding image coordinates
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Simple perspective projection
        for y in range(0, self.height, 2):
            for x in range(0, self.width, 2):
                # Back-project pixel to 3D ray
                pixel_coords = np.array([x, y, 1])
                ray_dir = np.linalg.inv(camera_matrix) @ pixel_coords
                
                # Intersect ray with pattern plane (z = 0 in world coordinates)
                # Transform ray to world coordinates
                cam_pos = T[:3, 3]
                cam_rot = T[:3, :3]
                
                world_ray_dir = cam_rot @ ray_dir
                
                # Find intersection with z=0 plane
                if abs(world_ray_dir[2]) > 1e-6:
                    t = -cam_pos[2] / world_ray_dir[2]
                    if t > 0:
                        world_point = cam_pos + t * world_ray_dir
                        
                        # Map to pattern coordinates
                        pattern_x = int(world_point[0] + pattern_w//2)
                        pattern_y = int(world_point[1] + pattern_h//2)
                        
                        # Check bounds and sample
                        if (0 <= pattern_x < pattern_w and 
                            0 <= pattern_y < pattern_h):
                            color = self.pattern[pattern_y, pattern_x]
                            img[y:y+2, x:x+2] = color
        
        # Add some noise for realism
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def generate_sequence(self, num_frames, output_dir, fx=800, fy=800):
        """Generate complete image sequence."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        camera_matrix = self.create_camera_matrix(fx, fy)
        trajectory = self.generate_trajectory(num_frames)
        
        print(f"Generating {num_frames} test images...")
        
        for i, pose in enumerate(trajectory):
            img = self.render_image(pose, camera_matrix)
            
            # Save image
            filename = f"frame_{i:04d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), img)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_frames} images")
        
        # Save ground truth trajectory
        gt_data = []
        for i, pose in enumerate(trajectory):
            pos = pose['position']
            gt_data.append({
                'frame_id': i,
                'timestamp': i * 0.033,  # 30 FPS
                'x': pos[0] / 100.0,  # Convert to meters
                'y': pos[1] / 100.0,
                'z': pos[2] / 100.0,
                'roll': pose['orientation'][0],
                'pitch': pose['orientation'][1],
                'yaw': pose['orientation'][2]
            })
        
        # Save ground truth
        import pandas as pd
        gt_df = pd.DataFrame(gt_data)
        gt_df.to_csv(output_path / 'ground_truth.csv', index=False)
        
        # Save camera parameters
        camera_info = {
            'width': self.width,
            'height': self.height,
            'fx': fx,
            'fy': fy,
            'cx': self.width / 2,
            'cy': self.height / 2
        }
        
        import json
        with open(output_path / 'camera_info.json', 'w') as f:
            json.dump(camera_info, f, indent=2)
        
        print(f"✅ Generated {num_frames} images in {output_dir}")
        print(f"📊 Ground truth saved to {output_dir}/ground_truth.csv")
        print(f"📷 Camera info saved to {output_dir}/camera_info.json")

def main():
    parser = argparse.ArgumentParser(description='Generate test images for VO')
    parser.add_argument('--num_frames', type=int, default=50, 
                       help='Number of frames to generate')
    parser.add_argument('--output_dir', default='test_images', 
                       help='Output directory')
    parser.add_argument('--width', type=int, default=640, 
                       help='Image width')
    parser.add_argument('--height', type=int, default=480, 
                       help='Image height')
    parser.add_argument('--fx', type=float, default=800, 
                       help='Camera focal length X')
    parser.add_argument('--fy', type=float, default=800, 
                       help='Camera focal length Y')
    
    args = parser.parse_args()
    
    generator = TestImageGenerator(args.width, args.height)
    generator.generate_sequence(args.num_frames, args.output_dir, args.fx, args.fy)

if __name__ == '__main__':
    main()