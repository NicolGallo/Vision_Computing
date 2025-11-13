"""
Computer Vision Lab 1: Test Data Generation and Synthetic Datasets
=================================================================

This module generates realistic synthetic calibration data for testing and validation.
Useful for students who don't have access to physical checkerboards or cameras.

Features:
- Synthetic checkerboard pattern generation
- Realistic camera projection with distortion
- Multiple viewpoint generation
- Noise simulation
- Ground truth validation data

Author: CV Course
Date: 2024
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Optional
import json

class SyntheticCalibrationGenerator:
    """Generate synthetic calibration datasets with known ground truth"""
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480),
                 pattern_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):
        """
        Initialize the synthetic data generator
        
        Args:
            image_size: Output image dimensions (width, height)
            pattern_size: Checkerboard internal corners (cols, rows)
            square_size: Size of each square in real units (mm)
        """
        self.image_size = image_size
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Define ground truth camera parameters
        self.true_camera_matrix = np.array([
            [800, 0, image_size[0]/2],
            [0, 800, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.true_dist_coeffs = np.array([0.1, -0.05, 0.001, 0.0005, 0.01], dtype=np.float32)
        
        # Generate 3D checkerboard pattern
        self.object_points = self._generate_object_points()
        
    def _generate_object_points(self) -> np.ndarray:
        """Generate 3D coordinates of checkerboard corners"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def generate_camera_poses(self, n_poses: int = 20, 
                            distance_range: Tuple[float, float] = (200, 600),
                            angle_range: Tuple[float, float] = (-45, 45)) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate realistic camera poses for calibration
        
        Args:
            n_poses: Number of poses to generate
            distance_range: Min/max distance from pattern (mm)
            angle_range: Min/max rotation angles (degrees)
            
        Returns:
            List of (rotation_vector, translation_vector) tuples
        """
        poses = []
        
        for i in range(n_poses):
            # Random distance
            distance = np.random.uniform(*distance_range)
            
            # Random viewing angles
            rx = np.random.uniform(*angle_range) * np.pi / 180
            ry = np.random.uniform(*angle_range) * np.pi / 180
            rz = np.random.uniform(-10, 10) * np.pi / 180
            
            # Create rotation vector
            rvec = np.array([rx, ry, rz], dtype=np.float32)
            
            # Create translation (pattern at origin, camera moves)
            # Place camera at distance with some random offset
            tx = np.random.uniform(-50, 50)
            ty = np.random.uniform(-50, 50)
            tz = distance
            
            tvec = np.array([tx, ty, tz], dtype=np.float32)
            
            poses.append((rvec, tvec))
        
        return poses
    
    def project_points(self, rvec: np.ndarray, tvec: np.ndarray,
                      add_noise: bool = True, noise_std: float = 0.5) -> Tuple[np.ndarray, bool]:
        """
        Project 3D points to image plane with distortion and noise
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
            add_noise: Whether to add pixel noise
            noise_std: Standard deviation of noise in pixels
            
        Returns:
            Tuple of (image_points, success_flag)
        """
        # Project points using OpenCV
        img_points, _ = cv2.projectPoints(
            self.object_points, rvec, tvec, 
            self.true_camera_matrix, self.true_dist_coeffs
        )
        
        img_points = img_points.reshape(-1, 2)
        
        # Check if points are within image bounds
        valid_points = ((img_points[:, 0] >= 10) & (img_points[:, 0] < self.image_size[0] - 10) &
                       (img_points[:, 1] >= 10) & (img_points[:, 1] < self.image_size[1] - 10))
        
        if np.sum(valid_points) < len(img_points) * 0.8:  # Need at least 80% visible
            return img_points, False
        
        # Add realistic noise
        if add_noise:
            noise = np.random.normal(0, noise_std, img_points.shape)
            img_points += noise
        
        return img_points, True
    
    def generate_checkerboard_image(self, img_points: np.ndarray,
                                   background_texture: bool = True,
                                   lighting_variation: bool = True) -> np.ndarray:
        """
        Generate realistic checkerboard image
        
        Args:
            img_points: Projected corner points
            background_texture: Add realistic background
            lighting_variation: Add lighting variations
            
        Returns:
            Synthetic checkerboard image
        """
        # Create base image
        if background_texture:
            # Add textured background
            img = np.random.randint(40, 80, (*self.image_size[::-1], 3), dtype=np.uint8)
            # Add some texture
            texture = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        else:
            img = np.full((*self.image_size[::-1], 3), 128, dtype=np.uint8)
        
        # Generate checkerboard pattern
        # Create ideal checkerboard first
        board_size = 400  # High resolution for smooth perspective
        ideal_board = np.zeros((board_size, board_size), dtype=np.uint8)
        
        square_pixels = board_size // max(self.pattern_size)
        for i in range(max(self.pattern_size) + 1):
            for j in range(max(self.pattern_size) + 1):
                if (i + j) % 2 == 0:
                    y1, y2 = i * square_pixels, (i + 1) * square_pixels
                    x1, x2 = j * square_pixels, (j + 1) * square_pixels
                    ideal_board[y1:y2, x1:x2] = 255
        
        # Create perspective transformation
        # Map from ideal board to image coordinates
        ideal_corners = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Find bounding box of checkerboard in image
        min_x, max_x = np.min(img_points[:, 0]), np.max(img_points[:, 0])
        min_y, max_y = np.min(img_points[:, 1]), np.max(img_points[:, 1])
        
        # Add margin
        margin = 20
        img_corners = np.array([
            [min_x - margin, min_y - margin],
            [max_x + margin, min_y - margin],
            [max_x + margin, max_y + margin],
            [min_x - margin, max_y + margin]
        ], dtype=np.float32)
        
        # Ensure corners are within image bounds
        img_corners[:, 0] = np.clip(img_corners[:, 0], 0, self.image_size[0] - 1)
        img_corners[:, 1] = np.clip(img_corners[:, 1], 0, self.image_size[1] - 1)
        
        # Apply perspective transformation
        H = cv2.getPerspectiveTransform(ideal_corners, img_corners)
        warped_board = cv2.warpPerspective(ideal_board, H, self.image_size)
        
        # Blend with background
        mask = warped_board > 128
        for c in range(3):
            img[mask, c] = warped_board[mask]
            
        # Add lighting variation
        if lighting_variation:
            # Create smooth lighting gradient
            y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
            center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
            
            # Radial gradient
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            gradient = 1.0 - 0.3 * distance / max_distance
            
            # Apply gradient
            for c in range(3):
                img[:, :, c] = np.clip(img[:, :, c] * gradient, 0, 255).astype(np.uint8)
        
        # Add slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def generate_calibration_dataset(self, n_images: int = 25,
                                   output_dir: str = "synthetic_calibration",
                                   save_ground_truth: bool = True) -> Dict:
        """
        Generate complete calibration dataset
        
        Args:
            n_images: Number of calibration images to generate
            output_dir: Directory to save images
            save_ground_truth: Whether to save ground truth parameters
            
        Returns:
            Dictionary with dataset information
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate camera poses
        poses = self.generate_camera_poses(n_images * 2)  # Generate extra, filter good ones
        
        successful_images = []
        all_object_points = []
        all_image_points = []
        pose_data = []
        
        image_count = 0
        
        for i, (rvec, tvec) in enumerate(poses):
            if image_count >= n_images:
                break
                
            # Project points
            img_points, success = self.project_points(rvec, tvec)
            
            if not success:
                continue
            
            # Generate image
            img = self.generate_checkerboard_image(img_points)
            
            # Verify corner detection works
            ret, detected_corners = cv2.findChessboardCorners(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.pattern_size, None
            )
            
            if not ret:
                continue
            
            # Save image
            img_filename = f"calibration_{image_count:03d}.png"
            img_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Store data
            successful_images.append(img_path)
            all_object_points.append(self.object_points)
            all_image_points.append(img_points)
            pose_data.append({
                'rvec': rvec.tolist(),
                'tvec': tvec.tolist(),
                'image_points': img_points.tolist()
            })
            
            image_count += 1
            print(f"Generated image {image_count}/{n_images}: {img_filename}")
        
        # Save ground truth data
        dataset_info = {
            'image_size': self.image_size,
            'pattern_size': self.pattern_size,
            'square_size': self.square_size,
            'n_images': len(successful_images),
            'image_files': successful_images
        }
        
        if save_ground_truth:
            ground_truth = {
                'camera_matrix': self.true_camera_matrix.tolist(),
                'distortion_coefficients': self.true_dist_coeffs.tolist(),
                'poses': pose_data,
                'object_points': self.object_points.tolist()
            }
            
            # Save as JSON
            with open(os.path.join(output_dir, 'ground_truth.json'), 'w') as f:
                json.dump(ground_truth, f, indent=2)
            
            # Save as numpy
            np.savez(os.path.join(output_dir, 'ground_truth.npz'),
                    camera_matrix=self.true_camera_matrix,
                    distortion_coeffs=self.true_dist_coeffs,
                    object_points=self.object_points,
                    all_image_points=all_image_points)
        
        # Save dataset info
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✓ Successfully generated {len(successful_images)} calibration images")
        print(f"✓ Saved to: {output_dir}")
        
        return dataset_info
    
    def validate_synthetic_data(self, dataset_dir: str) -> Dict:
        """
        Validate synthetic calibration data by running calibration
        
        Args:
            dataset_dir: Directory containing synthetic dataset
            
        Returns:
            Validation results comparing estimated vs ground truth
        """
        # Load dataset info
        with open(os.path.join(dataset_dir, 'dataset_info.json'), 'r') as f:
            dataset_info = json.load(f)
        
        # Load ground truth
        with open(os.path.join(dataset_dir, 'ground_truth.json'), 'r') as f:
            ground_truth = json.load(f)
        
        true_mtx = np.array(ground_truth['camera_matrix'])
        true_dist = np.array(ground_truth['distortion_coefficients'])
        
        # Run calibration on synthetic data
        print("Running calibration on synthetic data...")
        
        objpoints = []
        imgpoints = []
        
        for img_file in dataset_info['image_files']:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, tuple(dataset_info['pattern_size']), None)
            
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(self.object_points)
                imgpoints.append(corners)
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (dataset_info['image_size'][0], dataset_info['image_size'][1]), None, None
        )
        
        # Compare results
        validation_results = {
            'calibration_successful': ret,
            'n_images_used': len(objpoints),
            'ground_truth': {
                'camera_matrix': true_mtx.tolist(),
                'distortion_coeffs': true_dist.tolist()
            },
            'estimated': {
                'camera_matrix': mtx.tolist(),
                'distortion_coeffs': dist.tolist()
            },
            'errors': {},
            'reprojection_error': 0.0
        }
        
        if ret:
            # Compute parameter errors
            fx_error = abs(mtx[0, 0] - true_mtx[0, 0]) / true_mtx[0, 0] * 100
            fy_error = abs(mtx[1, 1] - true_mtx[1, 1]) / true_mtx[1, 1] * 100
            cx_error = abs(mtx[0, 2] - true_mtx[0, 2])
            cy_error = abs(mtx[1, 2] - true_mtx[1, 2])
            
            dist_errors = np.abs(dist.ravel()[:len(true_dist)] - true_dist) / (np.abs(true_dist) + 1e-6) * 100
            
            validation_results['errors'] = {
                'fx_percent': fx_error,
                'fy_percent': fy_error,
                'cx_pixels': cx_error,
                'cy_pixels': cy_error,
                'distortion_percent': dist_errors.tolist()
            }
            
            # Compute reprojection error
            total_error = 0
            total_points = 0
            
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
                total_points += 1
            
            validation_results['reprojection_error'] = total_error / total_points
            
            # Print results
            print("\n=== Synthetic Data Validation ===")
            print(f"Calibration successful: {ret}")
            print(f"Images used: {len(objpoints)}")
            print(f"Reprojection error: {validation_results['reprojection_error']:.4f} pixels")
            print("\nParameter Errors:")
            print(f"  fx: {fx_error:.2f}%")
            print(f"  fy: {fy_error:.2f}%")
            print(f"  cx: {cx_error:.2f} pixels")
            print(f"  cy: {cy_error:.2f} pixels")
            print(f"  Distortion coefficients: {[f'{e:.2f}%' for e in dist_errors]}")
            
        # Save validation results
        with open(os.path.join(dataset_dir, 'validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results

def create_test_datasets():
    """Create various test datasets for different scenarios"""
    
    # Dataset 1: Standard calibration
    print("Creating standard calibration dataset...")
    generator1 = SyntheticCalibrationGenerator(
        image_size=(640, 480),
        pattern_size=(9, 6),
        square_size=25.0
    )
    
    dataset1 = generator1.generate_calibration_dataset(
        n_images=20,
        output_dir="datasets/standard_calibration"
    )
    
    # Dataset 2: High distortion camera
    print("\nCreating high distortion dataset...")
    generator2 = SyntheticCalibrationGenerator(
        image_size=(800, 600),
        pattern_size=(10, 7),
        square_size=20.0
    )
    generator2.true_dist_coeffs = np.array([0.3, -0.2, 0.002, 0.001, 0.05])
    
    dataset2 = generator2.generate_calibration_dataset(
        n_images=25,
        output_dir="datasets/high_distortion"
    )
    
    # Dataset 3: Wide angle camera
    print("\nCreating wide angle dataset...")
    generator3 = SyntheticCalibrationGenerator(
        image_size=(1024, 768),
        pattern_size=(8, 5),
        square_size=30.0
    )
    generator3.true_camera_matrix[0, 0] = 400  # Lower focal length = wider angle
    generator3.true_camera_matrix[1, 1] = 400
    
    dataset3 = generator3.generate_calibration_dataset(
        n_images=30,
        output_dir="datasets/wide_angle"
    )
    
    print("\n✓ All test datasets created successfully!")
    
    # Validate all datasets
    print("\nValidating datasets...")
    validation_results = {}
    
    for name, path in [("standard", "datasets/standard_calibration"),
                       ("high_distortion", "datasets/high_distortion"),
                       ("wide_angle", "datasets/wide_angle")]:
        print(f"\nValidating {name} dataset...")
        if name == "standard":
            results = generator1.validate_synthetic_data(path)
        elif name == "high_distortion":
            results = generator2.validate_synthetic_data(path)
        else:
            results = generator3.validate_synthetic_data(path)
        
        validation_results[name] = results
    
    return validation_results

def generate_challenge_data():
    """Generate challenging datasets for advanced exercises"""
    
    # Challenge 1: Stereo calibration data
    print("Generating stereo calibration dataset...")
    
    # Left camera
    left_generator = SyntheticCalibrationGenerator(
        image_size=(640, 480),
        pattern_size=(9, 6),
        square_size=25.0
    )
    
    # Right camera (baseline = 100mm)
    right_generator = SyntheticCalibrationGenerator(
        image_size=(640, 480),
        pattern_size=(9, 6),
        square_size=25.0
    )
    right_generator.true_camera_matrix[0, 2] += 100  # Baseline offset
    
    # Generate synchronized poses
    poses = left_generator.generate_camera_poses(20)
    
    os.makedirs("datasets/stereo_calibration/left", exist_ok=True)
    os.makedirs("datasets/stereo_calibration/right", exist_ok=True)
    
    for i, (rvec, tvec) in enumerate(poses[:15]):
        # Left image
        left_points, success_left = left_generator.project_points(rvec, tvec)
        if success_left:
            left_img = left_generator.generate_checkerboard_image(left_points)
            cv2.imwrite(f"datasets/stereo_calibration/left/left_{i:03d}.png", 
                       cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
        
        # Right image (offset pose)
        right_tvec = tvec.copy()
        right_tvec[0] += 100  # Baseline
        right_points, success_right = right_generator.project_points(rvec, right_tvec)
        if success_right:
            right_img = right_generator.generate_checkerboard_image(right_points)
            cv2.imwrite(f"datasets/stereo_calibration/right/right_{i:03d}.png", 
                       cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
    
    print("✓ Stereo calibration dataset created")
    
    # Challenge 2: Partial pattern visibility
    print("Generating partial pattern dataset...")
    
    partial_generator = SyntheticCalibrationGenerator(
        image_size=(640, 480),
        pattern_size=(12, 8),  # Larger pattern
        square_size=20.0
    )
    
    # Generate poses where pattern is partially outside image
    close_poses = partial_generator.generate_camera_poses(
        20, distance_range=(100, 300), angle_range=(-60, 60)
    )
    
    os.makedirs("datasets/partial_pattern", exist_ok=True)
    
    for i, (rvec, tvec) in enumerate(close_poses[:15]):
        img_points, _ = partial_generator.project_points(rvec, tvec, add_noise=True)
        img = partial_generator.generate_checkerboard_image(img_points)
        cv2.imwrite(f"datasets/partial_pattern/partial_{i:03d}.png", 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("✓ Partial pattern dataset created")

if __name__ == "__main__":
    print("=== Synthetic Calibration Data Generator ===")
    
    # Create standard test datasets
    validation_results = create_test_datasets()
    
    # Create challenge datasets
    generate_challenge_data()
    
    print("\n=== Generation Complete ===")
    print("Available datasets:")
    print("  - datasets/standard_calibration/")
    print("  - datasets/high_distortion/")
    print("  - datasets/wide_angle/")
    print("  - datasets/stereo_calibration/")
    print("  - datasets/partial_pattern/")
    
    print("\nDatasets are ready for Lab 1 exercises!")