"""
Computer Vision Lab 1: Camera Calibration and Image Transformations
Demo Code Implementation

This module implements all the core algorithms from Lecture 1:
- Zhang's camera calibration method
- Homography estimation with DLT
- RANSAC for robust estimation
- Lens distortion correction
- Color space transformations
- Histogram equalization

Author: CV Course
Date: 2024
"""

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import glob
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class CVLab1:
    """Complete implementation of Computer Vision Lab 1 algorithms"""
    
    def __init__(self):
        """Initialize the lab with default parameters"""
        self.calibration_data = {}
        
    # ================================
    # PYTORCH AND IMAGE I/O UTILITIES
    # ================================
    
    def numpy_to_torch(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Convert HxWxC numpy array to CxHxW torch tensor
        
        Args:
            img_np: Input image as numpy array (H, W, C)
            
        Returns:
            torch.Tensor: Image tensor (C, H, W)
        """
        if len(img_np.shape) == 3:
            return torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        else:
            return torch.from_numpy(img_np).unsqueeze(0).float()
    
    def torch_to_numpy(self, img_torch: torch.Tensor) -> np.ndarray:
        """
        Convert CxHxW torch tensor to HxWxC numpy array
        
        Args:
            img_torch: Input tensor (C, H, W)
            
        Returns:
            np.ndarray: Image array (H, W, C)
        """
        if img_torch.dim() == 3:
            return img_torch.permute(1, 2, 0).cpu().numpy()
        else:
            return img_torch.squeeze().cpu().numpy()
    
    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load image using OpenCV and convert to RGB
        
        Args:
            filepath: Path to image file
            
        Returns:
            np.ndarray: RGB image array
        """
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def visualize_images(self, images: List[np.ndarray], titles: List[str], 
                        figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Display multiple images in a row
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
            
        for i, (img, title) in enumerate(zip(images, titles)):
            cmap = 'gray' if len(img.shape) == 2 else None
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    
    # ================================
    # GEOMETRIC TRANSFORMATIONS
    # ================================
    
    def create_similarity_transform(self, scale: float, rotation: float, 
                                  translation: Tuple[float, float]) -> np.ndarray:
        """
        Create similarity transformation matrix
        
        Args:
            scale: Scaling factor
            rotation: Rotation angle in radians
            translation: Translation (tx, ty)
            
        Returns:
            np.ndarray: 3x3 transformation matrix
        """
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        tx, ty = translation
        
        transform_matrix = np.array([
            [scale * cos_r, -scale * sin_r, tx],
            [scale * sin_r,  scale * cos_r, ty],
            [0,              0,              1]
        ], dtype=np.float32)
        
        return transform_matrix
    
    def apply_transformation(self, img: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply 3x3 transformation matrix to image
        
        Args:
            img: Input image
            transform_matrix: 3x3 transformation matrix
            
        Returns:
            np.ndarray: Transformed image
        """
        h, w = img.shape[:2]
        transformed = cv2.warpPerspective(img, transform_matrix, (w, h))
        return transformed
    
    # ================================
    # CORNER DETECTION FOR CALIBRATION
    # ================================
    
    def detect_checkerboard_corners(self, img: np.ndarray, 
                                   pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard corners with sub-pixel accuracy
        
        Args:
            img: Input image
            pattern_size: Number of internal corners (cols, rows)
            
        Returns:
            Tuple of (success, corner_coordinates)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def visualize_corners(self, img: np.ndarray, corners: np.ndarray, 
                         pattern_size: Tuple[int, int]) -> np.ndarray:
        """
        Draw detected corners on image
        
        Args:
            img: Input image
            corners: Detected corner coordinates
            pattern_size: Checkerboard pattern size
            
        Returns:
            np.ndarray: Image with corners drawn
        """
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, True)
        return img_with_corners
    
    # ================================
    # ZHANG'S CAMERA CALIBRATION
    # ================================
    
    def calibrate_camera(self, calibration_images: List[str], 
                        pattern_size: Tuple[int, int], 
                        square_size: float) -> Tuple[bool, np.ndarray, np.ndarray, List, List]:
        """
        Implement Zhang's camera calibration method
        
        Args:
            calibration_images: List of calibration image paths
            pattern_size: Checkerboard pattern size (cols, rows)
            square_size: Size of checkerboard squares in real units
            
        Returns:
            Tuple of (success, camera_matrix, distortion_coeffs, rvecs, tvecs)
        """
        # Prepare object points (3D points in real world space)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        successful_images = []
        
        for img_path in calibration_images:
            try:
                img = self.load_image(img_path)
                ret, corners = self.detect_checkerboard_corners(img, pattern_size)
                
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    successful_images.append(img_path)
                    print(f"✓ Successfully processed: {os.path.basename(img_path)}")
                else:
                    print(f"✗ Failed to find corners in: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        if len(objpoints) < 3:
            print(f"Error: Need at least 3 successful images, got {len(objpoints)}")
            return False, None, None, None, None
        
        # Perform calibration
        img_shape = self.load_image(successful_images[0]).shape[:2][::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)
        
        # Store calibration data
        self.calibration_data = {
            'camera_matrix': mtx,
            'distortion_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'objpoints': objpoints,
            'imgpoints': imgpoints,
            'successful_images': successful_images
        }
        
        return ret, mtx, dist, rvecs, tvecs
    
    def compute_reprojection_error(self) -> Tuple[float, np.ndarray]:
        """
        Compute RMS reprojection error for calibrated camera
        
        Returns:
            Tuple of (mean_error, per_image_errors)
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available. Run calibrate_camera first.")
        
        mtx = self.calibration_data['camera_matrix']
        dist = self.calibration_data['distortion_coeffs']
        rvecs = self.calibration_data['rvecs']
        tvecs = self.calibration_data['tvecs']
        objpoints = self.calibration_data['objpoints']
        imgpoints = self.calibration_data['imgpoints']
        
        total_error = 0
        per_image_errors = []
        
        for i in range(len(objpoints)):
            # Project 3D points to image plane
            imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            
            # Compute error
            error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            per_image_errors.append(error)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        return mean_error, np.array(per_image_errors)
    
    # ================================
    # HOMOGRAPHY ESTIMATION
    # ================================
    
    def normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize points for DLT numerical stability (Hartley normalization)
        
        Args:
            points: Input points (N, 2)
            
        Returns:
            Tuple of (normalized_points, transformation_matrix)
        """
        points = np.array(points, dtype=np.float32)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Compute mean distance to centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        scale = np.sqrt(2) / np.mean(distances)
        
        # Create transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply transformation
        points_hom = np.column_stack([points, np.ones(len(points))])
        points_norm = (T @ points_hom.T).T
        points_norm = points_norm[:, :2] / points_norm[:, [2]]
        
        return points_norm, T
    
    def estimate_homography_dlt(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Estimate homography using Direct Linear Transform
        
        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)
            
        Returns:
            np.ndarray: 3x3 homography matrix
        """
        assert len(src_pts) >= 4, "Need at least 4 point correspondences"
        
        # Normalize points for numerical stability
        src_norm, T1 = self.normalize_points(src_pts)
        dst_norm, T2 = self.normalize_points(dst_pts)
        
        # Set up linear system Ah = 0
        A = []
        for i in range(len(src_norm)):
            x, y = src_norm[i]
            u, v = dst_norm[i]
            
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        
        A = np.array(A)
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :] / Vt[-1, -1]  # Normalize by last element
        H_norm = h.reshape(3, 3)
        
        # Denormalize
        H = np.linalg.inv(T2) @ H_norm @ T1
        
        return H / H[2, 2]  # Normalize by bottom-right element
    
    def ransac_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                         threshold: float = 3.0, max_iters: int = 1000) -> Tuple[np.ndarray, List[int]]:
        """
        Robust homography estimation using RANSAC
        
        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)
            threshold: Inlier threshold in pixels
            max_iters: Maximum RANSAC iterations
            
        Returns:
            Tuple of (best_homography, inlier_indices)
        """
        best_H = None
        best_inliers = []
        best_score = 0
        
        n_points = len(src_pts)
        
        for iteration in range(max_iters):
            # Randomly sample 4 points
            sample_idx = np.random.choice(n_points, 4, replace=False)
            sample_src = src_pts[sample_idx]
            sample_dst = dst_pts[sample_idx]
            
            try:
                # Estimate homography from sample
                H = self.estimate_homography_dlt(sample_src, sample_dst)
                
                # Count inliers
                inliers = []
                for i in range(n_points):
                    # Transform source point
                    src_hom = np.append(src_pts[i], 1)
                    dst_pred_hom = H @ src_hom
                    
                    if abs(dst_pred_hom[2]) < 1e-8:
                        continue
                        
                    dst_pred = dst_pred_hom[:2] / dst_pred_hom[2]
                    
                    # Compute reprojection error
                    error = np.linalg.norm(dst_pred - dst_pts[i])
                    if error < threshold:
                        inliers.append(i)
                
                # Update best model if better
                if len(inliers) > best_score:
                    best_score = len(inliers)
                    best_inliers = inliers
                    best_H = H
                    
            except Exception:
                continue
        
        # Refine with all inliers
        if best_H is not None and len(best_inliers) >= 4:
            try:
                refined_H = self.estimate_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
                return refined_H, best_inliers
            except Exception:
                pass
        
        return best_H, best_inliers
    
    # ================================
    # LENS DISTORTION CORRECTION
    # ================================
    
    def undistort_points(self, points: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """
        Undistort points using camera parameters
        
        Args:
            points: Distorted points (N, 2)
            mtx: Camera matrix
            dist: Distortion coefficients
            
        Returns:
            np.ndarray: Undistorted points
        """
        # Use OpenCV's built-in function for accuracy
        points_undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), mtx, dist, P=mtx)
        return points_undistorted.reshape(-1, 2)
    
    def undistort_image(self, img: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """
        Undistort entire image using camera parameters
        
        Args:
            img: Input distorted image
            mtx: Camera matrix
            dist: Distortion coefficients
            
        Returns:
            np.ndarray: Undistorted image
        """
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
        
        # Crop the image
        x, y, w_crop, h_crop = roi
        undistorted = undistorted[y:y+h_crop, x:x+w_crop]
        
        return undistorted
    
    # ================================
    # COLOR SPACE TRANSFORMATIONS
    # ================================
    
    def rgb_to_hsv_manual(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Manual RGB to HSV conversion following lecture algorithm
        
        Args:
            rgb_img: RGB image (H, W, 3) with values [0, 255]
            
        Returns:
            np.ndarray: HSV image with H in [0, 360], S and V in [0, 1]
        """
        rgb_normalized = rgb_img.astype(np.float32) / 255.0
        h, w, c = rgb_img.shape
        hsv_img = np.zeros_like(rgb_normalized)
        
        for i in range(h):
            for j in range(w):
                r, g, b = rgb_normalized[i, j]
                
                # Value (brightness)
                v = max(r, g, b)
                min_val = min(r, g, b)
                delta = v - min_val
                
                # Saturation
                s = 0 if v == 0 else delta / v
                
                # Hue
                if delta == 0:
                    h_val = 0
                elif v == r:
                    h_val = 60 * ((g - b) / delta % 6)
                elif v == g:
                    h_val = 60 * ((b - r) / delta + 2)
                else:  # v == b
                    h_val = 60 * ((r - g) / delta + 4)
                
                hsv_img[i, j] = [h_val, s, v]
        
        return hsv_img
    
    def hsv_to_rgb_manual(self, hsv_img: np.ndarray) -> np.ndarray:
        """
        Manual HSV to RGB conversion
        
        Args:
            hsv_img: HSV image with H in [0, 360], S and V in [0, 1]
            
        Returns:
            np.ndarray: RGB image with values [0, 255]
        """
        h, w, c = hsv_img.shape
        rgb_img = np.zeros_like(hsv_img)
        
        for i in range(h):
            for j in range(w):
                h_val, s, v = hsv_img[i, j]
                
                c_val = v * s
                x = c_val * (1 - abs((h_val / 60) % 2 - 1))
                m = v - c_val
                
                if 0 <= h_val < 60:
                    r_prime, g_prime, b_prime = c_val, x, 0
                elif 60 <= h_val < 120:
                    r_prime, g_prime, b_prime = x, c_val, 0
                elif 120 <= h_val < 180:
                    r_prime, g_prime, b_prime = 0, c_val, x
                elif 180 <= h_val < 240:
                    r_prime, g_prime, b_prime = 0, x, c_val
                elif 240 <= h_val < 300:
                    r_prime, g_prime, b_prime = x, 0, c_val
                else:
                    r_prime, g_prime, b_prime = c_val, 0, x
                
                r = (r_prime + m) * 255
                g = (g_prime + m) * 255
                b = (b_prime + m) * 255
                
                rgb_img[i, j] = [r, g, b]
        
        return rgb_img.astype(np.uint8)
    
    # ================================
    # HISTOGRAM EQUALIZATION
    # ================================
    
    def histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        Global histogram equalization from scratch
        
        Args:
            img: Grayscale image
            
        Returns:
            np.ndarray: Histogram equalized image
        """
        # Ensure image is grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Compute histogram
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        
        # Compute cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Apply transformation
        img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
        img_equalized = img_equalized.reshape(img.shape).astype(np.uint8)
        
        return img_equalized
    
    def adaptive_histogram_equalization(self, img: np.ndarray, 
                                       tile_size: Tuple[int, int] = (8, 8),
                                       clip_limit: float = 2.0) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            img: Input grayscale image
            tile_size: Size of tiles for local equalization
            clip_limit: Contrast limiting parameter
            
        Returns:
            np.ndarray: CLAHE processed image
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        img_clahe = clahe.apply(img)
        
        return img_clahe

def demo_calibration():
    """Demonstration of camera calibration workflow"""
    print("=== Computer Vision Lab 1 Demo ===")
    
    lab = CVLab1()
    
    # Create synthetic calibration data for demo
    print("\n1. Creating synthetic calibration images...")
    create_synthetic_calibration_images()
    
    # Load calibration images
    calibration_images = glob.glob("calibration_*.png")
    if not calibration_images:
        print("No calibration images found. Please create some checkerboard images.")
        return
    
    pattern_size = (9, 6)  # Internal corners
    square_size = 25.0  # mm
    
    print(f"\n2. Running camera calibration with {len(calibration_images)} images...")
    
    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = lab.calibrate_camera(
        calibration_images, pattern_size, square_size)
    
    if ret:
        print("\n✓ Calibration successful!")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients: {dist.ravel()}")
        
        # Compute reprojection error
        mean_error, per_image_errors = lab.compute_reprojection_error()
        print(f"\nReprojection error: {mean_error:.3f} pixels")
        
        # Visualize calibration results
        demo_image = lab.load_image(calibration_images[0])
        ret, corners = lab.detect_checkerboard_corners(demo_image, pattern_size)
        
        if ret:
            img_with_corners = lab.visualize_corners(demo_image, corners, pattern_size)
            undistorted_img = lab.undistort_image(demo_image, mtx, dist)
            
            lab.visualize_images(
                [demo_image, img_with_corners, undistorted_img],
                ['Original', 'With Corners', 'Undistorted'],
                figsize=(15, 5)
            )
    else:
        print("✗ Calibration failed!")

def demo_homography():
    """Demonstration of homography estimation"""
    print("\n3. Demonstrating homography estimation...")
    
    lab = CVLab1()
    
    # Create synthetic point correspondences
    src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    
    # Apply known transformation
    scale = 1.2
    rotation = np.pi / 6
    translation = (50, 30)
    
    T = lab.create_similarity_transform(scale, rotation, translation)
    
    # Transform points
    src_hom = np.column_stack([src_pts, np.ones(len(src_pts))])
    dst_hom = (T @ src_hom.T).T
    dst_pts = dst_hom[:, :2] / dst_hom[:, [2]]
    
    # Add some noise and outliers for RANSAC demo
    dst_pts_noisy = dst_pts + np.random.normal(0, 1, dst_pts.shape)
    
    # Add outliers
    outlier_idx = [1, 3]
    dst_pts_noisy[outlier_idx] += np.random.normal(0, 20, (len(outlier_idx), 2))
    
    # Estimate homography with DLT
    H_dlt = lab.estimate_homography_dlt(src_pts, dst_pts)
    
    # Estimate homography with RANSAC
    H_ransac, inliers = lab.ransac_homography(src_pts, dst_pts_noisy, threshold=2.0)
    
    print(f"DLT Homography:\n{H_dlt}")
    print(f"RANSAC Homography:\n{H_ransac}")
    print(f"RANSAC found {len(inliers)} inliers out of {len(src_pts)} points")

def demo_color_spaces():
    """Demonstration of color space transformations"""
    print("\n4. Demonstrating color space transformations...")
    
    lab = CVLab1()
    
    # Create a test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:, :33, 0] = 255  # Red
    test_img[:, 33:66, 1] = 255  # Green  
    test_img[:, 66:, 2] = 255  # Blue
    
    # Convert to HSV manually
    hsv_manual = lab.rgb_to_hsv_manual(test_img)
    
    # Convert back to RGB
    rgb_reconstructed = lab.hsv_to_rgb_manual(hsv_manual)
    
    # Compare with OpenCV
    hsv_opencv = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
    
    lab.visualize_images(
        [test_img, hsv_manual[:, :, 0]/360, rgb_reconstructed],
        ['Original RGB', 'HSV Hue Channel', 'Reconstructed RGB'],
        figsize=(15, 4)
    )

def create_synthetic_calibration_images():
    """Create synthetic calibration images for demo purposes"""
    print("Creating synthetic calibration images...")
    
    # This is a simplified version - in practice, you'd use real checkerboard images
    for i in range(5):
        # Create a simple checkerboard pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some checkerboard-like pattern
        for row in range(0, 480, 40):
            for col in range(0, 640, 40):
                if (row // 40 + col // 40) % 2 == 0:
                    img[row:row+40, col:col+40] = 255
        
        # Add some perspective distortion and noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f"calibration_{i:02d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Run all demonstrations
    demo_calibration()
    demo_homography()
    demo_color_spaces()
    
    print("\n=== Lab 1 Demo Complete ===")
    print("Try running the exercises in the lab notes!")