"""
Canonical Rotation - Xoay ảnh để QR ở góc dưới-phải
"""
import cv2
import numpy as np
from typing import Tuple


def rotate_points_90cw(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    """Rotate points 90 degrees clockwise"""
    rotated = np.zeros_like(points)
    rotated[:, 0] = H - 1 - points[:, 1]  # new_x = H - 1 - old_y
    rotated[:, 1] = points[:, 0]           # new_y = old_x
    return rotated, H, W  # New dimensions: (H, W) -> (W, H)


def rotate_points_90ccw(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    """Rotate points 90 degrees counter-clockwise"""
    rotated = np.zeros_like(points)
    rotated[:, 0] = points[:, 1]           # new_x = old_y
    rotated[:, 1] = W - 1 - points[:, 0]   # new_y = W - 1 - old_x
    return rotated, H, W


def rotate_points_180(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    """Rotate points 180 degrees"""
    rotated = np.zeros_like(points)
    rotated[:, 0] = W - 1 - points[:, 0]
    rotated[:, 1] = H - 1 - points[:, 1]
    return rotated, W, H


def choose_best_rotation(img: np.ndarray, qr_center: np.ndarray) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Chọn xoay 0/90cw/180/270cw sao cho tâm QR gần góc phải-dưới nhất.
    
    Args:
        img: Input image
        qr_center: Center point of QR code [x, y]
        
    Returns:
        (rotated_img, rot_code, qr_center_rotated)
        rot_code in {"rot0", "rot90cw", "rot180", "rot270cw"}
    """
    H, W = img.shape[:2]
    
    # Normalize QR center position to 0-1 range
    qr_x_norm = qr_center[0] / W
    qr_y_norm = qr_center[1] / H
    
    print(f"[DEBUG] Image size: {W}x{H}, QR center: ({qr_center[0]:.1f}, {qr_center[1]:.1f})")
    print(f"[DEBUG] QR normalized position: ({qr_x_norm:.2f}, {qr_y_norm:.2f})")
    
    # Determine which quadrant QR is in
    # Target: bottom-right (x > 0.5, y > 0.5)
    
    if qr_x_norm > 0.5 and qr_y_norm > 0.5:
        # Already bottom-right
        print("[DEBUG] QR in bottom-right, no rotation needed")
        return img, "rot0", qr_center
    
    elif qr_x_norm > 0.5 and qr_y_norm <= 0.5:
        # Top-right -> rotate 90 clockwise
        print("[DEBUG] QR in top-right, rotating 90cw")
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_center = np.array([H - 1 - qr_center[1], qr_center[0]], dtype=np.float32)
        return rotated, "rot90cw", new_center
    
    elif qr_x_norm <= 0.5 and qr_y_norm <= 0.5:
        # Top-left -> rotate 180
        print("[DEBUG] QR in top-left, rotating 180")
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        new_center = np.array([W - 1 - qr_center[0], H - 1 - qr_center[1]], dtype=np.float32)
        return rotated, "rot180", new_center
    
    else:
        # Bottom-left -> rotate 90 counter-clockwise (270 cw)
        print("[DEBUG] QR in bottom-left, rotating 270cw")
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_center = np.array([qr_center[1], W - 1 - qr_center[0]], dtype=np.float32)
        return rotated, "rot270cw", new_center


def make_canonical(img: np.ndarray, qr_center: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert image to canonical orientation with QR at bottom-right corner
    
    Args:
        img: Input image (should be warped/cropped box region)
        qr_center: Center of QR code in the image
        
    Returns:
        (canonical_image, rotation_code)
    """
    canonical, rot_code, _ = choose_best_rotation(img, qr_center)
    print(f"[INFO] Applied rotation: {rot_code}")
    return canonical, rot_code


def apply_rotation(img: np.ndarray, rot_code: str) -> np.ndarray:
    """Apply a specific rotation by code"""
    if rot_code == "rot0":
        return img
    elif rot_code == "rot90cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot_code == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rot_code == "rot270cw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img
