"""
U2Net Segmentation + Force Rectangle Mask
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image

# Global model cache
_u2net_model = None
_device = None


def load_u2net_model(weight_path: str, device: str = None):
    """Load U2Net model from weights"""
    global _u2net_model, _device
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device
    
    if _u2net_model is None:
        print(f"[INFO] Loading U2Net model from {weight_path}")
        
        # Import U2Net model architecture from local module
        from .u2net_model import U2NETP
        _u2net_model = U2NETP(3, 1)
        
        # Load weights
        state_dict = torch.load(weight_path, map_location=device)
        _u2net_model.load_state_dict(state_dict)
        _u2net_model.to(device)
        _u2net_model.eval()
        
        print(f"[INFO] U2Net model loaded on {device}")
    
    return _u2net_model


def segment_image(image: np.ndarray, model, imgsz: int = 320, threshold: float = 0.5) -> np.ndarray:
    """
    Run U2Net segmentation on image
    
    Returns:
        Binary mask (0/255)
    """
    global _device
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare tensor
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(imgsz, imgsz),
        mode='bilinear',
        align_corners=False
    )
    
    # Inference
    with torch.no_grad():
        img_resized = img_resized.to(_device)
        outputs = model(img_resized)
        
        # U2Net returns tuple - use main output
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(logits)
        
        # Resize back to original size
        probs_resized = F.interpolate(
            probs,
            size=(image.shape[0], image.shape[1]),
            mode='bilinear',
            align_corners=False
        )
        
        mask = (probs_resized.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    
    return mask


def process_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def force_rectangle_mask(mask: np.ndarray, expand_factor: float = 1.1, padding_px: int = 10) -> np.ndarray:
    """
    Force mask to be a perfect rectangle using minAreaRect
    
    Similar to white ring segment mechanism - căng đều 4 góc ra
    """
    try:
        if not np.any(mask > 0):
            return mask
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Use minAreaRect for intelligent rectangle fitting
        (cx, cy), (w, h), angle = cv2.minAreaRect(largest_contour)
        
        # Calculate expansion with padding
        new_w = w * expand_factor + padding_px
        new_h = h * expand_factor + padding_px
        
        # Create rectangle using boxPoints
        rect = ((cx, cy), (new_w, new_h), angle)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)
        
        # Create new mask with the rectangle
        h_mask, w_mask = mask.shape
        new_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
        
        # Fill the rectangle
        cv2.fillPoly(new_mask, [box_points], 255)
        
        print(f"[INFO] Created rectangle mask: {new_w:.1f}x{new_h:.1f} at angle {angle:.1f}°")
        return new_mask
        
    except Exception as e:
        print(f"[WARNING] Force rectangle failed: {e}")
        return mask


def get_mask_corners(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Get 4 corner points of the mask's bounding rectangle
    Returns ordered points: [top-left, top-right, bottom-right, bottom-left]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    
    # Order points clockwise starting from top-left
    box = order_points(box)
    
    return box


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    
    return rect


def warp_perspective(image: np.ndarray, src_points: np.ndarray) -> np.ndarray:
    """
    Warp image to straighten the box region
    """
    src_points = src_points.astype(np.float32)
    
    # Calculate width and height
    w1 = np.linalg.norm(src_points[0] - src_points[1])
    w2 = np.linalg.norm(src_points[3] - src_points[2])
    width = int(max(w1, w2))
    
    h1 = np.linalg.norm(src_points[0] - src_points[3])
    h2 = np.linalg.norm(src_points[1] - src_points[2])
    height = int(max(h1, h2))
    
    # Destination points
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Warp
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def visualize_segmentation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create visualization with mask overlay"""
    vis = image.copy()
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = [0, 255, 0]  # Green
    
    # Blend
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    # Draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    return vis
