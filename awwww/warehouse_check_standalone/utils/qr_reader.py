"""
QR Reader using zxingcpp
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import json
import os

try:
    import zxingcpp
except ImportError:
    print("[WARNING] zxingcpp not installed. Run: pip install zxingcpp")
    zxingcpp = None


def decode_qr(image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Decode QR code from image using zxingcpp
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        (qr_text, qr_points) or (None, None) if not found
    """
    if zxingcpp is None:
        return None, None
    
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        
        # Decode using zxingcpp
        results = zxingcpp.read_barcodes(rgb)
        
        for result in results:
            if result.format == zxingcpp.BarcodeFormat.QRCode:
                # Get text
                qr_text = result.text
                
                # Get position points
                pos = result.position
                points = np.array([
                    [pos.top_left.x, pos.top_left.y],
                    [pos.top_right.x, pos.top_right.y],
                    [pos.bottom_right.x, pos.bottom_right.y],
                    [pos.bottom_left.x, pos.bottom_left.y]
                ], dtype=np.float32)
                
                return qr_text, points
        
        return None, None
        
    except Exception as e:
        print(f"[ERROR] QR decode failed: {e}")
        return None, None


def get_qr_center(points: np.ndarray) -> np.ndarray:
    """Get center point of QR code"""
    if points is None:
        return None
    return np.mean(points, axis=0)


def parse_qr_payload(qr_text: str) -> Dict:
    """
    Parse QR payload to extract box info
    Expected format: JSON or simple key=value
    Always returns a dict with at least _qr key
    """
    try:
        # Try JSON first
        data = json.loads(qr_text)
        # If JSON returns a dict, use it but ensure _qr exists
        if isinstance(data, dict):
            data["_qr"] = data.get("_qr", qr_text.strip())
            return data
        else:
            # JSON returned int/str/list - wrap it
            return {"_raw": data, "_qr": str(data)}
    except:
        pass
    
    # Try simple format: BOX_ID or key=value pairs
    result = {"_raw": qr_text, "_qr": qr_text.strip()}
    
    if "=" in qr_text:
        for pair in qr_text.split("&"):
            if "=" in pair:
                key, val = pair.split("=", 1)
                result[key.strip()] = val.strip()
    
    return result


def load_qr_metadata(qr_id: str, qr_meta_dir: str = "qr_meta") -> Optional[Dict]:
    """
    Load QR metadata from JSON file
    
    Args:
        qr_id: QR code ID
        qr_meta_dir: Directory containing QR meta JSON files
        
    Returns:
        Metadata dict or None
    """
    try:
        meta_path = os.path.join(qr_meta_dir, f"{qr_id}.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load QR metadata for {qr_id}: {e}")
    
    return None
