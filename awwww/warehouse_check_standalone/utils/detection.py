"""
YOLO Detection + Validation
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO

# Global model cache
_yolo_model = None


def load_yolo_model(weight_path: str) -> YOLO:
    """Load YOLO model from weights"""
    global _yolo_model
    
    if _yolo_model is None:
        print(f"[INFO] Loading YOLO model from {weight_path}")
        _yolo_model = YOLO(weight_path)
        print("[INFO] YOLO model loaded successfully")
    
    return _yolo_model


def detect_objects(image: np.ndarray, model: YOLO, conf_threshold: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
    """
    Run YOLO detection on image
    
    Returns:
        (detections, visualization_image)
    """
    results = model(image, verbose=False, conf=conf_threshold)
    
    detections = []
    vis_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Draw on visualization
                color = (0, 255, 0) if class_name == "plastic box" else (0, 0, 255)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_image, f"{class_name}: {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
    
    return detections, vis_image


def validate_detections(detections: List[Dict], expected_items: Dict) -> Dict:
    """
    Validate detected items against expected from QR metadata
    
    Args:
        detections: List of detection dicts from YOLO
        expected_items: Dict of {item_name: count} from QR metadata
        
    Returns:
        Validation result dict
    """
    # Count detected items (exclude plastic box)
    detected_counts = {}
    for det in detections:
        cls = det["class"]
        if cls != "plastic box":
            detected_counts[cls] = detected_counts.get(cls, 0) + 1
    
    # Compare with expected
    missing = {}
    extra = {}
    matched = {}
    
    # Check expected items
    for item, expected_count in expected_items.items():
        detected_count = detected_counts.get(item, 0)
        
        if detected_count == expected_count:
            matched[item] = expected_count
        elif detected_count < expected_count:
            missing[item] = expected_count - detected_count
            if detected_count > 0:
                matched[item] = detected_count
        else:
            matched[item] = expected_count
            extra[item] = detected_count - expected_count
    
    # Check for unexpected items
    for item, count in detected_counts.items():
        if item not in expected_items:
            extra[item] = count
    
    passed = len(missing) == 0 and len(extra) == 0
    
    message = "✅ Validation passed!" if passed else "❌ Validation failed!"
    if missing:
        message += f" Missing: {missing}"
    if extra:
        message += f" Extra: {extra}"
    
    return {
        "passed": passed,
        "message": message,
        "expected": expected_items,
        "detected": detected_counts,
        "matched": matched,
        "missing": missing,
        "extra": extra
    }


def get_box_bbox(detections: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box of plastic box from detections"""
    for det in detections:
        if det["class"] == "plastic box":
            return tuple(det["bbox"])
    return None
