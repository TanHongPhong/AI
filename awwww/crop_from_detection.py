"""
Crop 700x500 images centered on YOLO class 0 (box) detections
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()

# Crop dimensions
CROP_WIDTH = 700
CROP_HEIGHT = 500


def crop_centered(image, center_x, center_y, crop_w, crop_h):
    """Crop an image centered at (center_x, center_y) with size (crop_w, crop_h).
    
    Handles edge cases where crop would exceed image boundaries.
    Returns cropped image or None if crop is not possible.
    """
    img_h, img_w = image.shape[:2]
    
    # Calculate crop boundaries
    x1 = center_x - crop_w // 2
    y1 = center_y - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    # Adjust if out of bounds
    if x1 < 0:
        x1 = 0
        x2 = crop_w
    if y1 < 0:
        y1 = 0
        y2 = crop_h
    if x2 > img_w:
        x2 = img_w
        x1 = max(0, img_w - crop_w)
    if y2 > img_h:
        y2 = img_h
        y1 = max(0, img_h - crop_h)
    
    # Check if crop is valid
    if x2 - x1 < crop_w or y2 - y1 < crop_h:
        # Image is smaller than crop size, return what we can
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(img_w, x1 + crop_w)
        y2 = min(img_h, y1 + crop_h)
    
    return image[y1:y2, x1:x2]


def process_images_with_yolo(input_dir: Path, output_dir: Path, yolo_model):
    """Process images in input_dir, detect class 0, crop 700x500 from center."""
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all image files
    image_files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"\n[INFO] Found {len(image_files)} images in {input_dir}")
    
    if not image_files:
        print("[WARNING] No images found!")
        return
    
    processed_count = 0
    crop_count = 0
    
    for img_path in sorted(image_files):
        print(f"\n[PROCESS] {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    [ERROR] Cannot read image")
            continue
        
        # Run YOLO detection
        results = yolo_model.predict(image, conf=0.25, iou=0.45, verbose=False)
        
        if not results or len(results) == 0:
            print(f"    [SKIP] No detections")
            continue
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            print(f"    [SKIP] No boxes detected")
            continue
        
        # Find class 0 detections
        class_0_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                class_0_boxes.append({
                    'xyxy': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'conf': conf
                })
        
        if not class_0_boxes:
            print(f"    [SKIP] No class 0 detected")
            continue
        
        print(f"    [FOUND] {len(class_0_boxes)} class 0 detection(s)")
        
        # Process each class 0 detection
        for idx, det in enumerate(class_0_boxes):
            center_x, center_y = det['center']
            conf = det['conf']
            
            # Crop 700x500 from center
            cropped = crop_centered(image, center_x, center_y, CROP_WIDTH, CROP_HEIGHT)
            
            if cropped is None or cropped.size == 0:
                print(f"    [ERROR] Crop failed for detection {idx+1}")
                continue
            
            # Save cropped image
            if len(class_0_boxes) == 1:
                output_name = f"{img_path.stem}_crop{img_path.suffix}"
            else:
                output_name = f"{img_path.stem}_crop_{idx+1}{img_path.suffix}"
            
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), cropped)
            
            crop_count += 1
            print(f"    [SAVED] {output_name} (conf: {conf:.2f}, center: {center_x},{center_y})")
        
        processed_count += 1
    
    print(f"\n[DONE] Processed {processed_count} images, saved {crop_count} crops")


def main():
    input_dir = SCRIPT_DIR / "data_new"
    output_dir = input_dir / "output"
    yolo_weights = SCRIPT_DIR / "warehouse_check_standalone" / "weights" / "best.pt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"YOLO Class 0 Detection -> Crop 700x500")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Crop size: {CROP_WIDTH}x{CROP_HEIGHT}")
    
    # Check input directory
    if not input_dir.exists():
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        return
    
    # Load YOLO model
    print("\n[INFO] Loading YOLO model...")
    if not yolo_weights.exists():
        print(f"[ERROR] YOLO weights not found: {yolo_weights}")
        return
    
    yolo_model = YOLO(str(yolo_weights))
    print("[INFO] Model loaded!")
    
    # Process images
    process_images_with_yolo(input_dir, output_dir, yolo_model)
    
    print(f"\n{'='*60}")
    print(f"Done! Output saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
