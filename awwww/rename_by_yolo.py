"""
Rename files based on YOLO detected class.
Format: {fruit_name}_class_{index}[_yolo].jpg

Classes: 1=cam, 3=táo, 4=lê
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent.resolve()

# Class mapping
CLASS_MAP = {
    1: "quyt",
    3: "tao", 
    4: "le"
}


def get_dominant_class(image_path: Path, yolo_model) -> int:
    """Detect objects and return the class id that is in CLASS_MAP (highest conf)."""
    image = cv2.imread(str(image_path))
    if image is None:
        return -1
    
    results = yolo_model.predict(image, conf=0.25, iou=0.45, verbose=False)
    
    if not results or len(results) == 0:
        return -1
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return -1
    
    # Find highest confidence detection that is in CLASS_MAP
    best_conf = 0
    best_class = -1
    for box in result.boxes:
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        # Only consider classes in CLASS_MAP
        if cls_id in CLASS_MAP and conf > best_conf:
            best_conf = conf
            best_class = cls_id
    
    return best_class


def rename_files_in_folder(folder: Path, yolo_model):
    """Rename all crop/yolo jpg files in folder based on detected class."""
    
    # Get all crop files (not yolo files, not already renamed)
    crop_files = sorted([f for f in folder.iterdir() if f.suffix.lower() == '.jpg' and '_yolo' not in f.stem and 'frame_' in f.stem])
    
    if not crop_files:
        print(f"[WARNING] No crop files found in {folder}")
        return
    
    print(f"\n[INFO] Processing folder: {folder}")
    print(f"       Found {len(crop_files)} crop files")
    
    renamed_count = 0
    # Track index per class
    class_counters = {cls_id: 0 for cls_id in CLASS_MAP}
    
    for crop_file in crop_files:
        # Detect class
        cls_id = get_dominant_class(crop_file, yolo_model)
        
        if cls_id not in CLASS_MAP:
            print(f"    [SKIP] {crop_file.name} - class {cls_id} not in map")
            continue
        
        fruit_name = CLASS_MAP[cls_id]
        class_counters[cls_id] += 1
        idx = class_counters[cls_id]
        
        # New name for crop file
        new_crop_name = f"{fruit_name}_class_{idx:03d}.jpg"
        new_crop_path = folder / new_crop_name
        
        # Check for corresponding yolo file
        yolo_file = folder / crop_file.name.replace('_crop', '_yolo')
        
        # Rename crop file
        crop_file.rename(new_crop_path)
        print(f"    [OK] {crop_file.name} -> {new_crop_name}")
        renamed_count += 1
        
        # Rename yolo file if exists
        if yolo_file.exists():
            new_yolo_name = f"{fruit_name}_class_{idx:03d}_yolo.jpg"
            new_yolo_path = folder / new_yolo_name
            yolo_file.rename(new_yolo_path)
            print(f"    [OK] {yolo_file.name} -> {new_yolo_name}")
            renamed_count += 1
    
    print(f"    [DONE] Renamed {renamed_count} files")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Rename files based on YOLO class detection")
    ap.add_argument("--folder", required=True, help="Folder containing images to rename")
    ap.add_argument("--weights", default="warehouse_check_standalone/weights/best.pt", help="YOLO weights")
    args = ap.parse_args()
    
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = SCRIPT_DIR / folder
    
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = SCRIPT_DIR / weights_path
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    
    print(f"\n{'='*60}")
    print(f"YOLO-based File Renamer")
    print(f"{'='*60}")
    print(f"Folder: {folder}")
    print(f"Classes: {CLASS_MAP}")
    
    print("\n[INFO] Loading YOLO model...")
    yolo_model = YOLO(str(weights_path))
    print("[INFO] Model loaded!")
    
    rename_files_in_folder(folder, yolo_model)
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
