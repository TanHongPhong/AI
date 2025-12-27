"""
Extract specific frames from canonical videos
Saves 3 versions: original frame + YOLO detection frame + cropped 700x500 from class 0 center
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
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(img_w, x1 + crop_w)
        y2 = min(img_h, y1 + crop_h)
    
    return image[y1:y2, x1:x2]


def extract_frames_with_yolo(video_path: Path, frame_numbers: list, output_dir: Path, yolo_model):
    """Extract specific frames from a video, save original + YOLO + cropped versions."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.stem
    
    print(f"\n[INFO] Processing: {video_path.name}")
    print(f"       Total frames: {total_frames}")
    print(f"       Extracting {len(frame_numbers)} frames")
    
    # Create output subdirectory for this video
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    crop_count = 0
    
    for frame_num in sorted(frame_numbers):
        if frame_num < 1 or frame_num > total_frames:
            print(f"    [SKIP] Frame {frame_num} out of range (1-{total_frames})")
            continue
        
        # Seek to frame (0-indexed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        
        if not ret:
            print(f"    [ERROR] Cannot read frame {frame_num}")
            continue
        
        # Run YOLO detection
        results = yolo_model.predict(frame, conf=0.25, iou=0.45, verbose=False)
        
        # Draw detections on frame copy + find class 0 for cropping
        yolo_vis = frame.copy()
        det_count = 0
        class_0_boxes = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(yolo_vis, f"{cls_id} {conf:.2f}", (x1, max(0, y1 - 6)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    det_count += 1
                    
                    # Collect class 0 detections for cropping
                    if cls_id == 0:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        class_0_boxes.append({
                            'center': (center_x, center_y),
                            'conf': conf
                        })
        
        # Save CROPPED frames for each class 0 detection (both yolo and clean versions)
        for idx, det in enumerate(class_0_boxes):
            center_x, center_y = det['center']
            
            # Crop from original frame (clean)
            cropped = crop_centered(frame, center_x, center_y, CROP_WIDTH, CROP_HEIGHT)
            # Crop from yolo visualization
            cropped_yolo = crop_centered(yolo_vis, center_x, center_y, CROP_WIDTH, CROP_HEIGHT)
            
            if cropped is not None and cropped.size > 0:
                if len(class_0_boxes) == 1:
                    crop_path = video_output_dir / f"frame_{frame_num:04d}_crop.jpg"
                    yolo_path = video_output_dir / f"frame_{frame_num:04d}_yolo.jpg"
                else:
                    crop_path = video_output_dir / f"frame_{frame_num:04d}_crop_{idx+1}.jpg"
                    yolo_path = video_output_dir / f"frame_{frame_num:04d}_yolo_{idx+1}.jpg"
                cv2.imwrite(str(crop_path), cropped)
                cv2.imwrite(str(yolo_path), cropped_yolo)
                crop_count += 1
        
        extracted_count += 1
        crop_info = f", {len(class_0_boxes)} crops" if class_0_boxes else ""
        print(f"    [OK] Frame {frame_num} ({det_count} dets{crop_info})")
    
    cap.release()
    print(f"    [DONE] Extracted {extracted_count} frames, {crop_count} crops")


def main():
    videos_frames = {
        # video 1
        "canonical_WIN_20251227_16_07_58_Pro.mp4": [
            183, 180, 185, 190, 195, 200, 414, 420, 425, 430, 435,
            687, 690, 695, 700, 706, 1018, 1020, 1025, 1030, 1035, 1040, 1045, 1051,
            1317, 1320, 1325, 1330, 1335, 1340, 1345, 1353,
            1641, 1645, 1650, 1655, 1657
        ],
        # video 2
        "canonical_WIN_20251227_16_05_47_Pro.mp4": [
            262, 265, 270, 275, 280, 282, 561, 565, 570, 575, 580, 586,
            851, 855, 860, 865, 1139, 1145, 1147,
            1588, 1590, 1595, 1600, 1620, 1640, 1660,
            2036, 2040, 2046, 2050, 2060, 2070, 2074
        ],
        # video 3
        "canonical_WIN_20251227_16_00_45_Pro.mp4": [
            231, 265, 547, 553, 560, 867, 875,
            1223, 1230, 1236, 1602, 1610, 1620, 1630, 1640, 1650,
            1999, 2010, 2020, 2030
        ]
    }
    
    video_dir = SCRIPT_DIR / "output"
    output_dir = SCRIPT_DIR / "output" / "extracted_frames"
    yolo_weights = SCRIPT_DIR / "warehouse_check_standalone" / "weights" / "best.pt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Frame Extraction: Origin + YOLO")
    print(f"{'='*60}")
    
    print("\n[INFO] Loading YOLO model...")
    yolo_model = YOLO(str(yolo_weights))
    print("[INFO] Model loaded!")
    
    for video_name, frame_numbers in videos_frames.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"\n[ERROR] Video not found: {video_path}")
            continue
        extract_frames_with_yolo(video_path, frame_numbers, output_dir, yolo_model)
    
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
