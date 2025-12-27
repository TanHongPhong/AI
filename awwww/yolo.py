import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict
from ultralytics import YOLO


def load_classes(classes_path: str) -> List[str]:
    if not classes_path:
        return []
    p = Path(classes_path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip()]


def draw_dets(frame: np.ndarray, dets: List[Dict], classes: List[str]) -> np.ndarray:
    vis = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cid = d["class_id"]
        conf = d["conf"]
        label = classes[cid] if classes and 0 <= cid < len(classes) else str(cid)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def yolo_detect(
    model: YOLO,
    image_bgr: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[Dict]:
    """Run YOLO detection using ultralytics."""
    results = model.predict(image_bgr, conf=conf_thres, iou=iou_thres, verbose=False)
    
    dets = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                dets.append({
                    "class_id": cls_id,
                    "conf": conf,
                    "bbox_xyxy": [x1, y1, x2, y2]
                })
    return dets


def process_video(
    model: YOLO,
    video_path: Path,
    output_path: Path,
    classes: List[str],
    conf_thres: float,
    iou_thres: float,
    skip: int,
    show_preview: bool = True
) -> None:
    """Process a single video and save the output."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\n[INFO] Processing: {video_path.name}")
    print(f"       Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    print(f"       Output: {output_path.name}")

    if show_preview:
        win = f"YOLO - {video_path.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

    frame_idx = 0
    last_vis = None
    skip_video = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # skip frames for speed if needed
        if skip <= 1 or (frame_idx % skip == 0):
            dets = yolo_detect(model, frame, conf_thres, iou_thres)
            last_vis = draw_dets(frame, dets, classes)
        else:
            last_vis = frame if last_vis is None else last_vis

        # Write frame to output video
        writer.write(last_vis)

        # Show progress
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            progress = (frame_idx / total_frames) * 100
            print(f"\r       Progress: {frame_idx}/{total_frames} ({progress:.1f}%)", end="", flush=True)

        if show_preview:
            cv2.imshow(win, last_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC to skip to next video
                skip_video = True
                break
            if key == ord('s'):  # S to skip preview for this video
                show_preview = False
                cv2.destroyWindow(win)

    print()  # New line after progress

    cap.release()
    writer.release()
    if show_preview:
        try:
            cv2.destroyWindow(win)
        except:
            pass

    if skip_video:
        print(f"       [SKIPPED] User pressed Q/ESC")
    else:
        print(f"       [DONE] Saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="video", help="directory containing input videos")
    ap.add_argument("--output_dir", default="output", help="directory to save output videos")
    ap.add_argument("--weights", default="warehouse_check_standalone/weights/best.pt", help="path to YOLO model (.pt)")
    ap.add_argument("--classes", default="", help="optional classes.txt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--skip", type=int, default=1, help="process every N frames (1=all)")
    ap.add_argument("--no_preview", action="store_true", help="disable video preview window")
    args = ap.parse_args()

    # Get script directory for relative paths
    script_dir = Path(__file__).parent.resolve()
    
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = script_dir / video_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = script_dir / weights_path

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = sorted([
        f for f in video_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in video_extensions
    ])

    if not video_files:
        raise FileNotFoundError(f"No video files found in: {video_dir}")

    print(f"\n{'='*60}")
    print(f"YOLO Video Batch Processor")
    print(f"{'='*60}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Weights: {weights_path}")
    print(f"Found {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    print(f"{'='*60}")

    classes = load_classes(args.classes) if args.classes else []

    # Load YOLO model once
    print("\n[INFO] Loading YOLO model...")
    model = YOLO(str(weights_path))
    print("[INFO] Model loaded successfully!")

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] ", end="")
        
        output_path = output_dir / f"detected_{video_path.stem}.mp4"
        
        process_video(
            model=model,
            video_path=video_path,
            output_path=output_path,
            classes=classes,
            conf_thres=args.conf,
            iou_thres=args.iou,
            skip=args.skip,
            show_preview=not args.no_preview
        )

    try:
        cv2.destroyAllWindows()
    except:
        pass

    print(f"\n{'='*60}")
    print(f"All videos processed! Check output in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
