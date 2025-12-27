"""
Canonical Video Processor
Processes videos with segmentation + QR-based canonical rotation pipeline
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple

# Add parent directory to path to import utils
SCRIPT_DIR = Path(__file__).parent.resolve()
UTILS_DIR = SCRIPT_DIR / "warehouse_check_standalone"
sys.path.insert(0, str(UTILS_DIR))

from utils.qr_reader import decode_qr, get_qr_center
from utils.segmentation import (
    load_u2net_model, segment_image, process_mask, 
    force_rectangle_mask, get_mask_corners
)
from utils.canonicalize import make_canonical


def process_frame_canonical(
    frame: np.ndarray,
    u2net_model,
    last_valid_result: Optional[dict] = None
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Process a single frame through the canonical pipeline.
    
    Returns:
        (canonical_image, result_info)
    """
    result = {
        "qr_found": False,
        "qr_center": None,
        "rotation": None,
        "success": False
    }
    
    # Step 1: Decode QR
    qr_text, qr_points = decode_qr(frame)
    if qr_text:
        result["qr_found"] = True
        result["qr_center"] = get_qr_center(qr_points)
    
    # Step 2: U2Net Segmentation
    try:
        mask = segment_image(frame, u2net_model)
        mask = process_mask(mask)
        mask = force_rectangle_mask(mask)
    except Exception as e:
        # If segmentation fails, return original frame or last valid
        if last_valid_result and last_valid_result.get("canonical") is not None:
            return last_valid_result["canonical"], result
        return frame, result
    
    # Step 3: Get mask corners and warp
    corners = get_mask_corners(mask)
    if corners is None:
        if last_valid_result and last_valid_result.get("canonical") is not None:
            return last_valid_result["canonical"], result
        return frame, result
    
    # Calculate warp matrix
    src_points = corners.astype(np.float32)
    w1 = np.linalg.norm(src_points[0] - src_points[1])
    w2 = np.linalg.norm(src_points[3] - src_points[2])
    width = int(max(w1, w2))
    h1 = np.linalg.norm(src_points[0] - src_points[3])
    h2 = np.linalg.norm(src_points[1] - src_points[2])
    height = int(max(h1, h2))
    
    if width < 10 or height < 10:
        if last_valid_result and last_valid_result.get("canonical") is not None:
            return last_valid_result["canonical"], result
        return frame, result
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    # Step 4: Transform QR center and apply canonical rotation
    if qr_points is not None:
        qr_center_orig = np.mean(qr_points, axis=0).reshape(1, 1, 2)
        qr_center_transformed = cv2.perspectiveTransform(qr_center_orig, M)
        qr_center_warped = qr_center_transformed.reshape(2)
    else:
        # Fallback: use center of warped image
        qr_center_warped = np.array([width // 2, height // 2], dtype=np.float32)
    
    # Apply canonical rotation
    canonical, rot_code = make_canonical(warped, qr_center_warped)
    
    result["rotation"] = rot_code
    result["success"] = True
    result["canonical"] = canonical
    
    return canonical, result


def resize_with_padding(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit target size while keeping aspect ratio, add black padding"""
    h, w = img.shape[:2]
    
    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize keeping aspect ratio
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create black canvas and center the image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def process_video(
    u2net_model,
    video_path: Path,
    output_path: Path,
    skip: int,
    show_preview: bool = True,
    output_size: Tuple[int, int] = (800, 600)  # Fixed output size (width, height)
) -> None:
    """Process a single video and save the canonical output."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = output_size

    print(f"\n[INFO] Processing: {video_path.name}")
    print(f"       FPS: {fps:.2f}, Frames: {total_frames}")
    print(f"       Output: {output_path.name} ({out_w}x{out_h})")

    # Initialize video writer with fixed size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if show_preview:
        win = f"Canonical - {video_path.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, out_w, out_h)

    frame_idx = 0
    last_valid_result = None
    skip_video = False
    
    qr_found_count = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed if needed
        if skip <= 1 or (frame_idx % skip == 0):
            canonical, result = process_frame_canonical(frame, u2net_model, last_valid_result)
            
            if result["success"]:
                last_valid_result = result
                success_count += 1
            if result["qr_found"]:
                qr_found_count += 1
            
            # Resize to fixed output size
            if canonical is not None:
                vis = resize_with_padding(canonical, out_w, out_h)
            else:
                # Fallback to resized original frame
                vis = resize_with_padding(frame, out_w, out_h)
            
            # Add debug info overlay
            status = f"Frame {frame_idx}/{total_frames}"
            if result["qr_found"]:
                status += " | QR: Found"
            else:
                status += " | QR: Not found"
            if result["rotation"]:
                status += f" | Rot: {result['rotation']}"
            
            cv2.putText(vis, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            writer.write(vis)
        else:
            # For skipped frames, use last valid result
            if last_valid_result and last_valid_result.get("canonical") is not None:
                vis = resize_with_padding(last_valid_result["canonical"], out_w, out_h)
            else:
                vis = resize_with_padding(frame, out_w, out_h)
            writer.write(vis)

        # Show progress
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            progress = (frame_idx / total_frames) * 100
            print(f"\r       Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | QR: {qr_found_count} | Success: {success_count}", end="", flush=True)

        if show_preview:
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                skip_video = True
                break
            if key == ord('s'):
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
        print(f"       Stats: QR found in {qr_found_count} frames, {success_count} successful transforms")


def main():
    ap = argparse.ArgumentParser(description="Batch video processor with segmentation + canonical rotation")
    ap.add_argument("--video_dir", default="video", help="directory containing input videos")
    ap.add_argument("--output_dir", default="output", help="directory to save output videos")
    ap.add_argument("--weights", default="warehouse_check_standalone/weights/u2net_last.pth", help="path to U2Net weights")
    ap.add_argument("--skip", type=int, default=1, help="process every N frames (1=all)")
    ap.add_argument("--no_preview", action="store_true", help="disable video preview window")
    args = ap.parse_args()

    # Resolve paths relative to script directory
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = SCRIPT_DIR / video_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = SCRIPT_DIR / weights_path

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"U2Net weights not found: {weights_path}")

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
    print(f"Canonical Video Batch Processor")
    print(f"{'='*60}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"U2Net weights: {weights_path}")
    print(f"Found {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    print(f"{'='*60}")

    # Load U2Net model once
    print("\n[INFO] Loading U2Net model...")
    u2net_model = load_u2net_model(str(weights_path))
    print("[INFO] Model loaded successfully!")

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] ", end="")
        
        output_path = output_dir / f"canonical_{video_path.stem}.mp4"
        
        process_video(
            u2net_model=u2net_model,
            video_path=video_path,
            output_path=output_path,
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
