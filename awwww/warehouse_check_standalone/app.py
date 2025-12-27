"""
Warehouse Check Standalone - FastAPI Backend
"""
import os
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Utils
from utils.qr_reader import decode_qr, parse_qr_payload, load_qr_metadata, get_qr_center
from utils.detection import load_yolo_model, detect_objects, validate_detections, get_box_bbox
from utils.segmentation import (
    load_u2net_model, segment_image, process_mask, 
    force_rectangle_mask, get_mask_corners, warp_perspective, visualize_segmentation
)
from utils.canonicalize import make_canonical

# Paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
QR_META_DIR = BASE_DIR / "qr_meta"
OUTPUT_DIR = BASE_DIR / "output"

# Create output dir
OUTPUT_DIR.mkdir(exist_ok=True)

# Global models
yolo_model = None
u2net_model = None

app = FastAPI(title="Warehouse Check Standalone")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_models():
    """Initialize models on startup"""
    global yolo_model, u2net_model
    
    yolo_path = WEIGHTS_DIR / "best.pt"
    u2net_path = WEIGHTS_DIR / "u2net_last.pth"
    
    if yolo_path.exists():
        yolo_model = load_yolo_model(str(yolo_path))
    else:
        print(f"[WARNING] YOLO weights not found at {yolo_path}")
    
    if u2net_path.exists():
        u2net_model = load_u2net_model(str(u2net_path))
    else:
        print(f"[WARNING] U2Net weights not found at {u2net_path}")


@app.on_event("startup")
async def startup():
    init_models()


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    """Check system status"""
    return {
        "yolo_loaded": yolo_model is not None,
        "u2net_loaded": u2net_model is not None,
        "qr_meta_count": len(list(QR_META_DIR.glob("*.json"))) if QR_META_DIR.exists() else 0
    }


def process_video_frame(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """Extract a frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def extract_video_frames(video_path: str, num_frames: int = 4) -> list:
    """Extract frames from video that contain readable QR codes"""
    import base64
    from utils.qr_reader import decode_qr
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # Sample more frames to find ones with QR
    sample_step = max(1, total_frames // 20)  # Sample up to 20 frames
    sample_indices = list(range(0, total_frames, sample_step))
    
    frames_with_qr = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Check if QR code can be decoded
            qr_text, qr_points = decode_qr(frame)
            if qr_text:  # QR found
                # Convert to base64 thumbnail
                thumbnail = cv2.resize(frame, (160, 120))
                _, buffer = cv2.imencode('.jpg', thumbnail)
                b64 = base64.b64encode(buffer).decode('utf-8')
                frames_with_qr.append({
                    "index": idx,
                    "thumbnail": b64,
                    "qr": qr_text
                })
                
                # Stop if we have enough frames
                if len(frames_with_qr) >= num_frames:
                    break
    
    cap.release()
    
    # If no QR frames found, fallback to evenly spaced frames
    if not frames_with_qr:
        print("[WARNING] No frames with QR found, using evenly spaced frames")
        cap = cv2.VideoCapture(video_path)
        step = max(1, total_frames // num_frames)
        for i in range(num_frames):
            idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                thumbnail = cv2.resize(frame, (160, 120))
                _, buffer = cv2.imencode('.jpg', thumbnail)
                b64 = base64.b64encode(buffer).decode('utf-8')
                frames_with_qr.append({
                    "index": idx,
                    "thumbnail": b64,
                    "qr": None
                })
        cap.release()
    
    return frames_with_qr


def process_image(image: np.ndarray, stage: str = "before") -> dict:
    """
    Full processing pipeline for an image
    
    Returns dict with all results and visualizations
    """
    global yolo_model, u2net_model
    
    result = {
        "success": False,
        "stage": stage,
        "qr_info": None,
        "qr_center": None,
        "detections": [],
        "validation": None,
        "mask": None,
        "images": {}
    }
    
    # Step 1: Decode QR
    qr_text, qr_points = decode_qr(image)
    if qr_text:
        result["qr_info"] = parse_qr_payload(qr_text)
        result["qr_center"] = get_qr_center(qr_points).tolist() if qr_points is not None else None
        
        # Load metadata
        qr_id = result["qr_info"].get("_qr")
        if qr_id:
            meta = load_qr_metadata(qr_id, str(QR_META_DIR))
            if meta:
                result["qr_meta"] = meta
    else:
        # QR not found - continue processing without QR info
        result["qr_info"] = {"_qr": "unknown", "_raw": "QR not detected"}
        qr_points = None
        print(f"[WARNING] QR code not found in {stage} image, continuing without QR")
    
    # Step 2: YOLO Detection
    if yolo_model is None:
        result["error"] = "YOLO model not loaded"
        return result
    
    detections, vis_yolo = detect_objects(image, yolo_model)
    result["detections"] = detections
    result["images"]["yolo"] = vis_yolo
    
    # Validate if we have expected items from QR
    expected = {}
    if result.get("qr_meta") and "fruits" in result["qr_meta"]:
        expected = result["qr_meta"]["fruits"]
    
    if expected:
        result["validation"] = validate_detections(detections, expected)
    else:
        result["validation"] = {"passed": True, "message": "No validation required"}
    
    # Step 3: U2Net Segmentation
    if u2net_model is None:
        result["error"] = "U2Net model not loaded"
        return result
    
    mask = segment_image(image, u2net_model)
    mask = process_mask(mask)
    mask = force_rectangle_mask(mask)
    result["mask"] = mask
    
    vis_seg = visualize_segmentation(image, mask)
    result["images"]["segmentation"] = vis_seg
    
    # Step 4: Warp and Canonicalize
    corners = get_mask_corners(mask)
    if corners is not None:
        # Calculate warp matrix
        src_points = corners.astype(np.float32)
        w1 = np.linalg.norm(src_points[0] - src_points[1])
        w2 = np.linalg.norm(src_points[3] - src_points[2])
        width = int(max(w1, w2))
        h1 = np.linalg.norm(src_points[0] - src_points[3])
        h2 = np.linalg.norm(src_points[1] - src_points[2])
        height = int(max(h1, h2))
        
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, (width, height))
        result["images"]["warped"] = warped
        
        # Transform QR center through warp matrix
        qr_center_warped = None
        if qr_points is not None:
            qr_center_orig = np.mean(qr_points, axis=0).reshape(1, 1, 2)
            qr_center_transformed = cv2.perspectiveTransform(qr_center_orig, M)
            qr_center_warped = qr_center_transformed.reshape(2)
            print(f"[INFO] QR center transformed: {qr_center_warped}")
        
        # Fallback: use center of warped image
        if qr_center_warped is None:
            qr_center_warped = np.array([width // 2, height // 2], dtype=np.float32)
            print(f"[WARNING] Using image center as QR position fallback")
        
        canonical, rot_code = make_canonical(warped, qr_center_warped)
        result["images"]["canonical"] = canonical
        result["rotation"] = rot_code
    
    result["success"] = True
    return result


@app.post("/api/extract-frames")
async def extract_frames(
    video: UploadFile = File(...),
    num_frames: int = Form(4)
):
    """Extract thumbnail frames from uploaded video"""
    import tempfile
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as f:
            f.write(await video.read())
            video_path = f.name
        
        frames = extract_video_frames(video_path, num_frames)
        
        os.unlink(video_path)
        
        return {"success": True, "frames": frames, "count": len(frames)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/process")
async def process_images(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    before_frame: int = Form(0),
    after_frame: int = Form(0)
):
    """
    Process before and after images/videos
    """
    try:
        # Save uploaded files temporarily
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(before.filename)[1]) as f:
            f.write(await before.read())
            before_path = f.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(after.filename)[1]) as f:
            f.write(await after.read())
            after_path = f.name
        
        # Extract frames
        if before_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            before_img = process_video_frame(before_path, before_frame)
        else:
            before_img = cv2.imread(before_path)
        
        if after_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            after_img = process_video_frame(after_path, after_frame)
        else:
            after_img = cv2.imread(after_path)
        
        # Clean up temp files
        os.unlink(before_path)
        os.unlink(after_path)
        
        if before_img is None:
            raise HTTPException(400, "Failed to read 'before' image/video")
        if after_img is None:
            raise HTTPException(400, "Failed to read 'after' image/video")
        
        # Process both
        before_result = process_image(before_img, "before")
        after_result = process_image(after_img, "after")
        
        # Resize canonical images to same size for comparison (with padding to keep aspect ratio)
        before_canonical = before_result.get("images", {}).get("canonical")
        after_canonical = after_result.get("images", {}).get("canonical")
        
        if before_canonical is not None and after_canonical is not None:
            # Get target size (use larger dimensions)
            h1, w1 = before_canonical.shape[:2]
            h2, w2 = after_canonical.shape[:2]
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            
            def resize_with_padding(img, target_w, target_h):
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
            
            before_result["images"]["canonical"] = resize_with_padding(before_canonical, target_w, target_h)
            after_result["images"]["canonical"] = resize_with_padding(after_canonical, target_w, target_h)
            print(f"[INFO] Resized canonical images to {target_w}x{target_h} (with padding)")
        
        # Save outputs
        box_id = None
        if before_result.get("qr_info"):
            box_id = before_result["qr_info"].get("_qr", "unknown")
        elif after_result.get("qr_info"):
            box_id = after_result["qr_info"].get("_qr", "unknown")
        else:
            box_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output folder
        output_folder = OUTPUT_DIR / box_id
        output_folder.mkdir(exist_ok=True)
        
        # Save images
        saved_files = []
        
        if "canonical" in before_result.get("images", {}):
            path = output_folder / "before_canonical.jpg"
            cv2.imwrite(str(path), before_result["images"]["canonical"])
            saved_files.append(str(path))
        
        if "canonical" in after_result.get("images", {}):
            path = output_folder / "after_canonical.jpg"
            cv2.imwrite(str(path), after_result["images"]["canonical"])
            saved_files.append(str(path))
        
        # Also save YOLO visualizations
        if "yolo" in before_result.get("images", {}):
            path = output_folder / "before_yolo.jpg"
            cv2.imwrite(str(path), before_result["images"]["yolo"])
            saved_files.append(str(path))
        
        if "yolo" in after_result.get("images", {}):
            path = output_folder / "after_yolo.jpg"
            cv2.imwrite(str(path), after_result["images"]["yolo"])
            saved_files.append(str(path))
        
        # Convert images to base64 for preview
        import base64
        
        def img_to_base64(img):
            if img is None:
                return None
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "box_id": box_id,
            "output_folder": str(output_folder),
            "saved_files": saved_files,
            "before": {
                "qr_info": before_result.get("qr_info"),
                "validation": before_result.get("validation"),
                "rotation": before_result.get("rotation"),
                "preview": {
                    "yolo": img_to_base64(before_result.get("images", {}).get("yolo")),
                    "segmentation": img_to_base64(before_result.get("images", {}).get("segmentation")),
                    "canonical": img_to_base64(before_result.get("images", {}).get("canonical"))
                }
            },
            "after": {
                "qr_info": after_result.get("qr_info"),
                "validation": after_result.get("validation"),
                "rotation": after_result.get("rotation"),
                "preview": {
                    "yolo": img_to_base64(after_result.get("images", {}).get("yolo")),
                    "segmentation": img_to_base64(after_result.get("images", {}).get("segmentation")),
                    "canonical": img_to_base64(after_result.get("images", {}).get("canonical"))
                }
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/output/{box_id}/{filename}")
async def get_output_image(box_id: str, filename: str):
    """Serve saved output images"""
    path = OUTPUT_DIR / box_id / filename
    if path.exists():
        return FileResponse(path)
    raise HTTPException(404, "Image not found")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
