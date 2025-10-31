#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_change3d_cross.py — Pipeline cross-pairing giữa 2 thư mục: ECC -> DINOv2 -> Change3D predict -> Label boxes

Pipeline:
1. Tìm các subfolder chung giữa 2 thư mục (ví dụ: le1, le2, ...)
2. Cross pair các ảnh giữa các subfolder
3. Với mỗi pair:
   - ECC alignment
   - DINOv2 tìm vùng khác biệt (chỉ trong ROI)
   - Crop 128x128 và predict bằng Change3D
   - Vẽ nhãn lên ảnh và lưu kết quả

Usage:
    Chỉnh sửa các biến trong hàm main() và nhấn Run
"""

import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF

# ================== CẤU HÌNH DINOv2 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = "dinov2_vits14_reg"
PATCH_SIZE = 14
USE_FP16 = True
SIM_SMOOTH_GAUSS = 3
MIN_BBOX_AREA = 0.002
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

# ================== TIỆN ÍCH DINOv2 ==================
def load_dinov2(arch=ARCH, ckpt_path=None):
    if ckpt_path and os.path.isfile(ckpt_path):
        model = torch.hub.load("facebookresearch/dinov2", arch, source="github", pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[DINOv2] Loaded local ckpt. Missing:{len(missing)} Unexpected:{len(unexpected)}")
    else:
        model = torch.hub.load("facebookresearch/dinov2", arch)
    model.eval().to(DEVICE)
    return model

def pil_load(path):
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def resize_to_multiple(img_pil, multiple=PATCH_SIZE):
    w, h = img_pil.size
    max_side = 1024
    scale = min(max_side / max(h, w), 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img_pil.resize((new_w, new_h), Image.BICUBIC)
    pad_w = (multiple - new_w % multiple) % multiple
    pad_h = (multiple - new_h % multiple) % multiple
    if pad_w or pad_h:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
    return img

def resize_pair_to_same_multiple(img1_pil, img2_pil, multiple=PATCH_SIZE, max_side=1024):
    i1 = resize_to_multiple(img1_pil, multiple)
    i2 = resize_to_multiple(img2_pil, multiple)

    w1, h1 = i1.size
    w2, h2 = i2.size

    tgt_w = max(w1, w2)
    tgt_h = max(h1, h2)
    tgt_w += (multiple - tgt_w % multiple) % multiple
    tgt_h += (multiple - tgt_h % multiple) % multiple

    if (w1, h1) != (tgt_w, tgt_h):
        i1 = ImageOps.expand(i1, border=(0, 0, tgt_w - w1, tgt_h - h1), fill=(0, 0, 0))
    if (w2, h2) != (tgt_w, tgt_h):
        i2 = ImageOps.expand(i2, border=(0, 0, tgt_w - w2, tgt_h - h2), fill=(0, 0, 0))
    return i1, i2, (tgt_h, tgt_w)

def to_tensor(img_pil):
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(img_pil).unsqueeze(0)

@torch.no_grad()
def extract_patch_tokens(model, img_t):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and USE_FP16)):
        out = model.forward_features(img_t.to(DEVICE))
    feats = None
    for k in ["x_norm_patchtokens", "x_prenorm", "x"]:
        if isinstance(out, dict) and k in out:
            feats = out[k]
            break
    if feats is None:
        feats = out if isinstance(out, torch.Tensor) else None
    assert feats is not None, "Không tìm thấy patch tokens từ forward_features output."
    B, N, D = feats.shape
    H, W = img_t.shape[-2:]
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
    feats = feats[:, :Hp*Wp, :]
    feats = feats.reshape(B, Hp, Wp, D)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats[0].float().cpu(), (Hp, Wp)

def upsample_to_img(sim_map, img_hw):
    Hp, Wp = sim_map.shape
    H, W = img_hw
    sim_np = sim_map.astype(np.float32)
    sim_up = cv2.resize(sim_np, (W, H), interpolation=cv2.INTER_CUBIC)
    return sim_up

def make_roi_mask(sim_up, roi_w=880, roi_h=400, thr_in=0.6, thr_out=0.3):
    H, W = sim_up.shape
    rw = min(int(roi_w), W)
    rh = min(int(roi_h), H)
    x0 = (W - rw) // 2
    y0 = (H - rh) // 2

    mask = (sim_up < float(thr_out)).astype(np.uint8) * 255
    roi_m = (sim_up[y0:y0+rh, x0:x0+rw] < float(thr_in)).astype(np.uint8) * 255
    mask[y0:y0+rh, x0:x0+rw] = roi_m
    return mask, (x0, y0, rw, rh)

def draw_bboxes(img_bgr, mask_u8, min_area_ratio=MIN_BBOX_AREA, min_wh=PATCH_SIZE):
    H, W = mask_u8.shape
    min_area = H * W * min_area_ratio
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        if w * h < ((PATCH_SIZE*2) * (PATCH_SIZE*2)) or w < min_wh*2 or h < min_wh*2:
            continue
        boxes.append((x, y, w, h))
    return boxes

# ================== ECC ALIGNMENT ==================
def to_gray_f32(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0

def ecc_align_images(img1_bgr: np.ndarray, img2_bgr: np.ndarray, motion='affine', iters=100, eps=1e-6):
    """Căn chỉnh img2 với img1 bằng ECC"""
    H, W = img1_bgr.shape[:2]
    if img2_bgr.shape[:2] != (H, W):
        img2_bgr = cv2.resize(img2_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    tm = to_gray_f32(img1_bgr)
    mv = to_gray_f32(img2_bgr)

    motion_map = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean': cv2.MOTION_EUCLIDEAN,
        'affine': cv2.MOTION_AFFINE,
        'homography': cv2.MOTION_HOMOGRAPHY
    }
    mt = motion_map[motion]

    Wmat = np.eye(3, dtype=np.float32) if mt == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    try:
        cc, Wmat = cv2.findTransformECC(tm, mv, Wmat, mt, criteria)
    except cv2.error as e:
        cc = -1.0
        Wmat = np.eye(3, dtype=np.float32) if mt == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)

    if mt == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(
            img2_bgr, Wmat, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        img2_aligned = cv2.warpAffine(
            img2_bgr, Wmat, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

    return img2_aligned, cc

# ================== CHANGE3D MODEL ==================
class ConvBNReLU3d(nn.Module):
    def __init__(self, in_c, out_c, k=(3,3,3), s=(1,1,1), p=(1,1,1)):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SE3d(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Conv3d(c, c // r, kernel_size=1)
        self.fc2 = nn.Conv3d(c // r, c, kernel_size=1)
    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class Change3DNet(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 6, se_before_head: bool = False, drop: float = 0.0):
        super().__init__()
        self.stem    = ConvBNReLU3d(in_ch, 32, k=(3,3,3), s=(1,1,1), p=(1,1,1))
        self.stage1a = ConvBNReLU3d(32, 32, k=(3,3,3), s=(1,1,1), p=(1,1,1))
        self.stage1b = ConvBNReLU3d(32, 64, k=(2,3,3), s=(1,2,2), p=(0,1,1))
        self.stage2a = ConvBNReLU3d(64, 128, k=(1,3,3), s=(1,2,2), p=(0,1,1))
        self.stage2b = ConvBNReLU3d(128, 128, k=(1,3,3), s=(1,1,1), p=(0,1,1))
        self.se = SE3d(128) if se_before_head else nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1a(x)
        x = self.stage1b(x)
        x = self.stage2a(x)
        x = self.stage2b(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

def load_change3d_model(ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes: List[str] = ckpt.get('classes') or list((ckpt.get('class_to_id') or {}).keys())
    if not classes:
        raise RuntimeError('Classes not found in checkpoint metadata')
    args_meta: Dict = ckpt.get('args', {})
    add_delta = bool(args_meta.get('add_delta', False))
    se_before_head = bool(args_meta.get('se_before_head', False))
    dropout = float(args_meta.get('dropout', 0.0))
    in_ch = 6 if add_delta else 3

    model = Change3DNet(in_ch=in_ch, num_classes=len(classes), se_before_head=se_before_head, drop=dropout)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, classes, add_delta

def build_input_tensor(img1: np.ndarray, img2: np.ndarray, size: int, add_delta: bool) -> torch.Tensor:
    """Build input tensor from two numpy arrays (BGR images)"""
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    if img1_pil.size != (size, size):
        img1_pil = img1_pil.resize((size, size), Image.BILINEAR)
    if img2_pil.size != (size, size):
        img2_pil = img2_pil.resize((size, size), Image.BILINEAR)
    
    t1 = TF.to_tensor(img1_pil)
    t2 = TF.to_tensor(img2_pil)
    x = torch.stack([t1, t2], dim=1)  # [C,2,H,W]
    if add_delta:
        delta = torch.abs(t2 - t1)
        x = torch.cat([x, delta.unsqueeze(1).repeat(1,2,1,1)], dim=0)  # [C+3,2,H,W]
    return x.unsqueeze(0)  # [1,C(, +3),2,H,W]

def crop_image_around_center(img: np.ndarray, center_x: int, center_y: int, crop_size: int = 128) -> np.ndarray:
    """Cắt ảnh xung quanh tâm với kích thước crop_size x crop_size"""
    H, W = img.shape[:2]
    half_size = crop_size // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(W, center_x + half_size)
    y2 = min(H, center_y + half_size)
    cropped = img[y1:y2, x1:x2]
    if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        pad_y1 = (crop_size - cropped.shape[0]) // 2
        pad_x1 = (crop_size - cropped.shape[1]) // 2
        padded[pad_y1:pad_y1+cropped.shape[0], pad_x1:pad_x1+cropped.shape[1]] = cropped
        cropped = padded
    return cropped

# ================== MÀU SẮC CHO TỪNG CLASS ==================
def get_class_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """Tạo danh sách màu BGR cho các class"""
    colors = [
        (255, 0, 0),     # Đỏ
        (0, 255, 0),     # Xanh lá
        (0, 0, 255),     # Xanh dương
        (255, 255, 0),   # Vàng
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (128, 0, 128),   # Tím
        (255, 165, 0),   # Cam
        (255, 192, 203), # Hồng
        (0, 128, 128),   # Teal
    ]
    while len(colors) < num_classes:
        colors.append((np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    return colors[:num_classes]

# ================== TIỆN ÍCH CROSS PAIRING ==================
def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS

def list_images(dir_path):
    items = []
    for n in os.listdir(dir_path):
        p = os.path.join(dir_path, n)
        if os.path.isfile(p) and is_image_file(n):
            items.append(p)
    items.sort()
    return items

def list_image_subdirs(root_dir):
    subdirs = []
    if not os.path.isdir(root_dir):
        return subdirs
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if os.path.isdir(sub) and len(list_images(sub)) > 0:
            subdirs.append(sub)
    preferred_names = {"le1", "le2", "le3", "le4", "tao1", "tao2", "tao3", "tao4"}
    preferred = [d for d in subdirs if os.path.basename(os.path.normpath(d)) in preferred_names]
    if preferred:
        return preferred
    return subdirs

def pair_cross(dir_a, dir_b):
    la = list_images(dir_a)
    lb = list_images(dir_b)
    pairs = []
    for pa in la:
        for pb in lb:
            name_a = os.path.splitext(os.path.basename(pa))[0]
            name_b = os.path.splitext(os.path.basename(pb))[0]
            pairs.append((pa, pb, f"{name_a}__VS__{name_b}"))
    return pairs

# ================== PIPELINE CHÍNH ==================
def find_diff_regions(img1_bgr: np.ndarray, img2_bgr: np.ndarray, dino_model, 
                      orig_h: int, orig_w: int, sim_threshold: float = 0.35) -> List[Tuple[int, int, int, int]]:
    """Tìm các vùng khác biệt và trả về danh sách bounding boxes (chỉ trong ROI)"""
    pil1_raw = Image.fromarray(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
    pil2_raw = Image.fromarray(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))
    pil1, pil2, (H, W) = resize_pair_to_same_multiple(pil1_raw, pil2_raw, multiple=PATCH_SIZE)

    t1, t2 = to_tensor(pil1), to_tensor(pil2)
    f1, (Hp, Wp) = extract_patch_tokens(dino_model, t1)
    f2, _ = extract_patch_tokens(dino_model, t2)

    f1_np = f1.numpy()
    f2_np = f2.numpy()
    sim_map = (f1_np * f2_np).sum(axis=-1)
    sim_up = upsample_to_img(sim_map, (H, W))
    
    if SIM_SMOOTH_GAUSS > 0:
        sim_up = cv2.GaussianBlur(sim_up, (0,0), SIM_SMOOTH_GAUSS)

    thr_in = sim_threshold
    thr_out = thr_in / 2.0
    mask, (x0, y0, rw, rh) = make_roi_mask(sim_up, roi_w=870, roi_h=440, thr_in=thr_in, thr_out=thr_out)
    
    boxes = draw_bboxes(None, mask, min_area_ratio=MIN_BBOX_AREA, min_wh=PATCH_SIZE)
    
    # Lọc boxes: chỉ giữ lại những boxes nằm trong ROI
    roi_boxes = []
    for x, y, w, h in boxes:
        center_x = x + w // 2
        center_y = y + h // 2
        if (x0 <= center_x <= x0 + rw) and (y0 <= center_y <= y0 + rh):
            roi_boxes.append((x, y, w, h))
        else:
            box_right = x + w
            box_bottom = y + h
            roi_right = x0 + rw
            roi_bottom = y0 + rh
            
            overlap_x1 = max(x, x0)
            overlap_y1 = max(y, y0)
            overlap_x2 = min(box_right, roi_right)
            overlap_y2 = min(box_bottom, roi_bottom)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                box_area = w * h
                overlap_ratio = overlap_area / box_area if box_area > 0 else 0
                if overlap_ratio > 0.5:
                    roi_boxes.append((x, y, w, h))
    
    scale_x = orig_w / W
    scale_y = orig_h / H
    
    scaled_boxes = []
    for x, y, w, h in roi_boxes:
        sx = int(x * scale_x)
        sy = int(y * scale_y)
        sw = int(w * scale_x)
        sh = int(h * scale_y)
        scaled_boxes.append((sx, sy, sw, sh))
    
    return scaled_boxes

def predict_cropped_pairs(img1: np.ndarray, img2: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                          change3d_model, classes: List[str], add_delta: bool, device: torch.device) -> List[Dict]:
    """Predict từng cặp ảnh đã crop"""
    predictions = []
    crop_size = 128
    
    with torch.no_grad():
        for i, (x, y, w, h) in enumerate(boxes):
            center_x = x + w // 2
            center_y = y + h // 2
            
            crop1 = crop_image_around_center(img1, center_x, center_y, crop_size)
            crop2 = crop_image_around_center(img2, center_x, center_y, crop_size)
            
            x_tensor = build_input_tensor(crop1, crop2, crop_size, add_delta).to(device)
            logits = change3d_model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_id = int(probs.argmax())
            pred_cls = classes[pred_id]
            prob = float(probs[pred_id])
            
            predictions.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'pred_class': pred_cls,
                'pred_id': pred_id,
                'prob': prob,
            })
    
    return predictions

def draw_labeled_boxes(img: np.ndarray, predictions: List[Dict], classes: List[str], 
                       class_colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Vẽ boxes có nhãn lên ảnh"""
    img_out = img.copy()
    
    for pred in predictions:
        x, y, w, h = pred['bbox']
        pred_id = pred['pred_id']
        pred_cls = pred['pred_class']
        prob = pred['prob']
        
        color = class_colors[pred_id]
        
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, 2)
        
        label = f"{pred_cls} ({prob:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x, y - text_h - 4), (x + text_w, y), color, -1)
        cv2.putText(img_out, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_out

def process_one_pair(img1_path: str, img2_path: str, pair_name: str, dino_model, change3d_model, 
                     classes: List[str], add_delta: bool, output_dir: str, sim_threshold: float):
    """Xử lý một cặp ảnh"""
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)
    if img1_bgr is None or img2_bgr is None:
        print(f"[SKIP] Không thể đọc ảnh: {img1_path} hoặc {img2_path}")
        return None
    
    orig_h, orig_w = img1_bgr.shape[:2]
    
    # ECC alignment
    img2_aligned, ecc_cc = ecc_align_images(img1_bgr, img2_bgr, motion='affine', iters=100)
    
    # Find bounding boxes (chỉ trong ROI)
    boxes = find_diff_regions(img1_bgr, img2_aligned, dino_model, orig_h, orig_w, sim_threshold)
    if not boxes:
        return None
    
    # Predict
    predictions = predict_cropped_pairs(img1_bgr, img2_aligned, boxes, change3d_model, classes, add_delta, torch.device(DEVICE))
    
    # Draw labeled boxes
    class_colors = get_class_colors(len(classes))
    img1_labeled = draw_labeled_boxes(img1_bgr, predictions, classes, class_colors)
    img2_labeled = draw_labeled_boxes(img2_aligned, predictions, classes, class_colors)
    
    # Save outputs
    out_img1 = os.path.join(output_dir, f"{pair_name}_labeled_img1.jpg")
    out_img2 = os.path.join(output_dir, f"{pair_name}_labeled_img2.jpg")
    out_json = os.path.join(output_dir, f"{pair_name}_predictions.json")
    
    cv2.imwrite(out_img1, img1_labeled)
    cv2.imwrite(out_img2, img2_labeled)
    
    json_data = {
        'img1_path': img1_path,
        'img2_path': img2_path,
        'num_boxes': len(predictions),
        'predictions': [
            {
                'bbox': {'x': b[0], 'y': b[1], 'w': b[2], 'h': b[3]},
                'center': {'x': p['center'][0], 'y': p['center'][1]},
                'pred_class': p['pred_class'],
                'pred_id': p['pred_id'],
                'prob': p['prob'],
            }
            for b, p in zip([pred['bbox'] for pred in predictions], predictions)
        ],
        'classes': classes,
        'ecc_cc': float(ecc_cc),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return len(predictions)

# ================== MAIN ==================
def main():
    # ================== CẤU HÌNH - CHỈNH SỬA CÁC GIÁ TRỊ DƯỚI ĐÂY ==================
    dir1 = r'data_canonical_3'  # Thư mục 1
    dir2 = r'data_canonical_4'  # Thư mục 2
    dino_ckpt = None  # Đường dẫn checkpoint DINOv2 (None để dùng pretrained từ hub)
    change3d_ckpt = r'runs\change3d\20251031_043257\best_change3d_dir.pth'
    output_dir = 'outputs/pipeline_cross_results'
    sim_threshold = 0.4
    limit_pairs = 0  # Giới hạn số cặp (0 = không giới hạn)
    # =================================================================================
    
    print("🚀 Bắt đầu pipeline cross-pairing...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models một lần
    print("\n=== Load models ===")
    dino_model = load_dinov2(ckpt_path=dino_ckpt)
    change3d_model, classes, add_delta = load_change3d_model(change3d_ckpt, torch.device(DEVICE))
    print(f"[INFO] Change3D classes: {classes}")
    
    # Discover subfolders và cross pairing
    dirs1 = list_image_subdirs(dir1) or [dir1]
    dirs2 = list_image_subdirs(dir2) or [dir2]
    map1 = {os.path.basename(os.path.normpath(d)): d for d in dirs1}
    map2 = {os.path.basename(os.path.normpath(d)): d for d in dirs2}
    common_keys = [k for k in sorted(map1.keys()) if k in map2]
    
    if not common_keys:
        print(f"[ERROR] Không tìm thấy subfolder chung giữa {dir1} và {dir2}")
        return
    
    print(f"[INFO] Tìm thấy {len(common_keys)} subfolder chung: {common_keys}")
    
    grand_total = 0
    grand_processed = 0
    
    for key in common_keys:
        dir_a = map1[key]
        dir_b = map2[key]
        print(f"\n==== [GROUP] {key}: {dir_a} x {dir_b} ====")
        pairs = pair_cross(dir_a, dir_b)
        
        if limit_pairs > 0:
            pairs = pairs[:limit_pairs]
        
        total = len(pairs)
        grand_total += total
        
        for idx, (pa, pb, name) in enumerate(pairs, 1):
            print(f"\n[{key}] {idx}/{total}: {name}")
            result = process_one_pair(pa, pb, f"{key}_{name}", dino_model, change3d_model, 
                                      classes, add_delta, output_dir, sim_threshold)
            if result is not None:
                grand_processed += 1
                print(f"[OK] {result} boxes predicted")
            else:
                print("[SKIP] No boxes found or error")
    
    print(f"\n🎉 Hoàn thành!")
    print(f"📊 Thống kê:")
    print(f"   - Tổng số pairs: {grand_total}")
    print(f"   - Pairs đã xử lý thành công: {grand_processed}")
    print(f"   - Output directory: {output_dir}")

if __name__ == "__main__":
    main()

