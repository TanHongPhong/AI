# dino_diff_heatmap.py
import os, argparse, cv2, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T

# ================== CẤU HÌNH ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = "dinov2_vitb14_reg"
PATCH_SIZE = 14
USE_FP16 = True
SIM_SMOOTH_GAUSS = 3
MIN_BBOX_AREA = 0.001

# ================== TIỆN ÍCH ==================
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
    # scale từng ảnh về <= max_side và pad về bội số 'multiple'
    i1 = resize_to_multiple(img1_pil, multiple)
    i2 = resize_to_multiple(img2_pil, multiple)

    w1, h1 = i1.size
    w2, h2 = i2.size

    tgt_w = max(w1, w2)
    tgt_h = max(h1, h2)
    # ép về bội số của 'multiple'
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

def colorize_heatmap(diff_up):
    m = diff_up
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = (m * 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)

def overlay_heatmap(img_bgr, heatmap_bgr, alpha=0.45):
    return cv2.addWeighted(heatmap_bgr, alpha, img_bgr, 1 - alpha, 0)

def feat_to_vis(feat, img_hw):  # feat: [Hp, Wp, D]
    feat_avg = feat.mean(axis=-1)  # [Hp, Wp]
    feat_up = upsample_to_img(feat_avg.numpy(), img_hw)  # [H,W]
    norm = (feat_up - feat_up.min()) / (feat_up.max() - feat_up.min() + 1e-8)
    vis = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def otsu_mask(diff_up):
    norm = (diff_up - diff_up.min()) / (diff_up.max() - diff_up.min() + 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (0,0), 1.2)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def draw_bboxes(img_bgr, mask_u8, min_area_ratio=MIN_BBOX_AREA, min_wh=PATCH_SIZE):
    H, W = mask_u8.shape
    min_area = H * W * min_area_ratio
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # 1) lọc theo tỉ lệ diện tích (như cũ)
        if w * h < min_area:
            continue
        # 2) lọc hộp nhỏ hơn 1 patch (14x14 px)
        if w * h < ((PATCH_SIZE + 6 ) * (PATCH_SIZE + 6)) or w < min_wh or h < min_wh:
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, boxes

def make_roi_mask(sim_up, roi_w=880, roi_h=400, thr_in=0.45, thr_out=0.25):
    """
    sim_up: 2D similarity map (H, W), giá trị cosine similarity
    ROI đặt ở chính giữa ảnh, kích thước roi_w x roi_h (w x h)
    thr_in:  ngưỡng trong ROI
    thr_out: ngưỡng ngoài ROI
    Trả về: (mask_u8, (x0, y0, w, h))
    """
    H, W = sim_up.shape
    rw = min(int(roi_w), W)
    rh = min(int(roi_h), H)
    x0 = (W - rw) // 2
    y0 = (H - rh) // 2

    # mask ngoài ROI dùng thr_out
    mask = (sim_up < float(thr_out)).astype(np.uint8) * 255
    # trong ROI dùng thr_in
    roi_m = (sim_up[y0:y0+rh, x0:x0+rw] < float(thr_in)).astype(np.uint8) * 255
    mask[y0:y0+rh, x0:x0+rw] = roi_m
    return mask, (x0, y0, rw, rh)



# ================== LOGIC CHÍNH ==================
def compare_images(img1_path, img2_path, ckpt_path=None, sim_threshold=None, save_prefix="diff"):
    model = load_dinov2(ckpt_path=ckpt_path)

    pil1_raw = pil_load(img1_path)
    pil2_raw = pil_load(img2_path)
    pil1, pil2, (H, W) = resize_pair_to_same_multiple(pil1_raw, pil2_raw, multiple=PATCH_SIZE)

    t1, t2 = to_tensor(pil1), to_tensor(pil2)
    f1, (Hp, Wp) = extract_patch_tokens(model, t1)
    f2, _        = extract_patch_tokens(model, t2)

    # ... (giữ nguyên code trực quan hoá)
    f1_np = f1.numpy()
    f2_np = f2.numpy()
    sim_map = (f1_np * f2_np).sum(axis=-1)   # giờ đã cùng shape [Hp, Wp]
    diff_map = 1.0 - sim_map
    diff_up  = upsample_to_img(diff_map, (H, W))
    if SIM_SMOOTH_GAUSS > 0:
        diff_up = cv2.GaussianBlur(diff_up, (0,0), SIM_SMOOTH_GAUSS)

    img1_bgr = cv2.cvtColor(np.array(pil1), cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)
    heat_bgr = colorize_heatmap(diff_up)
    overlay_bgr = overlay_heatmap(img1_bgr, heat_bgr, alpha=0.2)

    # luôn tính sim_up để dùng cho ROI
    sim_up = upsample_to_img(sim_map, (H, W))

    # ❌ BUG: dùng args ở ngoài scope
    # ✅ SỬA: dùng biến sim_threshold truyền vào hàm
    thr_in  = sim_threshold if sim_threshold is not None else 0.35
    thr_out = thr_in / 2.0
    mask, (x0, y0, rw, rh) = make_roi_mask(sim_up, roi_w=880, roi_h=400, thr_in=thr_in, thr_out=thr_out)
    print(f"[ROI] Center ROI {rw}x{rh} at (x={x0}, y={y0}) | thr_in={thr_in:.2f}, thr_out={thr_out:.2f}")

    cv2.rectangle(overlay_bgr, (x0, y0), (x0+rw, y0+rh), (255, 255, 255), 2)

    # đảm bảo thư mục tồn tại
    out_dir = os.path.join("outputs", "sum", "cos_sim")
    os.makedirs(out_dir, exist_ok=True)

    out_heat   = os.path.join(out_dir, f"{save_prefix}_heat.jpg")
    out_ovly   = os.path.join(out_dir, f"{save_prefix}_overlay.jpg")
    out_mask   = os.path.join(out_dir, f"{save_prefix}_mask.png")
    out_boxed1 = os.path.join(out_dir, f"{save_prefix}_boxed_img1.jpg")
    out_boxed2 = os.path.join(out_dir, f"{save_prefix}_boxed_img2.jpg")
    out_feat1  = os.path.join(out_dir, f"{save_prefix}_feat1.jpg")
    out_feat2  = os.path.join(out_dir, f"{save_prefix}_feat2.jpg")

    boxed1, _ = draw_bboxes(img1_bgr, mask)
    boxed2, _ = draw_bboxes(img2_bgr, mask)
    boxed_overlay, boxes = draw_bboxes(overlay_bgr, mask)

    cv2.imwrite(out_feat1, feat_to_vis(f1, (H, W)))
    cv2.imwrite(out_feat2, feat_to_vis(f2, (H, W)))
    cv2.imwrite(out_heat, heat_bgr)
    cv2.imwrite(out_ovly, boxed_overlay)
    cv2.imwrite(out_mask, mask)
    cv2.imwrite(out_boxed1, boxed1)
    cv2.imwrite(out_boxed2, boxed2)

    print(f"[DONE] Saved:\n  {out_heat}\n  {out_ovly}\n  {out_mask}\n  {out_boxed1}\n  {out_boxed2}\n  {out_feat1}\n  {out_feat2}")
    print(f"[INFO] Patches: {Hp} x {Wp} (patch size {PATCH_SIZE}), Image: {H}x{W}")
    if boxes:
        print("[BBOX] Different regions:")
        for (x,y,w,h) in boxes:
            print(f"  x={x} y={y} w={w} h={h}")
    else:
        print("[BBOX] No large different regions found (after filtering).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", default=r"outputs\sum\03_template.png")
    ap.add_argument("--img2", default=r"outputs\sum\04_moving_aligned.png")
    ap.add_argument("--ckpt", default=r".cache\checkpoints\dinov2_vitb14_reg4_pretrain.pth")
    ap.add_argument("--sim_threshold", type=float, default=0.35)
    ap.add_argument("--prefix", default="diff")
    args = ap.parse_args()
    compare_images(args.img1, args.img2, ckpt_path=args.ckpt,
                   sim_threshold=args.sim_threshold, save_prefix=args.prefix)
