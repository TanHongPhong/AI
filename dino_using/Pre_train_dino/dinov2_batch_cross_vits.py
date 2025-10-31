import os, argparse, cv2, torch, numpy as np, json
from PIL import Image, ImageOps
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = "dinov2_vits14_reg"
PATCH_SIZE = 14
USE_FP16 = True
SIM_SMOOTH_GAUSS = 3
MIN_BBOX_AREA = 0.002
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def ecc_align_and_crop_to_patchmultiple(img1_path, img2_path, iterations=100, patch_size=PATCH_SIZE):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Không thể đọc ảnh đầu vào cho ECC.")
    h1, w1 = img1.shape[:2]
    img2_rs = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(img2_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned2 = cv2.warpAffine(img2_rs, warp_matrix, (w1, h1), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        aligned2 = img2_rs
    newH = h1 - (h1 % patch_size)
    newW = w1 - (w1 % patch_size)
    if newH <= 0 or newW <= 0:
        raise ValueError("Kích thước ảnh quá nhỏ sau khi cắt cho bội số patch.")
    img1_c = img1[0:newH, 0:newW]
    img2_c = aligned2[0:newH, 0:newW]
    pil1 = Image.fromarray(cv2.cvtColor(img1_c, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB))
    return pil1, pil2, (newH, newW)

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

def make_roi_mask(sim_up, roi_w=880, roi_h=400, thr_in=0.45, thr_out=0.25):
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
    out = img_bgr.copy()
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        if w * h < ((PATCH_SIZE * 2) * (PATCH_SIZE * 2)) or w < min_wh * 2 or h < min_wh * 2:
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, boxes

def crop_image_around_center(img, center_x, center_y, crop_size=96):
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

def compare_images(img1_path, img2_path, ckpt_path=None, sim_threshold=None, save_prefix="diff", model=None):
    if model is None:
        model = load_dinov2(ckpt_path=ckpt_path)
    pil1, pil2, (H, W) = ecc_align_and_crop_to_patchmultiple(img1_path, img2_path, iterations=100, patch_size=PATCH_SIZE)
    t1, t2 = to_tensor(pil1), to_tensor(pil2)
    f1, (Hp, Wp) = extract_patch_tokens(model, t1)
    f2, _        = extract_patch_tokens(model, t2)
    f1_np = f1.numpy()
    f2_np = f2.numpy()
    sim_map = (f1_np * f2_np).sum(axis=-1)
    diff_map = 1.0 - sim_map
    diff_up  = upsample_to_img(diff_map, (H, W))
    if SIM_SMOOTH_GAUSS > 0:
        diff_up = cv2.GaussianBlur(diff_up, (0,0), SIM_SMOOTH_GAUSS)
    img1_bgr = cv2.cvtColor(np.array(pil1), cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)
    heat_bgr = colorize_heatmap(diff_up)
    overlay_bgr = overlay_heatmap(img1_bgr, heat_bgr, alpha=0.2)
    sim_up = upsample_to_img(sim_map, (H, W))
    thr_in  = sim_threshold if sim_threshold is not None else 0.35
    thr_out = thr_in / 2.0
    mask, (x0, y0, rw, rh) = make_roi_mask(sim_up, roi_w=870, roi_h=440, thr_in=thr_in, thr_out=thr_out)
    cv2.rectangle(overlay_bgr, (x0, y0), (x0+rw, y0+rh), (255, 255, 255), 2)
    out_dir = os.path.join("outputs", "sum", "cos_sim")
    os.makedirs(out_dir, exist_ok=True)
    out_ovly   = os.path.join(out_dir, f"{save_prefix}_overlay.jpg")
    out_json   = os.path.join(out_dir, f"{save_prefix}_bboxes.json")
    boxed_overlay, boxes = draw_bboxes(overlay_bgr, mask)
    bbox_data = {
        "image_info": {
            "img1_path": img1_path,
            "img2_path": img2_path,
            "image_size": {"width": W, "height": H},
            "patch_size": PATCH_SIZE,
            "patches": {"width": Wp, "height": Hp}
        },
        "roi_info": {
            "roi_size": {"width": rw, "height": rh},
            "roi_position": {"x": x0, "y": y0},
            "thresholds": {"thr_in": thr_in, "thr_out": thr_out}
        },
        "bounding_boxes": []
    }
    for i, (x, y, w, h) in enumerate(boxes):
        bbox_info = {
            "id": i,
            "position": {"x": int(x), "y": int(y)},
            "size": {"width": int(w), "height": int(h)},
            "center": {"x": int(x + w/2), "y": int(y + h/2)},
            "area": int(w * h)
        }
        bbox_data["bounding_boxes"].append(bbox_info)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(bbox_data, f, indent=2, ensure_ascii=False)
    cv2.imwrite(out_ovly, boxed_overlay)
    return out_json, boxes

def load_json_bboxes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def crop_bbox_images(json_path, output_dir, crop_size=96, img_source="img1", prefix_for_files=None):
    bbox_data = load_json_bboxes(json_path)
    img_path = bbox_data["image_info"]["img1_path"] if img_source == "img1" else bbox_data["image_info"]["img2_path"]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Không thể đọc ảnh: {img_path}")
        return []
    os.makedirs(output_dir, exist_ok=True)
    cropped_images = []
    for bbox in bbox_data["bounding_boxes"]:
        bbox_id = bbox["id"]
        cx = bbox["center"]["x"]
        cy = bbox["center"]["y"]
        cropped = crop_image_around_center(img, cx, cy, crop_size)
        # short, consistent filename
        base_name = f"{prefix_for_files}__{bbox_id:03d}.jpg" if prefix_for_files else f"patch_{bbox_id:03d}.jpg"
        output_path = os.path.join(output_dir, base_name)
        cv2.imwrite(output_path, cropped)
        cropped_images.append(output_path)
    return cropped_images

def list_images(dir_path):
    items = []
    for n in os.listdir(dir_path):
        p = os.path.join(dir_path, n)
        if os.path.isfile(p) and os.path.splitext(n.lower())[1] in IMG_EXTS:
            items.append(p)
    items.sort()
    return items

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

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Batch - Cross Pairing (ViT-S)")
    parser.add_argument("--dir1", default=r"data_canonical_1\le1", help="Thư mục A")
    parser.add_argument("--dir2", default=r"data_canonical_1\le2", help="Thư mục B")
    parser.add_argument("--ckpt", default=r".cache\checkpoints\dinov2_vits14_reg4_pretrain.pth")
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--prefix", default="diff")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--crop_source", choices=["img1", "img2"], default="img1")
    parser.add_argument("--outputs_cos", default=r"outputs\sum\cos_sim")
    parser.add_argument("--outputs_crop", default=r"outputs\sum\cropped_images")
    parser.add_argument("--limit_pairs", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda", help="Thiết bị chạy: cuda/cpu/auto")
    args = parser.parse_args()

    # Configure device preference
    global DEVICE
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng nhưng --device=cuda được yêu cầu.")
        DEVICE = "cuda"
    elif args.device == "cpu":
        DEVICE = "cpu"
    else:  # auto
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Load backbone once and reuse
    model = load_dinov2(ckpt_path=args.ckpt)

    pairs = pair_cross(args.dir1, args.dir2)
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    os.makedirs(args.outputs_cos, exist_ok=True)
    os.makedirs(args.outputs_crop, exist_ok=True)
    total = len(pairs)
    for idx, (pa, pb, name) in enumerate(pairs, 1):
        print(f"\n==== [BATCH CROSS VITS] {idx}/{total}: {name} ====")
        save_prefix = f"{args.prefix}_{name}"
        json_path, boxes = compare_images(pa, pb, None, args.sim_threshold, save_prefix, model=model)
        if boxes:
            crop_bbox_images(json_path, args.outputs_crop, args.crop_size, args.crop_source, prefix_for_files=name)
        else:
            print("[INFO] No boxes, skip cropping.")
    print(f"\n[FINISHED] {total} pairs processed.")

if __name__ == "__main__":
    main()


