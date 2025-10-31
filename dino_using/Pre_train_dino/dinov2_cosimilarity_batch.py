import os, argparse, cv2, torch, numpy as np, json
from PIL import Image, ImageOps
from torchvision import transforms as T

# ================== CẤU HÌNH ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = "dinov2_vitb14_reg"
PATCH_SIZE = 14
USE_FP16 = True
SIM_SMOOTH_GAUSS = 3
MIN_BBOX_AREA = 0.001

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

def resize_pair_to_same_multiple(img1_pil, img2_pil, multiple=PATCH_SIZE):
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

# ================== TIỆN ÍCH XỬ LÝ ẢNH ==================
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

def feat_to_vis(feat, img_hw):
    feat_avg = feat.mean(axis=-1)
    feat_up = upsample_to_img(feat_avg.numpy(), img_hw)
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
        if w * h < min_area:
            continue
        if w * h < ((PATCH_SIZE + 6) * (PATCH_SIZE + 6)) or w < min_wh or h < min_wh:
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, boxes

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

# ================== TIỆN ÍCH CROP ẢNH ==================
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

def load_json_bboxes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ================== LOGIC CHÍNH: 1 CẶP ẢNH ==================
def compare_images(img1_path, img2_path, ckpt_path=None, sim_threshold=None, save_prefix="diff"):
    model = load_dinov2(ckpt_path=ckpt_path)

    pil1_raw = pil_load(img1_path)
    pil2_raw = pil_load(img2_path)
    pil1, pil2, (H, W) = resize_pair_to_same_multiple(pil1_raw, pil2_raw, multiple=PATCH_SIZE)

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
    mask, (x0, y0, rw, rh) = make_roi_mask(sim_up, roi_w=880, roi_h=400, thr_in=thr_in, thr_out=thr_out)
    print(f"[ROI] Center ROI {rw}x{rh} at (x={x0}, y={y0}) | thr_in={thr_in:.2f}, thr_out={thr_out:.2f}")

    cv2.rectangle(overlay_bgr, (x0, y0), (x0+rw, y0+rh), (255, 255, 255), 2)

    out_dir = os.path.join("outputs", "sum", "cos_sim")
    os.makedirs(out_dir, exist_ok=True)

    out_heat   = os.path.join(out_dir, f"{save_prefix}_heat.jpg")
    out_ovly   = os.path.join(out_dir, f"{save_prefix}_overlay.jpg")
    out_mask   = os.path.join(out_dir, f"{save_prefix}_mask.png")
    out_boxed1 = os.path.join(out_dir, f"{save_prefix}_boxed_img1.jpg")
    out_boxed2 = os.path.join(out_dir, f"{save_prefix}_boxed_img2.jpg")
    out_feat1  = os.path.join(out_dir, f"{save_prefix}_feat1.jpg")
    out_feat2  = os.path.join(out_dir, f"{save_prefix}_feat2.jpg")
    out_json   = os.path.join(out_dir, f"{save_prefix}_bboxes.json")

    boxed1, _ = draw_bboxes(img1_bgr, mask)
    boxed2, _ = draw_bboxes(img2_bgr, mask)
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

    cv2.imwrite(out_feat1, feat_to_vis(f1, (H, W)))
    cv2.imwrite(out_feat2, feat_to_vis(f2, (H, W)))
    cv2.imwrite(out_heat, heat_bgr)
    cv2.imwrite(out_ovly, boxed_overlay)
    cv2.imwrite(out_mask, mask)
    cv2.imwrite(out_boxed1, boxed1)
    cv2.imwrite(out_boxed2, boxed2)

    print(f"[DONE] Saved:\n  {out_heat}\n  {out_ovly}\n  {out_mask}\n  {out_boxed1}\n  {out_boxed2}\n  {out_feat1}\n  {out_feat2}\n  {out_json}")
    print(f"[INFO] Patches: {Hp} x {Wp} (patch size {PATCH_SIZE}), Image: {H}x{W}")
    if boxes:
        print("[BBOX] Different regions:")
        for (x,y,w,h) in boxes:
            print(f"  x={x} y={y} w={w} h={h}")
    else:
        print("[BBOX] No large different regions found (after filtering).")

    return out_json, boxes

def crop_bbox_images(json_path, output_dir, crop_size=96, img_source="img1", prefix_for_files=None):
    bbox_data = load_json_bboxes(json_path)

    if img_source == "img1":
        img_path = bbox_data["image_info"]["img1_path"]
    else:
        img_path = bbox_data["image_info"]["img2_path"]

    print(f"[INFO] Đọc ảnh từ: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Không thể đọc ảnh: {img_path}")
        return []

    H, W = img.shape[:2]
    print(f"[INFO] Kích thước ảnh: {W}x{H}")

    os.makedirs(output_dir, exist_ok=True)

    bboxes = bbox_data["bounding_boxes"]
    print(f"[INFO] Tìm thấy {len(bboxes)} bounding boxes")
    if len(bboxes) == 0:
        print("[WARN] Không có bounding box nào để crop")
        return []

    cropped_images = []
    for bbox in bboxes:
        bbox_id = bbox["id"]
        center_x = bbox["center"]["x"]
        center_y = bbox["center"]["y"]
        cropped = crop_image_around_center(img, center_x, center_y, crop_size)
        base_name = f"bbox_{bbox_id:03d}_center_{center_x}_{center_y}_{crop_size}x{crop_size}.jpg"
        if prefix_for_files:
            filename = f"{prefix_for_files}__{base_name}"
        else:
            filename = base_name
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)
        cropped_images.append(output_path)
        print(f"[CROP] Bbox {bbox_id}: center=({center_x},{center_y}) -> {filename}")

    summary_path = os.path.join(output_dir, "crop_summary.txt")
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"Pair: {prefix_for_files or os.path.splitext(os.path.basename(json_path))[0]}\n")
        f.write(f"Source image: {img_path}\n")
        f.write(f"Image size: {W}x{H}\n")
        f.write(f"Crop size: {crop_size}x{crop_size}\n")
        f.write(f"Total crops: {len(cropped_images)}\n\n")

    print(f"[DONE] Đã crop {len(cropped_images)} ảnh vào thư mục: {output_dir}")
    return cropped_images

# ================== BATCH MODE ==================
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def list_images(dir_path):
    if not dir_path:
        return []
    items = []
    for n in os.listdir(dir_path):
        p = os.path.join(dir_path, n)
        if os.path.isfile(p) and os.path.splitext(n.lower())[1] in IMG_EXTS:
            items.append(p)
    items.sort()
    return items

def pair_same_names(dir_a, dir_b):
    a = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(dir_a)}
    b = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(dir_b)}
    common = sorted(set(a.keys()) & set(b.keys()))
    pairs = []
    for k in common:
        pairs.append((a[k], b[k], k))
    print(f"[PAIR] same_names: {len(pairs)} cặp chung tên")
    return pairs

def pair_cross(dir_a, dir_b):
    la = list_images(dir_a)
    lb = list_images(dir_b)
    pairs = []
    for pa in la:
        for pb in lb:
            name_a = os.path.splitext(os.path.basename(pa))[0]
            name_b = os.path.splitext(os.path.basename(pb))[0]
            pairs.append((pa, pb, f"{name_a}__VS__{name_b}"))
    print(f"[PAIR] cross: {len(pairs)} cặp tổ hợp (|A|={len(la)}, |B|={len(lb)})")
    return pairs

def run_batch(mode, dir_a, dir_b, ckpt, sim_threshold, prefix, crop_size, crop_source,
              outputs_cos_dir=None, outputs_crop_dir=None, limit_pairs=None):
    if mode == "same_names":
        pairs = pair_same_names(dir_a, dir_b)
    elif mode == "cross":
        pairs = pair_cross(dir_a, dir_b)
    else:
        print("[ERROR] Batch mode không hợp lệ.")
        return

    if limit_pairs is not None:
        pairs = pairs[:int(limit_pairs)]

    cos_dir = outputs_cos_dir or os.path.join("outputs", "sum", "cos_sim")
    crop_dir = outputs_crop_dir or os.path.join("outputs", "sum", "cropped_images")
    os.makedirs(cos_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    done = 0
    for img1_path, img2_path, pair_name in pairs:
        save_prefix = f"{prefix}_{pair_name}"
        print(f"\n==== [BATCH] {done+1}/{len(pairs)}: {pair_name} ====")
        json_path, boxes = compare_images(img1_path, img2_path, ckpt, sim_threshold, save_prefix)
        if not boxes:
            print("[BATCH] Không bbox, bỏ qua crop.")
        else:
            crop_bbox_images(json_path, crop_dir, crop_size, crop_source, prefix_for_files=pair_name)
        done += 1
    print(f"\n[BATCH] Hoàn thành {done}/{len(pairs)} cặp.")

# ================== WORKFLOW ĐƠN GIẢN ==================
def full_workflow(img1_path, img2_path, ckpt_path=None, sim_threshold=0.35, 
                  save_prefix="diff", crop_size=96, crop_source="img1", 
                  crop_output_dir=None):
    print("🚀 Bắt đầu workflow hoàn chỉnh...")
    print("\n=== BƯỚC 1: Phân tích similarity và tạo JSON ===")
    json_path, boxes = compare_images(img1_path, img2_path, ckpt_path, sim_threshold, save_prefix)
    if not boxes:
        print("❌ Không tìm thấy bounding box nào, dừng workflow")
        return None, []
    print("\n=== BƯỚC 2: Cắt ảnh từ JSON ===")
    if crop_output_dir is None:
        crop_output_dir = os.path.join("outputs", "sum", "cropped_images")
    cropped_images = crop_bbox_images(json_path, crop_output_dir, crop_size, crop_source, prefix_for_files=save_prefix)
    print("\n🎉 Workflow hoàn thành thành công!")
    print("📁 Kết quả:")
    print(f"   - JSON: {json_path}")
    print(f"   - Cropped images: {crop_output_dir}")
    print(f"   - Total crops: {len(cropped_images) if cropped_images else 0}")
    return json_path, cropped_images

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="DINOv2 Cosine Similarity (single and batch)")
    parser.add_argument("--mode", choices=["compare", "crop", "full", "batch_same", "batch_cross"], default="full",
                       help="compare/crop/full cho 1 cặp; batch_same: ghép tên giống; batch_cross: tổ hợp 2 thư mục")

    # single compare/crop
    parser.add_argument("--img1", default=r"outputs\sum\03_template.png", help="Đường dẫn ảnh template")
    parser.add_argument("--img2", default=r"outputs\sum\04_moving_aligned.png", help="Đường dẫn ảnh moving")
    parser.add_argument("--ckpt", default=r".cache\checkpoints\dinov2_vitb14_reg4_pretrain.pth", help="Đường dẫn checkpoint DINOv2")
    parser.add_argument("--sim_threshold", type=float, default=0.35, help="Ngưỡng similarity")
    parser.add_argument("--prefix", default="diff", help="Prefix cho file output")

    parser.add_argument("--json", default=r"outputs\sum\cos_sim\diff_bboxes.json", help="File JSON (cho crop mode)")
    parser.add_argument("--crop_output", default=r"outputs\sum\cropped_images", help="Thư mục lưu ảnh crop")
    parser.add_argument("--crop_size", type=int, default=96, help="Kích thước ảnh crop")
    parser.add_argument("--crop_source", choices=["img1", "img2"], default="img1", help="Ảnh nào để crop")

    # batch
    parser.add_argument("--dir1", default=r"", help="Thư mục A")
    parser.add_argument("--dir2", default=r"", help="Thư mục B")
    parser.add_argument("--limit_pairs", type=int, default=None, help="Giới hạn số cặp để chạy")
    parser.add_argument("--outputs_cos", default=r"outputs\sum\cos_sim", help="Thư mục lưu heat/overlay/json")
    parser.add_argument("--outputs_crop", default=r"outputs\sum\cropped_images", help="Thư mục lưu crop (một thư mục chung)")

    args = parser.parse_args()

    if args.mode == "compare":
        compare_images(args.img1, args.img2, args.ckpt, args.sim_threshold, args.prefix)

    elif args.mode == "crop":
        if not os.path.exists(args.json):
            print(f"[ERROR] File JSON không tồn tại: {args.json}")
            print(f"[INFO] Hãy chạy với --mode compare trước để tạo file JSON")
            return
        crop_bbox_images(args.json, args.crop_output, args.crop_size, args.crop_source, prefix_for_files=args.prefix)

    elif args.mode == "full":
        full_workflow(args.img1, args.img2, args.ckpt, args.sim_threshold, args.prefix, args.crop_size, args.crop_source, args.crop_output)

    elif args.mode == "batch_same":
        if not args.dir1 or not args.dir2:
            print("[ERROR] Cần --dir1 và --dir2 cho batch_same")
            return
        run_batch("same_names", args.dir1, args.dir2, args.ckpt, args.sim_threshold, args.prefix,
                  args.crop_size, args.crop_source, args.outputs_cos, args.outputs_crop, args.limit_pairs)

    elif args.mode == "batch_cross":
        if not args.dir1 or not args.dir2:
            print("[ERROR] Cần --dir1 và --dir2 cho batch_cross")
            return
        run_batch("cross", args.dir1, args.dir2, args.ckpt, args.sim_threshold, args.prefix,
                  args.crop_size, args.crop_source, args.outputs_cos, args.outputs_crop, args.limit_pairs)

if __name__ == "__main__":
    main()


