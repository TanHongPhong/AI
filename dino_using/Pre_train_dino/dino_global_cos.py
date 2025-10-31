# global_cosine_gradio.py
import os, torch, numpy as np, cv2, gradio as gr
from PIL import Image, ImageOps
from torchvision import transforms as T

# ================= CẤU HÌNH =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH   = "dinov2_vitb14_reg"
PATCH_SIZE = 14
USE_FP16   = True

# ================= TIỆN ÍCH =================
_MODEL = None
_MODEL_SRC = {"arch": None, "ckpt": None, "device": None}

def _pil_load(img: Image.Image) -> Image.Image:
    if img is None:
        return None
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def _preprocess(img_pil: Image.Image, max_side=1024) -> torch.Tensor:
    w, h = img_pil.size
    scale = min(max_side / max(h, w), 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img_pil.resize((new_w, new_h), Image.BICUBIC)

    # pad để chia hết PATCH_SIZE
    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
    if pad_w or pad_h:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img).unsqueeze(0)

@torch.no_grad()
def _extract_global_embed(model, img_t: torch.Tensor) -> np.ndarray:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and USE_FP16)):
        out = model.forward_features(img_t.to(DEVICE))

    if isinstance(out, dict) and "x_norm_clstoken" in out:
        v = out["x_norm_clstoken"]                    # [B, D]
    elif isinstance(out, dict) and "x_norm_patchtokens" in out:
        v = out["x_norm_patchtokens"].mean(dim=1)     # [B, N, D] -> [B, D]
    else:
        t = out["x"] if (isinstance(out, dict) and "x" in out) else out
        if isinstance(t, torch.Tensor) and t.dim() == 3:
            v = t.mean(dim=1)                         # [B, D]
        elif isinstance(t, torch.Tensor):
            v = t
        else:
            raise RuntimeError("Không lấy được embedding toàn cục từ forward_features.")

    v = torch.nn.functional.normalize(v, dim=-1)
    return v[0].float().cpu().numpy()                 # [D]

def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))

def _load_dinov2(arch=ARCH, ckpt_path=None):
    global _MODEL, _MODEL_SRC
    need_reload = (
        _MODEL is None or
        _MODEL_SRC["arch"] != arch or
        _MODEL_SRC["ckpt"] != (ckpt_path or "") or
        _MODEL_SRC["device"] != DEVICE
    )
    if not need_reload:
        return _MODEL

    if ckpt_path and os.path.isfile(ckpt_path):
        model = torch.hub.load("facebookresearch/dinov2", arch, source="github", pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        src_msg = f"[DINOv2] Loaded local ckpt: {ckpt_path}"
    else:
        model = torch.hub.load("facebookresearch/dinov2", arch)
        src_msg = "[DINOv2] Loaded pretrained weights from hub"

    model.eval().to(DEVICE)
    _MODEL, _MODEL_SRC = model, {"arch": arch, "ckpt": (ckpt_path or ""), "device": DEVICE}
    print(src_msg)
    return _MODEL

def _ecc_align_bgr(ref_bgr, mov_bgr, mode='affine'):
    try:
        if mode == 'homography':
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            def _warp(im, M):
                return cv2.warpPerspective(im, M, (ref_bgr.shape[1], ref_bgr.shape[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_REPLICATE)
        else:
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            def _warp(im, M):
                return cv2.warpAffine(im, M, (ref_bgr.shape[1], ref_bgr.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)

        ref = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (0,0), 1.2)
        mov = cv2.GaussianBlur(cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY), (0,0), 1.2)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
        cc, warp_matrix = cv2.findTransformECC(ref, mov, warp_matrix, warp_mode, criteria, None, 5)
        return _warp(mov_bgr, warp_matrix), cc
    except Exception:
        return None, None

# ================== HÀM CHẠY NÚT ==================
def run_similarity(img1: Image.Image, img2: Image.Image, use_ecc: bool, ckpt_path: str):
    logs = []
    if img1 is None or img2 is None:
        return None, "Vui lòng upload đủ 2 ảnh.", None

    # Load model (cache theo ckpt)
    try:
        model = _load_dinov2(ARCH, ckpt_path.strip() or None)
    except Exception as e:
        return None, f"Lỗi load DINOv2: {e}", None

    # Chuẩn hóa EXIF → RGB
    img1 = _pil_load(img1)
    img2 = _pil_load(img2)

    # (tùy chọn) ECC align ảnh 2 về ảnh 1
    aligned_preview = None
    if use_ecc:
        bgr1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        bgr2a, cc = _ecc_align_bgr(bgr1, bgr2, mode='affine')
        if bgr2a is not None:
            img2 = Image.fromarray(cv2.cvtColor(bgr2a, cv2.COLOR_BGR2RGB))
            aligned_preview = img2
            logs.append(f"[ECC] Alignment OK (affine), corr={cc:.6f}")
        else:
            logs.append("[ECC] Alignment failed → dùng ảnh 2 gốc.")

    # Preprocess và embedding
    try:
        t1 = _preprocess(img1)
        t2 = _preprocess(img2)
        g1 = _extract_global_embed(model, t1)
        g2 = _extract_global_embed(model, t2)
        sim = _cosine_sim(g1, g2)
        logs.append(f"[RESULT] Global Cosine Similarity = {sim:.6f}")
        return round(sim, 6), "\n".join(logs), aligned_preview
    except Exception as e:
        return None, f"Lỗi xử lý: {e}", None

# ================== GRADIO UI ==================
with gr.Blocks(theme=gr.themes.Monochrome(), title="DINOv2 Global Cosine Similarity") as demo:
    gr.Markdown("## DINOv2 — Global Cosine Similarity\nTải **2 ảnh** để so sánh. Tùy chọn **ECC alignment** giúp giảm lệch khung trước khi tính embedding.")
    with gr.Row():
        with gr.Column():
            img1 = gr.Image(type="pil", label="Ảnh 1", height=280)
        with gr.Column():
            img2 = gr.Image(type="pil", label="Ảnh 2", height=280)
    with gr.Row():
        use_ecc = gr.Checkbox(label="Dùng ECC alignment (affine) trước khi so", value=False)
        ckpt = gr.Textbox(label="Đường dẫn checkpoint DINOv2 (tùy chọn — để trống sẽ tải từ hub)", value="")
    run_btn = gr.Button("🚀 Tính Cosine Similarity", variant="primary")
    with gr.Row():
        sim_out = gr.Number(label="Cosine similarity", precision=6)
        aligned_preview = gr.Image(label="Ảnh 2 (sau ECC, nếu bật)", height=220)
    logs = gr.Textbox(label="Logs", lines=6)

    run_btn.click(
        fn=run_similarity,
        inputs=[img1, img2, use_ecc, ckpt],
        outputs=[sim_out, logs, aligned_preview]
    )

if __name__ == "__main__":
    # Mẹo: share=True nếu muốn truy cập qua mạng LAN/Internet
    demo.launch()