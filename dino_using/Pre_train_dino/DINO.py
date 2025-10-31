# dino_only_compare.py
# DINOv2 (ViT-B/14 REG) so sánh cosine similarity 2 ảnh, KHÔNG dùng SAM.
# Giao diện giữ nguyên như bản SAM+DINO trước đó.

import os, time, traceback
import torch
import numpy as np
import gradio as gr
from PIL import Image
from typing import Optional
from torchvision import transforms as T

# ===================== CẤU HÌNH =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DINOv2 (ViT-B/14 REG) ---
DINOV2_ARCH_CANDIDATES = ["dinov2_vitb14_reg", "dinov2_vitb14"]  # fallback nếu ckpt local không khớp
DINOV2_CKPT_PATH = r"D:\NCKH CODE\dinov2_vitb14_reg4_pretrain.pth"  # đổi theo chỗ bạn lưu
DINO_USE_FP16 = True
DINO_IMAGE_SIZE = 448  # bội số của 14 (14*32)

DEFAULT_SIM_THRESHOLD = 0.88  # ngưỡng cảnh báo cosine similarity

# ===================== TIỆN ÍCH =====================
torch.set_grad_enabled(False)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

def log(s: str):
    print(s, flush=True)

# ----------------- KHỞI TẠO DINOv2 -----------------
def build_dino_from_local(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Không thấy DINOv2 checkpoint: {ckpt_path}")
    last_err: Optional[Exception] = None
    for arch in DINOV2_ARCH_CANDIDATES:
        try:
            model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            model.load_state_dict(sd, strict=False)
            log(f"✅ Loaded DINOv2 ({arch}) from local .pth")
            return model
        except Exception as e:
            last_err = e
            log(f"⚠️ Try arch='{arch}' failed: {e}")
    raise RuntimeError(f"Load DINOv2 từ ckpt thất bại. Lỗi cuối: {last_err}")

def build_dino_fallback_from_hub():
    arch = DINOV2_ARCH_CANDIDATES[0]
    log(f"⏳ Falling back to hub: {arch} (pretrained=True)")
    return torch.hub.load("facebookresearch/dinov2", arch)

try:
    dino = build_dino_from_local(DINOV2_CKPT_PATH)
except Exception as e:
    log(f"⚠️ Local load failed, fallback hub: {e}")
    dino = build_dino_fallback_from_hub()

dino = dino.to(DEVICE).eval()
log("✅ DINOv2 (ViT-B/14 REG) ready.")

# ---- Chuẩn hoá ảnh cho DINOv2 (preview & embed) ----
PREVIEW_TRANSFORM = T.Compose([
    T.Resize(DINO_IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(DINO_IMAGE_SIZE),
])
DINO_TRANSFORM = T.Compose([
    T.Resize(DINO_IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(DINO_IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def dino_embed(pil_img: Image.Image) -> torch.Tensor:
    x = DINO_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        if DEVICE == "cuda" and DINO_USE_FP16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = dino(x)
        else:
            feat = dino(x)
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat  # shape (1, D)

# ================== PIPELINE (DINO-only) ======================
def compare_two_images_dino_only(
    img_before: Image.Image,
    img_after: Image.Image,
    # Các tham số dưới đây chỉ để giữ tương thích UI cũ (SAM), sẽ bị bỏ qua
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=3000,
    border_ratio=0.02,
    topk_masks=2,
    max_fg_ratio=0.45,
    min_edge_density=0.02,
    sim_threshold=DEFAULT_SIM_THRESHOLD
):
    try:
        if img_before is None or img_after is None:
            return (None, None, None, None, None, None, None, "❌ Vui lòng upload đủ 2 ảnh.")

        # ---- Preview ảnh đã tiền xử lý (để hiển thị ở ô 'Kết quả segment')
        prev_b = PREVIEW_TRANSFORM(img_before.convert("RGB"))
        prev_a = PREVIEW_TRANSFORM(img_after.convert("RGB"))

        # ---- Tính embedding & similarity
        feat_b = dino_embed(img_before.convert("RGB"))
        feat_a = dino_embed(img_after.convert("RGB"))
        sim = torch.nn.functional.cosine_similarity(feat_b, feat_a).item()
        verdict = "✅ MATCH" if sim >= float(sim_threshold) else "🚨 ALERT: similarity thấp"

        # ---- Lưu preview (cho 2 ô file ở UI cũ)
        os.makedirs("outputs", exist_ok=True)
        ts = int(time.time() * 1000)
        b_prev_path = os.path.abspath(f"outputs/before_preprocessed_{ts}.png")
        a_prev_path = os.path.abspath(f"outputs/after_preprocessed_{ts}.png")
        prev_b.save(b_prev_path)
        prev_a.save(a_prev_path)

        log_msg = (
            f"🔎 Cosine similarity (DINO-only): {sim:.4f} | Ngưỡng: {float(sim_threshold):.2f}\n"
            f"{verdict}\n"
            f"Before PREP: {b_prev_path}\nAfter  PREP: {a_prev_path}\n"
            f"Model: dinov2_vitb14_reg | ImageSize: {DINO_IMAGE_SIZE}"
        )

        # Trả về theo đúng thứ tự UI cũ:
        # [seg_before_img, seg_after_img, file_b_rgba, file_b_crop, file_a_rgba, file_a_crop, sim_txt, log_box]
        return (
            prev_b,                 # seg_before (giờ là preview)
            prev_a,                 # seg_after  (giờ là preview)
            b_prev_path,            # file_b_rgba  (dùng file preview)
            None,                   # file_b_crop  (không dùng)
            a_prev_path,            # file_a_rgba  (dùng file preview)
            None,                   # file_a_crop  (không dùng)
            f"{sim:.4f}",           # similarity text
            log_msg                 # logs
        )

    except Exception as e:
        print("❌ ERROR in compare_two_images_dino_only:\n", traceback.format_exc())
        return (None, None, None, None, None, None, None, f"❌ Lỗi: {type(e).__name__}: {e}")

# ================== GIAO DIỆN (giữ nguyên layout cũ) ======================
with gr.Blocks(title="DINOv2 | Similarity Checker (No-SAM)") as demo:
    gr.Markdown("## 🧪 **DINOv2-B/14 REG** (embedding) → So sánh hai ảnh **trước/sau** (không dùng SAM)")
    with gr.Row():
        with gr.Column():
            img_before = gr.Image(type="pil", label="Ảnh TRƯỚC (nhà cung cấp)")
            img_after  = gr.Image(type="pil", label="Ảnh SAU (băng chuyền)")
            with gr.Accordion("Tùy chỉnh (nâng cao)", open=False):
                # Giữ nguyên các slider cũ cho tương thích, nhưng chỉ **Ngưỡng similarity** có tác dụng
                pps  = gr.Slider(8, 128, value=32, step=8,  label="points_per_side (ignored)")
                iou  = gr.Slider(0.50, 0.99, value=0.90, step=0.01, label="pred_iou_thresh (ignored)")
                stab = gr.Slider(0.50, 0.99, value=0.95, step=0.01, label="stability_score_thresh (ignored)")
                area = gr.Slider(0, 20000, value=3000, step=50, label="min_mask_region_area (ignored)")
                br   = gr.Slider(0.002, 0.05, value=0.02, step=0.002, label="border_ratio (ignored)")
                topk = gr.Slider(1, 5, value=2, step=1, label="Top-K masks để union (ignored)")
                maxr = gr.Slider(0.25, 0.90, value=0.45, step=0.01, label="Tỉ lệ tối đa FG (ignored)")
                edge_min = gr.Slider(0.0, 0.20, value=0.02, step=0.005, label="Min edge density (ignored)")
                thr  = gr.Slider(0.50, 0.99, value=DEFAULT_SIM_THRESHOLD, step=0.01, label="Ngưỡng cảnh báo similarity")
            run_btn = gr.Button("Segment + So sánh ⚡")  # giữ nguyên nút

        with gr.Column():
            gr.Markdown("### Kết quả 'segment' (ở đây hiển thị **preview** ảnh cho DINO)")
            seg_before = gr.Image(label="Before (Preview for DINO)")
            seg_after  = gr.Image(label="After (Preview for DINO)")
            file_b_rgba = gr.File(label="Before - Ảnh đã tiền xử lý (PNG)")
            file_b_crop = gr.File(label="Before - (không dùng)")
            file_a_rgba = gr.File(label="After  - Ảnh đã tiền xử lý (PNG)")
            file_a_crop = gr.File(label="After  - (không dùng)")
            sim_txt = gr.Textbox(label="Cosine similarity", interactive=False)
            log_box = gr.Textbox(label="Logs & Kết luận", lines=6)

    run_btn.click(
        fn=compare_two_images_dino_only,
        inputs=[img_before, img_after, pps, iou, stab, area, br, topk, maxr, edge_min, thr],
        outputs=[seg_before, seg_after, file_b_rgba, file_b_crop, file_a_rgba, file_a_crop, sim_txt, log_box]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
