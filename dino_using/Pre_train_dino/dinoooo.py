import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T
import os

# ========== CẤU HÌNH ==========
ARCH = "dinov2_vitb14_reg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 14
MAX_SIZE = 1024

# ========== LOAD ==========
def load_model():
    model = torch.hub.load("facebookresearch/dinov2", ARCH)
    model.eval().to(DEVICE)
    return model

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # Resize về max_size, rồi pad bội số PATCH_SIZE
    w, h = img.size
    scale = min(MAX_SIZE / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
    img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))

    return img

def preprocess(img_pil):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_pil).unsqueeze(0)  # [1, 3, H, W]

# ========== TRÍCH XUẤT FEATURES ==========
@torch.no_grad()
def extract_patch_features(model, img_tensor):
    out = model.forward_features(img_tensor.to(DEVICE))
    feats = out["x_norm_patchtokens"]  # [1, N, D]
    return feats.squeeze(0).cpu().numpy()  # [N, D]

# ========== VISUALIZE ==========
def visualize_heatmap(feature_map, img_pil, save_prefix="feature"):
    H, W = img_pil.size
    Hp, Wp, D = feature_map.shape

    # 1. Trung bình kênh
    feat_avg = feature_map.mean(axis=-1)  # [Hp, Wp]

    # 2. Resize lên [H, W]
    feat_up = cv2.resize(feat_avg, (H, W), interpolation=cv2.INTER_CUBIC)

    # 3. Normalize → 0–255
    norm = (feat_up - feat_up.min()) / (feat_up.max() - feat_up.min() + 1e-8)
    heatmap = (norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 4. Overlay
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heatmap_color, 0.5, img_bgr, 0.5, 0)

    # 5. Save
    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/{save_prefix}_features.npy", feature_map)
    cv2.imwrite(f"outputs/{save_prefix}_heatmap.jpg", heatmap_color)
    cv2.imwrite(f"outputs/{save_prefix}_overlay.jpg", overlay)
    print(f"[DONE] Saved to outputs/:")
    print(f"  → {save_prefix}_features.npy  ({Hp}x{Wp} patches, {D}-dim)")
    print(f"  → {save_prefix}_heatmap.jpg")
    print(f"  → {save_prefix}_overlay.jpg")

# ========== MAIN ==========
if __name__ == "__main__":
    IMAGE_PATH = "img/img8.jpg"  # ← Thay ảnh ở đây
    SAVE_PREFIX = "feature"        # ← Tiền tố lưu file

    model = load_model()
    img_pil = load_image(IMAGE_PATH)
    img_tensor = preprocess(img_pil)

    H, W = img_tensor.shape[-2:]
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE

    features = extract_patch_features(model, img_tensor)  # [N, D]
    feature_map = features.reshape(Hp, Wp, -1)            # [Hp, Wp, D]

    visualize_heatmap(feature_map, img_pil, save_prefix=SAVE_PREFIX)
