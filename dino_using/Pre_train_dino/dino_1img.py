import torch
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import cv2

# =============== Load model ===============
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

# =============== Preprocess ===============
patch_size = dinov2.patch_size
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

img = Image.open("outputs\sum\03_template.png").convert("RGB")
img_t = transform(img).unsqueeze(0)

# =============== Feature map ===============
with torch.no_grad():
    feats_dict = dinov2.forward_features(img_t)
    patch_tokens = feats_dict["x_norm_patchtokens"]  # (1, num_patches, dim)

patch_tokens_np = patch_tokens[0].cpu().numpy()

# =============== PCA (1 component) ===============
pca = PCA(n_components=1)
comp1 = pca.fit_transform(patch_tokens_np)[:, 0]
comp1 = (comp1 - comp1.min()) / (comp1.max() - comp1.min())
comp1 = (comp1 * 255).astype(np.uint8)

# Map thành grid
grid_size = int(np.sqrt(len(comp1)))
feat_map = comp1.reshape(grid_size, grid_size)

# Upscale bằng Bicubic
feat_map = cv2.resize(feat_map, img.size, interpolation=cv2.INTER_CUBIC)

# Áp colormap để tạo heatmap
heatmap = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)

# Làm mượt bằng Gaussian Blur
heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

# =============== Save ===============
cv2.imwrite("outputs/dino_featuremap_heatmap.png", heatmap)
print("✅ Saved: dino_featuremap_heatmap.png")
