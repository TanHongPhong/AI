# dino_single_dump.py
import os, argparse, json, cv2, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T

# ==== CẤU HÌNH ====
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ARCH        = "dinov2_vitb14_reg"
PATCH_SIZE  = 14
USE_FP16    = True
MAX_SIDE    = 1024

def load_dinov2(ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        model = torch.hub.load("facebookresearch/dinov2", ARCH, source="github", pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("[DINOv2] Loaded local ckpt.")
    else:
        print("[DINOv2] Local ckpt not found -> using torch.hub pretrained.")
        model = torch.hub.load("facebookresearch/dinov2", ARCH)
    return model.eval().to(DEVICE)

def load_preprocess(path):
    img = Image.open(path)
    try: img = ImageOps.exif_transpose(img)
    except: pass
    img = img.convert("RGB")
    # scale → pad về bội số PATCH_SIZE
    w, h = img.size
    s = min(MAX_SIDE / max(h, w), 1.0)
    nw, nh = int(round(w*s)), int(round(h*s))
    img = img.resize((nw, nh), Image.BICUBIC)
    pw = (PATCH_SIZE - nw % PATCH_SIZE) % PATCH_SIZE
    ph = (PATCH_SIZE - nh % PATCH_SIZE) % PATCH_SIZE
    if pw or ph: img = ImageOps.expand(img, border=(0,0,pw,ph), fill=(0,0,0))
    tfm = T.Compose([T.ToTensor(),
                     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    t = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return img, t

@torch.no_grad()
def forward_and_tokens(model, img_t):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and USE_FP16)):
        out = model.forward_features(img_t.to(DEVICE))
    # lấy patch tokens
    feats = None
    if isinstance(out, dict):
        for k in ("x_norm_patchtokens","x_prenorm","x"):
            if k in out and isinstance(out[k], torch.Tensor):
                feats = out[k]; break
    if feats is None:  # fallback nếu forward_features trả tensor
        feats = out if isinstance(out, torch.Tensor) else None
    assert feats is not None, "Không tìm thấy patch tokens trong forward_features."
    B, N, D = feats.shape
    H, W = img_t.shape[-2:]
    Hp, Wp = H//PATCH_SIZE, W//PATCH_SIZE
    feats = feats[:, :Hp*Wp, :].reshape(1, Hp, Wp, D)
    feats = torch.nn.functional.normalize(feats, dim=-1)[0].float().cpu()  # [Hp,Wp,D]
    # cls (nếu có)
    cls_tok = None
    if isinstance(out, dict):
        for k in ("x_norm_clstoken","x_cls","cls_token"):
            if k in out and isinstance(out[k], torch.Tensor):
                cls_tok = out[k][0].float().cpu()
                if cls_tok.dim()==2 and cls_tok.shape[0]==1: cls_tok = cls_tok.squeeze(0)
                break
    return out, feats, (Hp,Wp), (H,W), cls_tok

def upsample_map(m, out_hw):
    H, W = out_hw
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

def save_heatmaps(feat_3d, img_pil, prefix):
    # avg theo kênh → upsample → colormap
    avg = feat_3d.mean(dim=-1).numpy()            # [Hp,Wp]
    up  = upsample_map(avg, img_pil.size[::-1])   # [H,W] (PIL size is (W,H))
    gray = ( (up - up.min()) / (up.max() - up.min() + 1e-8) * 255 ).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat, 0.45, img_bgr, 0.55, 0)
    cv2.imwrite(f"outputs/{prefix}_feat_avg.jpg", heat)
    cv2.imwrite(f"outputs/{prefix}_feat_overlay.jpg", overlay)

def dump_meta_forward(out, path_json):
    meta = {"forward_features_keys": {}}
    if isinstance(out, dict):
        for k,v in out.items():
            try: meta["forward_features_keys"][k] = list(v.shape)
            except: meta["forward_features_keys"][k] = str(type(v))
    else:
        try: meta["forward_features_keys"]["<tensor>"] = list(out.shape)
        except: meta["forward_features_keys"]["<unknown>"] = str(type(out))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="img/img8.jpg")
    ap.add_argument("--ckpt",  default=".cache/checkpoints/dinov2_vitb14_reg4_pretrain.pth")
    ap.add_argument("--prefix", default="single")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    model = load_dinov2(args.ckpt)
    img_pil, img_t = load_preprocess(args.img)
    out, feats, (Hp,Wp), (H,W), cls_tok = forward_and_tokens(model, img_t)

    # Lưu heatmap + overlay
    save_heatmaps(feats, img_pil, args.prefix)

    # Lưu số liệu
    np.save(f"outputs/{args.prefix}_patch_tokens.npy", feats.numpy())  # [Hp,Wp,D]
    if cls_tok is not None: np.save(f"outputs/{args.prefix}_cls_token.npy", cls_tok.numpy())
    dump_meta_forward(out, f"outputs/{args.prefix}_forward_meta.json")

    print(f"[DONE] H={H} W={W} | patches: {Hp}x{Wp} | saved to outputs/")
    print(f" - outputs/{args.prefix}_feat_avg.jpg")
    print(f" - outputs/{args.prefix}_feat_overlay.jpg")
    print(f" - outputs/{args.prefix}_patch_tokens.npy"
          + (f"\n - outputs/{args.prefix}_cls_token.npy" if cls_tok is not None else ""))
    print(f" - outputs/{args.prefix}_forward_meta.json")

if __name__ == "__main__":
    main()
