#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
change3d_train_128x128.py — same as change3d_train_96x96.py but input size = 128×128
- All augmentation/output image sizes updated to 128
- Non-augment fallback resize updated to 128
"""

import os
import re
import math
import json
import argparse
import random
import datetime
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

import torchvision.transforms.functional as TF

# For plotting/saving charts
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def list_subdirs(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def split_stem_and_suffix(stem: str):
    m = re.match(r'^(.*?)[\.-_]?(t?0*1|t?0*2|1|2|01|02|a|b)$', stem, flags=re.IGNORECASE)
    if m:
        base = m.group(1)
        tag = m.group(2).lower()
        if tag in ('t1', '1', '01', 'a'):
            return base, 't1'
        if tag in ('t2', '2', '02', 'b'):
            return base, 't2'
    return stem, None


def build_pairs_from_dir(class_dir: str) -> List[Tuple[str, str]]:
    """
    Build (t1, t2) pairs for a class directory.

    Supports two layouts:
    1) Two-subfolder layout: "before/" and "after/" containing corresponding images
       with matching filenames. We pair by filename stem present in both.
    2) Flat layout: files in one folder, try to pair by suffix tags (t1/t2),
       otherwise fallback to sequential pairing.
    """
    before_dir = os.path.join(class_dir, 'before')
    after_dir  = os.path.join(class_dir, 'after')
    if os.path.isdir(before_dir) and os.path.isdir(after_dir):
        before_files = sorted([f for f in os.listdir(before_dir) if is_image_file(f)])
        after_files  = sorted([f for f in os.listdir(after_dir) if is_image_file(f)])
        before_map: Dict[str, str] = {}
        after_map: Dict[str, str] = {}
        for f in before_files:
            stem, _ = os.path.splitext(f)
            before_map[stem] = os.path.join(before_dir, f)
        for f in after_files:
            stem, _ = os.path.splitext(f)
            after_map[stem] = os.path.join(after_dir, f)
        common = sorted(set(before_map.keys()) & set(after_map.keys()))
        pairs = [(before_map[k], after_map[k]) for k in common]
        return pairs

    # Fallback: original flat directory behavior
    files = [f for f in os.listdir(class_dir) if is_image_file(f)]
    files = sorted(files)
    by_base: Dict[str, Dict[str, str]] = {}
    unmatched = []
    for f in files:
        stem, ext = os.path.splitext(f)
        base, tag = split_stem_and_suffix(stem)
        full = os.path.join(class_dir, f)
        if tag in ('t1', 't2'):
            slot = by_base.setdefault(base, {})
            if tag not in slot:
                slot[tag] = full
            else:
                unmatched.append(full)
        else:
            unmatched.append(full)
    pairs: List[Tuple[str, str]] = []
    for base, slot in by_base.items():
        if 't1' in slot and 't2' in slot:
            pairs.append((slot['t1'], slot['t2']))
        else:
            for v in slot.values():
                unmatched.append(v)
    unmatched = sorted(unmatched)
    if len(unmatched) % 2 != 0:
        unmatched = unmatched[:-1]
    for i in range(0, len(unmatched), 2):
        pairs.append((unmatched[i], unmatched[i+1]))
    return pairs


# =========================
# Augment
# =========================

@dataclass
class PairedAugmentCfg:
    out_hw: int = 128
    rotate_deg: float = 10.0
    scale_low: float = 0.95
    scale_high: float = 1.05
    shear_deg: float = 3.0
    hflip_p: float = 0.5
    blur_p: float = 0.25
    blur_radius: float = 0.6
    noise_std: float = 0.01
    misalign_px: int = 2
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
    to_linear_gamma: bool = False


def _resize(img: Image.Image, size: int) -> Image.Image:
    if img.size != (size, size):
        return img.resize((size, size), Image.BILINEAR)
    return img


def _sample_jitter_params(cfg: PairedAugmentCfg):
    b = 1.0 + random.uniform(-cfg.jitter_brightness, cfg.jitter_brightness)
    c = 1.0 + random.uniform(-cfg.jitter_contrast,   cfg.jitter_contrast)
    s = 1.0 + random.uniform(-cfg.jitter_saturation, cfg.jitter_saturation)
    h = random.uniform(-cfg.jitter_hue, cfg.jitter_hue)
    return b, c, s, h


def _apply_jitter(img: Image.Image, params):
    b, c, s, h = params
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    img = TF.adjust_hue(img, h)
    return img


def add_gaussian_noise(t: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    if std <= 0:
        return t
    return torch.clamp(t + torch.randn_like(t) * std, 0.0, 1.0)


class PairedAugment:
    def __init__(self, cfg: PairedAugmentCfg, diff_photometric_class_ids: Optional[List[int]] = None):
        self.cfg = cfg
        self.diff_photometric_ids = set(diff_photometric_class_ids or [])

    def _geom_params(self):
        angle = random.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg)
        scale = random.uniform(self.cfg.scale_low, self.cfg.scale_high)
        shear = [random.uniform(-self.cfg.shear_deg, self.cfg.shear_deg), 0.0]
        hflip = (random.random() < self.cfg.hflip_p)
        tx = random.randint(-self.cfg.misalign_px, self.cfg.misalign_px)
        ty = random.randint(-self.cfg.misalign_px, self.cfg.misalign_px)
        return angle, scale, shear, hflip, tx, ty

    def _apply_geom(self, img: Image.Image, angle, scale, shear, hflip, tx, ty) -> Image.Image:
        if hflip:
            img = TF.hflip(img)
        img = TF.affine(img, angle=angle, translate=[tx, ty], scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)
        return img

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)
        if self.cfg.to_linear_gamma:
            t = torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
        return t

    def __call__(self, img1: Image.Image, img2: Image.Image, label: int, add_delta: bool = False) -> torch.Tensor:
        size = self.cfg.out_hw
        img1 = _resize(img1, size)
        img2 = _resize(img2, size)

        # geometry (shared)
        angle, scale, shear, hflip, tx, ty = self._geom_params()
        img1 = self._apply_geom(img1, angle, scale, shear, hflip, tx, ty)
        img2 = self._apply_geom(img2, angle, scale, shear, hflip, tx, ty)

        # photometric policy
        if label in self.diff_photometric_ids:
            img1 = _apply_jitter(img1, _sample_jitter_params(self.cfg))
            img2 = _apply_jitter(img2, _sample_jitter_params(self.cfg))
        else:
            params = _sample_jitter_params(self.cfg)
            img1 = _apply_jitter(img1, params)
            img2 = _apply_jitter(img2, params)

        # blur
        if random.random() < self.cfg.blur_p:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=self.cfg.blur_radius))
        if random.random() < self.cfg.blur_p:
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=self.cfg.blur_radius))

        # to tensor + noise
        t1 = add_gaussian_noise(self._to_tensor(img1), self.cfg.noise_std)
        t2 = add_gaussian_noise(self._to_tensor(img2), self.cfg.noise_std)

        x = torch.stack([t1, t2], dim=1)  # [C, T=2, H, W]

        if add_delta:
            delta = torch.abs(t2 - t1)  # [C,H,W]
            deltaT = delta.unsqueeze(1).repeat(1, 2, 1, 1)  # [C,2,H,W]
            x = torch.cat([x, deltaT], dim=0)  # [C+3,2,H,W]
        return x


# =========================
# Dataset from directory
# =========================

class DirPairsDataset(Dataset):
    def __init__(self, split_dir: str, classes: List[str], class_to_id: Dict[str, int],
                 augmenter: Optional[PairedAugment], add_delta: bool,
                 diff_photometric_ids: List[int]):
        self.samples: List[Tuple[str, str, int]] = []  # (t1, t2, y)
        self.augmenter = augmenter
        self.add_delta = add_delta
        self.diff_photometric_ids = set(diff_photometric_ids)

        for cname in classes:
            cdir = os.path.join(split_dir, cname)
            if not os.path.isdir(cdir):
                continue
            y = class_to_id[cname]
            pairs = build_pairs_from_dir(cdir)
            for p1, p2 in pairs:
                self.samples.append((p1, p2, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under: {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, y = self.samples[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        if self.augmenter is not None:
            x = self.augmenter(img1, img2, y, add_delta=self.add_delta)
        else:
            size = 128
            img1 = img1.resize((size, size), Image.BILINEAR)
            img2 = img2.resize((size, size), Image.BILINEAR)
            t1 = TF.to_tensor(img1)
            t2 = TF.to_tensor(img2)
            x = torch.stack([t1, t2], dim=1)
            if self.add_delta:
                delta = torch.abs(t2 - t1)
                x = torch.cat([x, delta.unsqueeze(1).repeat(1,2,1,1)], dim=0)
        return x, y


# Top-level collate function (must be picklable for Windows DataLoader)
def collate_dir_pairs(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

# =========================
# Model
# =========================

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


# =========================
# Loss / Schedule / Train
# =========================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(use_focal: bool, class_weights: Optional[List[float]], label_smoothing: float):
    weight_t = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    if use_focal:
        return FocalLoss(gamma=2.0, weight=weight_t, label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(weight=weight_t, label_smoothing=label_smoothing)


def build_warmup_cosine_lr(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()
    if _HAVE_SK:
        macro_f1 = f1_score(y_true, preds, average='macro')
        per_cls_f1 = f1_score(y_true, preds, average=None, labels=list(range(num_classes)))
        cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
        report = classification_report(y_true, preds, digits=4, target_names=None)
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1
        per_cls_f1 = []
        for i in range(num_classes):
            tp = cm[i,i]
            fp = cm[:,i].sum() - tp
            fn = cm[i,:].sum() - tp
            precision = tp / max(1, tp+fp)
            recall = tp / max(1, tp+fn)
            f1 = 2*precision*recall / max(1e-12, precision+recall)
            per_cls_f1.append(f1)
        macro_f1 = float(np.mean(per_cls_f1))
        report = None
    return macro_f1, per_cls_f1, cm, report


# ---------- Reporting helpers ----------

def ensure_out_dir(base_out: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_out, ts)
    os.makedirs(out, exist_ok=True)
    return out

def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str, normalize: bool = True):
    # Preserve counts for annotation
    cm_counts = cm.copy()
    cm_plot = cm.astype(np.float32)
    if normalize:
        with np.errstate(all='ignore'):
            cm_sum = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = cm_plot / np.maximum(cm_sum, 1e-12)

    plt.figure()
    plt.imshow(cm_plot, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

    # Annotate each cell with count and percentage; add Vietnamese labels for common 2-class case
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            # percentage relative to row i
            row_sum = cm_counts[i, :].sum()
            pct = (cm_counts[i, j] / row_sum * 100.0) if row_sum > 0 else 0.0
            text_main = f"{int(cm_counts[i, j])}\n({pct:.1f}%)"

            extra = None
            if num_classes == 2:
                # Normalize class name variants for matching
                cls_norm = [c.lower().replace(' ', '').replace('&', '&') for c in class_names]
                idx_moc = None
                if 'mốc' in cls_norm:
                    idx_moc = cls_norm.index('mốc')
                variants_xoay = {'xoay&dịchchuyển', 'xoaydịchchuyển', 'xoay&dichchuyen', 'xoaydichchuyen'}
                idx_xoay = None
                for v in variants_xoay:
                    if v in cls_norm:
                        idx_xoay = cls_norm.index(v)
                        break
                if idx_moc is not None and idx_xoay is not None:
                    label_map = {
                        (idx_moc, idx_moc): 'mốc mốc',
                        (idx_moc, idx_xoay): 'mốc và xoay',
                        (idx_xoay, idx_moc): 'xoay và mốc',
                        (idx_xoay, idx_xoay): 'xoay dịch chuyển',
                    }
                    extra = label_map.get((i, j))

            display_text = f"{text_main}\n{extra}" if extra else text_main
            color = 'white' if cm_plot[i, j] > 0.5 else 'black'
            plt.text(j, i, display_text, ha='center', va='center', color=color)

    plt.tight_layout()
    plt.colorbar()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_per_class_f1(per_cls_f1: List[float], class_names: List[str], out_png: str):
    plt.figure()
    x = np.arange(len(class_names))
    plt.bar(x, per_cls_f1)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylabel('F1')
    plt.title('Per-class F1 (val)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_curve(values: List[float], ylabel: str, title: str, out_png: str):
    plt.figure()
    plt.plot(np.arange(1, len(values)+1), values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def save_val_samples(val_loader: DataLoader, out_dir: str, max_samples: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    for x, y in val_loader:
        N = min(max_samples, x.size(0))
        for i in range(N):
            t1 = x[i, 0:3, 0]
            t2 = x[i, 0:3, 1]
            delta = torch.abs(t2 - t1)
            base = os.path.join(out_dir, f"sample_{i:02d}")
            tensor_to_pil(t1).save(base + "_t1.png")
            tensor_to_pil(t2).save(base + "_t2.png")
            tensor_to_pil(delta).save(base + "_delta.png")
        break


# =========================
# Main logic (train/eval)
# =========================

@dataclass
class Args:
    data_root: str = ""
    classes: Optional[str] = None
    diff_photometric_classes: Optional[str] = None
    epochs: int = 50
    batch_size: int = 96
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    use_focal: bool = False
    label_smoothing: float = 0.05
    class_weights: Optional[str] = None
    add_delta: bool = False
    se_before_head: bool = False
    dropout: float = 0.0
    workers: int = 4
    seed: int = 42
    ckpt_out: str = "best_change3d_dir.pth"
    amp: bool = True
    eval_test: bool = False
    out_dir: str = "runs/change3d"
    dump_val_samples: int = 6


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Train Change3DNet on directory dataset (128x128) with reports")
    p.add_argument('--data_root', default=r'outputs\\sum\\dataset_new', type=str, help='root containing train/, val/, (optional) test/')
    p.add_argument('--classes', default=None, type=str, help='JSON list of class names (order defines label id)')
    p.add_argument('--diff_photometric_classes', default=None, type=str, help='JSON list of class names using different jitter between t1&t2')
    p.add_argument('--epochs', default=70, type=int)
    p.add_argument('--batch_size', default=96, type=int)
    p.add_argument('--lr', default=3e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--warmup_epochs', default=5, type=int)
    p.add_argument('--use_focal', action='store_true')
    p.add_argument('--label_smoothing', default=0.05, type=float)
    p.add_argument('--class_weights', default=None, type=str)
    p.add_argument('--add_delta', action='store_true', help='Add delta channel (disabled by default)')
    p.add_argument('--se_before_head', action='store_true')
    p.set_defaults(add_delta=False)  # Đảm bảo delta tắt mặc định
    p.add_argument('--dropout', default=0.0, type=float)
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--ckpt_out', default='best_change3d_dir.pth', type=str)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.add_argument('--eval_test', action='store_true')
    p.add_argument('--out_dir', default='runs/change3d', type=str)
    p.add_argument('--dump_val_samples', default=6, type=int)
    return Args(**vars(p.parse_args()))


def discover_classes(train_dir: str) -> List[str]:
    classes = list_subdirs(train_dir)
    if not classes:
        raise RuntimeError(f"No class folders found under {train_dir}")
    return classes


def build_loaders(args: Args):
    train_dir = os.path.join(args.data_root, 'train')
    val_dir   = os.path.join(args.data_root, 'val')
    test_dir  = os.path.join(args.data_root, 'test')

    if args.classes:
        classes = json.loads(args.classes)
    else:
        classes = discover_classes(train_dir)

    class_to_id = {c: i for i, c in enumerate(classes)}

    if args.diff_photometric_classes:
        diff_names = set(json.loads(args.diff_photometric_classes))
    else:
        diff_names = set()
    diff_ids = [class_to_id[n] for n in diff_names if n in class_to_id]

    aug_train = PairedAugment(PairedAugmentCfg(out_hw=128), diff_photometric_class_ids=diff_ids)
    aug_eval  = PairedAugment(PairedAugmentCfg(out_hw=128), diff_photometric_class_ids=diff_ids)

    train_ds = DirPairsDataset(train_dir, classes, class_to_id, augmenter=aug_train, add_delta=args.add_delta, diff_photometric_ids=diff_ids)
    val_ds   = DirPairsDataset(val_dir,   classes, class_to_id, augmenter=aug_eval,  add_delta=args.add_delta, diff_photometric_ids=diff_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_dir_pairs,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_dir_pairs,
        persistent_workers=(args.workers > 0),
    )

    test_loader = None
    if os.path.isdir(test_dir):
        try:
            test_ds = DirPairsDataset(test_dir, classes, class_to_id, augmenter=aug_eval, add_delta=args.add_delta, diff_photometric_ids=diff_ids)
            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_dir_pairs,
                persistent_workers=(args.workers > 0),
            )
        except Exception:
            test_loader = None

    return classes, class_to_id, diff_ids, train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # Use new autocast API to avoid FutureWarning
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def main():
    args = parse_args()
    set_seed(args.seed)
    # On Windows, DataLoader with multiple workers can fail pickling; force workers=0
    if os.name == 'nt' and args.workers > 0:
        print(f"[INFO] Windows detected; overriding workers from {args.workers} -> 0 to avoid spawn/pickle issues")
        args.workers = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_root = ensure_out_dir(args.out_dir)

    classes, class_to_id, diff_ids, train_loader, val_loader, test_loader = build_loaders(args)
    num_classes = len(classes)

    in_ch = 6 if args.add_delta else 3
    model = Change3DNet(in_ch=in_ch, num_classes=num_classes, se_before_head=args.se_before_head, drop=args.dropout).to(device)

    class_weights = json.loads(args.class_weights) if args.class_weights else None
    loss_fn = build_loss(args.use_focal, class_weights, args.label_smoothing)
    if getattr(loss_fn, 'weight', None) is not None:
        loss_fn.weight = loss_fn.weight.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_lr(optimizer, args.warmup_epochs, args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    history = {'epoch': [], 'train_loss': [], 'val_macro_f1': []}
    best = {'f1': -1.0, 'cm': None, 'per_f1': None, 'report': None}

    print(f"[INFO] Classes ({num_classes}): {', '.join(classes)}")
    print("Epoch |    LR    | TrainLoss | ValMacroF1 | Time(s)")
    print("------+----------+-----------+------------+--------")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        scheduler.step()

        # simple eval
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)
                logits = model(x)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        preds = logits.argmax(dim=1).numpy()
        y_true = targets.numpy()
        if _HAVE_SK:
            macro_f1 = f1_score(y_true, preds, average='macro')
            per_cls_f1 = f1_score(y_true, preds, average=None, labels=list(range(num_classes)))
            cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
            report = classification_report(y_true, preds, digits=4, target_names=None)
        else:
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            for t, p in zip(y_true, preds):
                cm[t, p] += 1
            per_cls_f1 = []
            for i in range(num_classes):
                tp = cm[i,i]
                fp = cm[:,i].sum() - tp
                fn = cm[i,:].sum() - tp
                precision = tp / max(1, tp+fp)
                recall = tp / max(1, tp+fn)
                f1 = 2*precision*recall / max(1e-12, precision+recall)
                per_cls_f1.append(f1)
            macro_f1 = float(np.mean(per_cls_f1))
            report = None

        history['epoch'].append(epoch+1)
        history['train_loss'].append(float(train_loss))
        history['val_macro_f1'].append(float(macro_f1))

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:03d}/{args.epochs} | {current_lr:.6f} |   {train_loss:.4f}  |   {macro_f1:.4f}  | {elapsed:6.1f}")
        if _HAVE_SK and report:
            print(report)

        if macro_f1 > best['f1']:
            best['f1'] = macro_f1
            best['cm'] = cm.copy()
            best['per_f1'] = per_cls_f1.copy()
            best['report'] = report if report is not None else ""

            ckpt = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'classes': classes,
                'class_to_id': class_to_id,
                'args': vars(args),
                'best_macro_f1': float(macro_f1),
            }
            ckpt_path = os.path.join(out_root, args.ckpt_out)
            torch.save(ckpt, ckpt_path)
            print(f"[BEST] Saved checkpoint to {ckpt_path} (macro-F1={macro_f1:.4f})")

            save_confusion_matrix(best['cm'], classes, os.path.join(out_root, 'val_confusion_matrix.png'), normalize=True)
            save_per_class_f1(best['per_f1'], classes, os.path.join(out_root, 'val_per_class_f1.png'))
            if best['report']:
                with open(os.path.join(out_root, 'val_classification_report.txt'), 'w', encoding='utf-8') as f:
                    f.write(best['report'])

        save_curve(history['train_loss'], 'Train Loss', 'Train Loss vs Epoch', os.path.join(out_root, 'curve_train_loss.png'))
        save_curve(history['val_macro_f1'], 'Val Macro-F1', 'Val Macro-F1 vs Epoch', os.path.join(out_root, 'curve_val_macro_f1.png'))

    print(f"Training done. Best val Macro-F1 = {best['f1']:.4f}")

    if args.dump_val_samples > 0:
        samples_dir = os.path.join(out_root, 'val_samples')
        save_val_samples(val_loader, samples_dir, max_samples=args.dump_val_samples)

    if args.eval_test and test_loader is not None:
        test_macro_f1, test_per_cls_f1, test_cm, test_report = evaluate(model, test_loader, device, num_classes)
        print("=== TEST RESULTS ===")
        print(f"Macro-F1: {test_macro_f1:.4f}")
        if _HAVE_SK and test_report:
            print(test_report)
        save_confusion_matrix(test_cm, classes, os.path.join(out_root, 'test_confusion_matrix.png'), normalize=True)
        save_per_class_f1(test_per_cls_f1, classes, os.path.join(out_root, 'test_per_class_f1.png'))
        if test_report:
            with open(os.path.join(out_root, 'test_classification_report.txt'), 'w', encoding='utf-8') as f:
                f.write(test_report)


if __name__ == '__main__':
    main()


