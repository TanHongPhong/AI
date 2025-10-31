#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_change3d_dir.py — Inference for Change3DNet on directory-based pairs (128x128)

Usage example (defaults point to your checkpoint and val split):
  python dino_using/error_train/predict_change3d_dir.py \
    --ckpt runs\\change3d\\20251031_043257\\best_change3d_dir.pth \
    --data_dir outputs\\sum\\dataset_new\\val

It will write predictions to an auto-created folder under runs/change3d/<timestamp>/predict/:
  - predictions.csv
  - predictions.jsonl
"""

import os
import re
import json
import argparse
import datetime
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


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
        return [(before_map[k], after_map[k]) for k in common]

    files = [f for f in os.listdir(class_dir) if is_image_file(f)]
    files = sorted(files)
    by_base: Dict[str, Dict[str, str]] = {}
    unmatched: List[str] = []
    for f in files:
        stem, _ = os.path.splitext(f)
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


def ensure_out_dir(base_out: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_out, ts, 'predict')
    os.makedirs(out, exist_ok=True)
    return out


class PairDirDataset(Dataset):
    def __init__(self, split_dir: str, classes: List[str], class_to_id: Dict[str, int], out_hw: int, add_delta: bool):
        self.samples: List[Tuple[str, str, int]] = []
        self.out_hw = out_hw
        self.add_delta = add_delta

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
        size = self.out_hw
        if img1.size != (size, size):
            img1 = img1.resize((size, size), Image.BILINEAR)
        if img2.size != (size, size):
            img2 = img2.resize((size, size), Image.BILINEAR)
        t1 = TF.to_tensor(img1)
        t2 = TF.to_tensor(img2)
        x = torch.stack([t1, t2], dim=1)  # [C, 2, H, W]
        if self.add_delta:
            delta = torch.abs(t2 - t1)
            x = torch.cat([x, delta.unsqueeze(1).repeat(1, 2, 1, 1)], dim=0)
        return x, y, p1, p2


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


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def parse_args():
    p = argparse.ArgumentParser(description="Predict with Change3DNet on directory pairs (128x128)")
    p.add_argument('--ckpt', default=r'runs\\change3d\\20251031_043257\\best_change3d_dir.pth', type=str)
    p.add_argument('--data_dir', default=r'outputs\\sum\\dataset_new\\val', type=str, help='Folder containing class subfolders to run inference on')
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--workers', default=0, type=int)
    p.add_argument('--out_root', default='runs/change3d', type=str)
    p.add_argument('--save_probs', action='store_true', help='Include full probability vector in outputs')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt_args = ckpt.get('args', {})
    classes: List[str] = ckpt.get('classes') or list((ckpt.get('class_to_id') or {}).keys())
    if not classes:
        raise RuntimeError('Classes not found in checkpoint metadata')
    class_to_id = {c: i for i, c in enumerate(classes)}

    add_delta = bool(ckpt_args.get('add_delta', False))
    se_before_head = bool(ckpt_args.get('se_before_head', False))
    dropout = float(ckpt_args.get('dropout', 0.0))
    in_ch = 6 if add_delta else 3

    model = Change3DNet(in_ch=in_ch, num_classes=len(classes), se_before_head=se_before_head, drop=dropout)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    # Determine input size (defaults to 128)
    out_hw = 128

    # Build dataset and loader
    def list_subdirs(path: str) -> List[str]:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    target_classes = list_subdirs(args.data_dir)
    # Keep order consistent with training classes; filter missing ones
    target_classes = [c for c in classes if c in set(target_classes)]
    if not target_classes:
        raise RuntimeError(f"No class folders under {args.data_dir} matching training classes {classes}")

    ds = PairDirDataset(args.data_dir, target_classes, class_to_id, out_hw=out_hw, add_delta=add_delta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    out_dir = ensure_out_dir(args.out_root)
    csv_path = os.path.join(out_dir, 'predictions.csv')
    jsonl_path = os.path.join(out_dir, 'predictions.jsonl')

    # Write headers
    with open(csv_path, 'w', encoding='utf-8') as fcsv:
        cols = ['t1_path', 't2_path', 'true_class', 'pred_class', 'pred_id', 'prob']
        if args.save_probs:
            cols.append('probs_json')
        fcsv.write(','.join(cols) + '\n')

    num_correct = 0
    num_total = 0
    with torch.no_grad(), open(csv_path, 'a', encoding='utf-8') as fcsv, open(jsonl_path, 'w', encoding='utf-8') as fjsonl:
        for x, y, p1s, p2s in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            y_np = y.numpy()
            for i in range(x.size(0)):
                pred_id = int(preds[i])
                pred_cls = classes[pred_id]
                true_cls = classes[int(y_np[i])] if 0 <= int(y_np[i]) < len(classes) else ''
                prob = float(probs[i, pred_id])
                row = [p1s[i], p2s[i], true_cls, pred_cls, str(pred_id), f"{prob:.6f}"]
                obj = {
                    't1_path': p1s[i],
                    't2_path': p2s[i],
                    'true_class': true_cls,
                    'pred_class': pred_cls,
                    'pred_id': pred_id,
                    'prob': prob,
                }
                if args.save_probs:
                    row.append(json.dumps(probs[i].tolist(), ensure_ascii=False))
                    obj['probs'] = probs[i].tolist()
                fcsv.write(','.join(row) + '\n')
                fjsonl.write(json.dumps(obj, ensure_ascii=False) + '\n')

            num_correct += int((preds == y_np).sum())
            num_total += int(x.size(0))

    acc = num_correct / max(1, num_total)
    meta = {
        'ckpt': args.ckpt,
        'data_dir': args.data_dir,
        'num_samples': num_total,
        'accuracy_if_labels_present': acc,
        'classes': classes,
        'add_delta': add_delta,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Predictions saved to:\n  {csv_path}\n  {jsonl_path}\n  {os.path.join(out_dir, 'metadata.json')}\nAcc (if labels present): {acc:.4f}")


if __name__ == '__main__':
    main()


