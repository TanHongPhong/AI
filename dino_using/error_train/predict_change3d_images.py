#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_change3d_images.py — Predict from explicitly provided image pairs.

Examples:
  Single pair:
    python dino_using/error_train/predict_change3d_images.py \
      --ckpt runs\\change3d\\20251031_043257\\best_change3d_dir.pth \
      --t1 path\\to\\before.jpg --t2 path\\to\\after.jpg

  Batch pairs from JSONL (each line: {"t1":"...","t2":"...","id":"optional"}):
    python dino_using/error_train/predict_change3d_images.py \
      --ckpt runs\\change3d\\20251031_043257\\best_change3d_dir.pth \
      --pairs_jsonl pairs.jsonl --save_probs --out predictions.jsonl
"""

import os
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _resize_square(img: Image.Image, size: int) -> Image.Image:
    if img.size != (size, size):
        return img.resize((size, size), Image.BILINEAR)
    return img


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


def load_model_and_meta(ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes: List[str] = ckpt.get('classes') or list((ckpt.get('class_to_id') or {}).keys())
    if not classes:
        raise RuntimeError('Classes not found in checkpoint metadata')
    args_meta: Dict = ckpt.get('args', {})
    add_delta = bool(args_meta.get('add_delta', False))
    se_before_head = bool(args_meta.get('se_before_head', False))
    dropout = float(args_meta.get('dropout', 0.0))
    in_ch = 6 if add_delta else 3

    model = Change3DNet(in_ch=in_ch, num_classes=len(classes), se_before_head=se_before_head, drop=dropout)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, classes, add_delta


def build_input_tensor(t1_path: str, t2_path: str, size: int, add_delta: bool) -> torch.Tensor:
    img1 = Image.open(t1_path).convert('RGB')
    img2 = Image.open(t2_path).convert('RGB')
    img1 = _resize_square(img1, size)
    img2 = _resize_square(img2, size)
    t1 = TF.to_tensor(img1)
    t2 = TF.to_tensor(img2)
    x = torch.stack([t1, t2], dim=1)  # [C,2,H,W]
    if add_delta:
        delta = torch.abs(t2 - t1)
        x = torch.cat([x, delta.unsqueeze(1).repeat(1,2,1,1)], dim=0)  # [C+3,2,H,W]
    return x.unsqueeze(0)  # [1,C(, +3),2,H,W]


def parse_args():
    p = argparse.ArgumentParser(description='Predict Change3D class from explicit image pairs')
    p.add_argument('--ckpt', default=r'runs\\change3d\\20251031_043257\\best_change3d_dir.pth', type=str)
    p.add_argument('--t1', type=str, help='Path to before image')
    p.add_argument('--t2', type=str, help='Path to after image')
    p.add_argument('--pairs_jsonl', type=str, default=None, help='JSONL with lines {"t1":"..","t2":"..","id":"optional"}')
    p.add_argument('--save_probs', action='store_true', help='Print/save full probability vector')
    p.add_argument('--out', type=str, default=None, help='If set with --pairs_jsonl, write predictions JSONL to this path')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes, add_delta = load_model_and_meta(args.ckpt, device)
    in_size = 128  # model trained at 128x128

    if args.pairs_jsonl:
        if not os.path.isfile(args.pairs_jsonl):
            raise FileNotFoundError(args.pairs_jsonl)
        out_f = open(args.out, 'w', encoding='utf-8') if args.out else None
        wrote = 0
        with torch.no_grad(), open(args.pairs_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t1p = obj['t1']
                t2p = obj['t2']
                pair_id: Optional[str] = obj.get('id')
                x = build_input_tensor(t1p, t2p, in_size, add_delta).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_id = int(probs.argmax())
                pred_cls = classes[pred_id]
                result = {
                    'id': pair_id,
                    't1': t1p,
                    't2': t2p,
                    'pred_class': pred_cls,
                    'pred_id': pred_id,
                    'prob': float(probs[pred_id]),
                }
                if args.save_probs:
                    result['probs'] = probs.tolist()
                if out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    print(json.dumps(result, ensure_ascii=False))
                wrote += 1
        if out_f:
            out_f.close()
            print(f"[DONE] Wrote {wrote} predictions to {args.out}")
        return

    # Single pair mode
    if not args.t1 or not args.t2:
        raise SystemExit('Provide --t1 and --t2 for single-pair prediction, or use --pairs_jsonl for batch.')
    if not os.path.isfile(args.t1):
        raise FileNotFoundError(args.t1)
    if not os.path.isfile(args.t2):
        raise FileNotFoundError(args.t2)

    with torch.no_grad():
        x = build_input_tensor(args.t1, args.t2, in_size, add_delta).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        pred_cls = classes[pred_id]

    print(f"pred_class: {pred_cls}")
    print(f"pred_id: {pred_id}")
    print(f"prob_of_pred: {probs[pred_id]:.6f}")
    if args.save_probs:
        print("probs:", json.dumps(probs.tolist(), ensure_ascii=False))


if __name__ == '__main__':
    main()


