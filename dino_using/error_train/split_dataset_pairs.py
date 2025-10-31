#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import shutil
from typing import List, Dict, Tuple

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def list_classes(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def build_pairs_for_class(class_dir: str) -> List[Tuple[str, str]]:
    """
    Expect layout:
      class_dir/
        before/*.ext
        after/*.ext
    Pair by filename stem present in both.
    Returns list of (before_path, after_path)
    """
    before_dir = os.path.join(class_dir, 'before')
    after_dir  = os.path.join(class_dir, 'after')
    if not (os.path.isdir(before_dir) and os.path.isdir(after_dir)):
        return []
    before_files = [f for f in os.listdir(before_dir) if is_image_file(f)]
    after_files  = [f for f in os.listdir(after_dir) if is_image_file(f)]
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clear_dir(path: str):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass


def split_and_write(class_name: str, pairs: List[Tuple[str, str]], out_root: str, val_ratio: float, move: bool = False):
    random.shuffle(pairs)
    n_total = len(pairs)
    n_val = int(round(n_total * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Prepare dirs
    t_before = os.path.join(out_root, 'train', class_name, 'before')
    t_after  = os.path.join(out_root, 'train', class_name, 'after')
    v_before = os.path.join(out_root, 'val',   class_name, 'before')
    v_after  = os.path.join(out_root, 'val',   class_name, 'after')
    for d in [t_before, t_after, v_before, v_after]:
        ensure_dir(d)

    op = shutil.move if move else shutil.copy2

    def _copy_pair(dst_before_dir: str, dst_after_dir: str, pair: Tuple[str, str]):
        b, a = pair
        b_name = os.path.basename(b)
        a_name = os.path.basename(a)
        op(b, os.path.join(dst_before_dir, b_name))
        op(a, os.path.join(dst_after_dir,  a_name))

    for p in train_pairs:
        _copy_pair(t_before, t_after, p)
    for p in val_pairs:
        _copy_pair(v_before, v_after, p)

    print(f"[CLASS {class_name}] total={n_total} -> train={len(train_pairs)}, val={len(val_pairs)}")


def main():
    parser = argparse.ArgumentParser(description='Split paired dataset into train/val by stems (before/after).')
    parser.add_argument('--data_root', default=r'outputs\\dataset', type=str, help='Source dataset root containing per-class folders with before/after')
    parser.add_argument('--out_root',  default=r'outputs\\sum\\dataset_new', type=str, help='Destination root to write split dataset train/ and val/')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='Validation ratio (0-1)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--move', action='store_true', help='Move files instead of copy')
    parser.add_argument('--clear', action='store_true', help='Clear existing train/ and val/ under out_root before splitting')
    args = parser.parse_args()

    random.seed(args.seed)

    # Ensure destination root exists
    os.makedirs(args.out_root, exist_ok=True)

    # Classes are direct subfolders under data_root (each has before/after)
    classes = list_classes(args.data_root)
    if args.clear:
        clear_dir(os.path.join(args.out_root, 'train'))
        clear_dir(os.path.join(args.out_root, 'val'))

    grand_total, total_train, total_val = 0, 0, 0
    for cname in classes:
        cdir = os.path.join(args.data_root, cname)
        pairs = build_pairs_for_class(cdir)
        if not pairs:
            print(f"[WARN] Skip '{cname}': not found before/after or no paired stems")
            continue
        grand_total += len(pairs)
        split_and_write(cname, pairs, args.out_root, args.val_ratio, move=args.move)
        n_val = int(round(len(pairs) * args.val_ratio))
        total_val += n_val
        total_train += (len(pairs) - n_val)

    print(f"[DONE] total_pairs={grand_total} | train={total_train} | val={total_val}")
    print(f"[OUT]  wrote to: {args.out_root}")


if __name__ == '__main__':
    main()


