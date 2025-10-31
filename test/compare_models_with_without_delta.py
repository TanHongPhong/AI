#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
So sánh 2 models Change3D: có delta và không có delta
Tính toán các metrics: Accuracy, Macro F1, Per-class F1, Precision, Recall, ROC-AUC, FAR
Xuất kết quả dạng CSV, Markdown, JSON
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False
    print("[WARN] sklearn not available, some metrics may be limited")

# Import model architecture and dataset from training script
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
train_script_path = os.path.join(workspace_root, 'dino_using', 'error_train', 'change3d_train_128x128_hadam.py')
sys.path.insert(0, os.path.dirname(train_script_path))

from change3d_train_128x128_hadam import (
    Change3DNet, DirPairsDataset, build_pairs_from_dir, collate_dir_pairs
)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def find_checkpoint_with_delta_hadamard(runs_dir: str = 'runs/change3d', pattern: str = 'best_change3d_dir.pth') -> Optional[str]:
    """Tìm checkpoint mới nhất có cả delta và hadamard enabled"""
    matches = []
    for root, dirs, files in os.walk(runs_dir):
        if pattern in files:
            ckpt_path = os.path.join(root, pattern)
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                ckpt_args = ckpt.get('args', {})
                add_delta = bool(ckpt_args.get('add_delta', False))
                add_hadamard = bool(ckpt_args.get('add_hadamard', False))
                if add_delta and add_hadamard:
                    mtime = os.path.getmtime(ckpt_path)
                    matches.append((ckpt_path, mtime))
            except Exception as e:
                continue
    
    if not matches:
        return None
    
    # Trả về checkpoint mới nhất
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]


@dataclass
class EvaluationResult:
    """Kết quả đánh giá một model"""
    model_name: str
    accuracy: float
    macro_f1: float
    per_class_f1: List[float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    roc_auc: Optional[float]  # None nếu không tính được (binary/multi-class)
    far: float  # False Alarm Rate
    confusion_matrix: np.ndarray
    num_samples: int


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model từ checkpoint và trả về model + metadata"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"[LOAD] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract metadata
    ckpt_args = ckpt.get('args', {})
    classes: List[str] = ckpt.get('classes') or list((ckpt.get('class_to_id') or {}).keys())
    if not classes:
        raise RuntimeError('Classes not found in checkpoint metadata')
    
    class_to_id = {c: i for i, c in enumerate(classes)}
    add_delta = bool(ckpt_args.get('add_delta', False))
    add_hadamard = bool(ckpt_args.get('add_hadamard', False))
    se_before_head = bool(ckpt_args.get('se_before_head', False))
    dropout = float(ckpt_args.get('dropout', 0.0))
    
    # Tính số input channels
    in_ch = 3  # RGB base
    if add_delta:
        in_ch += 3
    if add_hadamard:
        in_ch += 3
    
    num_classes = len(classes)
    
    # Build model
    model = Change3DNet(in_ch=in_ch, num_classes=num_classes, 
                       se_before_head=se_before_head, drop=dropout)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    
    metadata = {
        'classes': classes,
        'class_to_id': class_to_id,
        'add_delta': add_delta,
        'add_hadamard': add_hadamard,
        'se_before_head': se_before_head,
        'dropout': dropout,
        'num_classes': num_classes,
        'in_ch': in_ch
    }
    
    print(f"[LOAD] Model loaded: in_ch={in_ch} (RGB=3, +Δ={'ON' if add_delta else 'OFF'}, +⊙={'ON' if add_hadamard else 'OFF'}), num_classes={num_classes}")
    return model, metadata


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, 
                   model_name: str, classes: List[str]) -> EvaluationResult:
    """Evaluate model và trả về kết quả chi tiết"""
    model.eval()
    all_logits, all_targets = [], []
    
    print(f"[EVAL] Evaluating {model_name}...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()
    num_classes = len(classes)
    
    # Tính các metrics
    accuracy = float(accuracy_score(y_true, preds))
    
    if _HAVE_SK:
        macro_f1 = float(f1_score(y_true, preds, average='macro'))
        per_class_f1 = f1_score(y_true, preds, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
        per_class_precision = precision_score(y_true, preds, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
        per_class_recall = recall_score(y_true, preds, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
        cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
        
        # ROC-AUC: chỉ tính được cho binary classification hoặc dùng macro/micro average cho multi-class
        try:
            if num_classes == 2:
                roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                # Multi-class: dùng one-vs-rest macro average
                y_true_onehot = np.eye(num_classes)[y_true]
                roc_auc = float(roc_auc_score(y_true_onehot, probs, average='macro', multi_class='ovr'))
        except Exception as e:
            print(f"[WARN] Could not compute ROC-AUC: {e}")
            roc_auc = None
    else:
        # Fallback calculation
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1
        
        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            per_class_precision.append(float(precision))
            per_class_recall.append(float(recall))
            per_class_f1.append(float(f1))
        
        macro_f1 = float(np.mean(per_class_f1))
        roc_auc = None
    
    # False Alarm Rate (FAR): tỷ lệ dự đoán sai (FP / (FP + TN))
    # Đối với multi-class: tính tổng FAR cho tất cả các class
    total_fp = cm.sum() - np.trace(cm)  # Tổng các predictions sai
    total_negatives = len(y_true) * (num_classes - 1)  # Số lượng negatives (không phải class đúng)
    far = total_fp / max(1, total_negatives) if total_negatives > 0 else 0.0
    
    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        roc_auc=roc_auc,
        far=float(far),
        confusion_matrix=cm,
        num_samples=len(y_true)
    )


def save_results_csv(results: List[EvaluationResult], classes: List[str], output_path: str):
    """Lưu kết quả dạng CSV"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header - hỗ trợ nhiều models
        model_names = [r.model_name for r in results]
        header = ['Metric'] + model_names
        writer.writerow(header)
        
        # Overall metrics
        writer.writerow(['Accuracy'] + [f"{r.accuracy:.4f}" for r in results])
        writer.writerow(['Macro F1'] + [f"{r.macro_f1:.4f}" for r in results])
        if all(r.roc_auc is not None for r in results):
            writer.writerow(['ROC-AUC'] + [f"{r.roc_auc:.4f}" for r in results])
        writer.writerow(['False Alarm Rate'] + [f"{r.far:.4f}" for r in results])
        writer.writerow(['Num Samples'] + [r.num_samples for r in results])
        writer.writerow([])
        
        # Per-class F1
        writer.writerow(['Per-Class F1 Score'])
        for i, cls in enumerate(classes):
            writer.writerow([f"F1_{cls}"] + [f"{r.per_class_f1[i]:.4f}" for r in results])
        writer.writerow([])
        
        # Per-class Precision
        writer.writerow(['Per-Class Precision'])
        for i, cls in enumerate(classes):
            writer.writerow([f"Precision_{cls}"] + [f"{r.per_class_precision[i]:.4f}" for r in results])
        writer.writerow([])
        
        # Per-class Recall
        writer.writerow(['Per-Class Recall'])
        for i, cls in enumerate(classes):
            writer.writerow([f"Recall_{cls}"] + [f"{r.per_class_recall[i]:.4f}" for r in results])


def save_results_markdown(results: List[EvaluationResult], classes: List[str], output_path: str):
    """Lưu kết quả dạng Markdown table"""
    with open(output_path, 'w', encoding='utf-8') as f:
        model_names = [r.model_name for r in results]
        f.write("# So sánh Models Change3D\n\n")
        
        # Overall metrics table
        f.write("## Overall Metrics\n\n")
        # Header
        header = "| Metric |"
        separator = "|--------|"
        for name in model_names:
            header += f" {name} |"
            separator += "-----------|"
        f.write(header + "\n")
        f.write(separator + "\n")
        
        # Rows
        f.write(f"| **Accuracy** | {' | '.join([f'{r.accuracy:.4f}' for r in results])} |\n")
        f.write(f"| **Macro F1** | {' | '.join([f'{r.macro_f1:.4f}' for r in results])} |\n")
        if all(r.roc_auc is not None for r in results):
            f.write(f"| **ROC-AUC** | {' | '.join([f'{r.roc_auc:.4f}' for r in results])} |\n")
        f.write(f"| **False Alarm Rate (FAR)** | {' | '.join([f'{r.far:.4f}' for r in results])} |\n")
        f.write(f"| **Số lượng mẫu** | {' | '.join([str(r.num_samples) for r in results])} |\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        f.write("### F1 Score\n\n")
        header = "| Class |"
        separator = "|-------|"
        for name in model_names:
            header += f" {name} |"
            separator += "-----------|"
        f.write(header + "\n")
        f.write(separator + "\n")
        for i, cls in enumerate(classes):
            f.write(f"| {cls} | {' | '.join([f'{r.per_class_f1[i]:.4f}' for r in results])} |\n")
        f.write("\n")
        
        f.write("### Precision\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for i, cls in enumerate(classes):
            f.write(f"| {cls} | {' | '.join([f'{r.per_class_precision[i]:.4f}' for r in results])} |\n")
        f.write("\n")
        
        f.write("### Recall\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for i, cls in enumerate(classes):
            f.write(f"| {cls} | {' | '.join([f'{r.per_class_recall[i]:.4f}' for r in results])} |\n")


def save_results_json(results: List[EvaluationResult], classes: List[str], output_path: str):
    """Lưu kết quả dạng JSON"""
    models_dict = {}
    for result in results:
        key = result.model_name.lower().replace(' ', '_').replace('&', 'and').replace('+', '_')
        models_dict[key] = {
            'name': result.model_name,
            'accuracy': result.accuracy,
            'macro_f1': result.macro_f1,
            'roc_auc': result.roc_auc,
            'far': result.far,
            'num_samples': result.num_samples,
            'per_class': {
                cls: {
                    'f1': result.per_class_f1[i],
                    'precision': result.per_class_precision[i],
                    'recall': result.per_class_recall[i]
                }
                for i, cls in enumerate(classes)
            },
            'confusion_matrix': result.confusion_matrix.tolist()
        }
    
    data = {
        'models': models_dict,
        'classes': classes
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='So sánh models Change3D: không delta, có delta, có delta+hadamard')
    parser.add_argument('--ckpt_no_delta', 
                       default=r'runs\change3d\20251031_224419_nodenta\best_change3d_dir.pth',
                       type=str, help='Checkpoint không có delta')
    parser.add_argument('--ckpt_with_delta',
                       default=r'runs\change3d\20251031_230655\best_change3d_dir.pth',
                       type=str, help='Checkpoint có delta')
    parser.add_argument('--ckpt_delta_hadamard',
                       default=None,
                       type=str, help='Checkpoint có delta + hadamard (optional, nếu None sẽ skip)')
    parser.add_argument('--test_dir',
                       default=r'outputs\sum\dataset_new\val',
                       type=str, help='Thư mục test set (chứa các class subfolders)')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--output_dir', default='test', type=str, help='Thư mục lưu kết quả')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    model_no_delta, meta_no_delta = load_model_from_checkpoint(args.ckpt_no_delta, device)
    model_with_delta, meta_with_delta = load_model_from_checkpoint(args.ckpt_with_delta, device)
    
    # Kiểm tra classes giống nhau
    if meta_no_delta['classes'] != meta_with_delta['classes']:
        print("[WARN] Classes khác nhau giữa 2 models!")
        print(f"  No Delta: {meta_no_delta['classes']}")
        print(f"  With Delta: {meta_with_delta['classes']}")
    
    classes = meta_no_delta['classes']
    class_to_id = meta_no_delta['class_to_id']
    
    # Build test dataset (không augmentation)
    print(f"[DATA] Loading test set from: {args.test_dir}")
    test_dataset = DirPairsDataset(
        split_dir=args.test_dir,
        classes=classes,
        class_to_id=class_to_id,
        augmenter=None,  # Không augmentation cho test
        add_delta=False,  # Không quan trọng vì không augmentation
        diff_photometric_ids=[]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_dir_pairs
    )
    
    print(f"[DATA] Test set: {len(test_dataset)} samples")
    
    # Evaluate các models
    # Tạo dataset riêng cho từng model (vì add_delta và add_hadamard khác nhau)
    def build_dataset_for_model(add_delta: bool, add_hadamard: bool):
        """Build dataset với add_delta và add_hadamard flags đúng"""
        class TestDataset(Dataset):
            def __init__(self, base_dataset: DirPairsDataset, add_delta: bool, add_hadamard: bool):
                self.samples = base_dataset.samples
                self.add_delta = add_delta
                self.add_hadamard = add_hadamard
                self.size = 128
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                p1, p2, y = self.samples[idx]
                img1 = Image.open(p1).convert('RGB')
                img2 = Image.open(p2).convert('RGB')
                img1 = img1.resize((self.size, self.size), Image.BILINEAR)
                img2 = img2.resize((self.size, self.size), Image.BILINEAR)
                t1 = TF.to_tensor(img1)
                t2 = TF.to_tensor(img2)
                x = torch.stack([t1, t2], dim=1)  # [C, T=2, H, W]
                
                if self.add_delta:
                    delta = torch.abs(t2 - t1)  # [C,H,W]
                    deltaT = delta.unsqueeze(1).repeat(1, 2, 1, 1)  # [C,2,H,W]
                    x = torch.cat([x, deltaT], dim=0)  # [C+3,2,H,W]
                
                if self.add_hadamard:
                    hadam = t2 * t1  # [C,H,W]
                    hadamT = hadam.unsqueeze(1).repeat(1, 2, 1, 1)  # [C,2,H,W]
                    x = torch.cat([x, hadamT], dim=0)  # [+3,2,H,W]
                
                return x, y
        
        return TestDataset(test_dataset, add_delta, add_hadamard)
    
    results = []
    
    # Evaluate model không delta
    test_dataset_no_delta = build_dataset_for_model(add_delta=False, add_hadamard=False)
    test_loader_no_delta = DataLoader(
        test_dataset_no_delta,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_dir_pairs
    )
    
    result_no_delta = evaluate_model(
        model_no_delta, test_loader_no_delta, device,
        "No_Delta", classes
    )
    results.append(result_no_delta)
    
    # Evaluate model có delta
    test_dataset_with_delta = build_dataset_for_model(add_delta=True, add_hadamard=False)
    test_loader_with_delta = DataLoader(
        test_dataset_with_delta,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_dir_pairs
    )
    
    result_with_delta = evaluate_model(
        model_with_delta, test_loader_with_delta, device,
        "With_Delta", classes
    )
    results.append(result_with_delta)
    
    # Evaluate model có delta + hadamard (nếu có checkpoint)
    ckpt_delta_hadamard = args.ckpt_delta_hadamard
    if not ckpt_delta_hadamard:
        # Tự động tìm checkpoint delta+hadamard mới nhất
        print("[INFO] Tự động tìm checkpoint Delta+Hadamard mới nhất...")
        ckpt_delta_hadamard = find_checkpoint_with_delta_hadamard()
        if ckpt_delta_hadamard:
            print(f"[INFO] Tìm thấy checkpoint Delta+Hadamard: {ckpt_delta_hadamard}")
    
    if ckpt_delta_hadamard and os.path.isfile(ckpt_delta_hadamard):
        print(f"\n[LOAD] Loading Delta+Hadamard checkpoint...")
        model_delta_hadamard, meta_delta_hadamard = load_model_from_checkpoint(ckpt_delta_hadamard, device)
        
        test_dataset_delta_hadamard = build_dataset_for_model(add_delta=True, add_hadamard=True)
        test_loader_delta_hadamard = DataLoader(
            test_dataset_delta_hadamard,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_dir_pairs
        )
        
        result_delta_hadamard = evaluate_model(
            model_delta_hadamard, test_loader_delta_hadamard, device,
            "Delta+Hadamard", classes
        )
        results.append(result_delta_hadamard)
    else:
        if args.ckpt_delta_hadamard:
            print(f"[WARN] Checkpoint Delta+Hadamard không tồn tại: {args.ckpt_delta_hadamard}, sẽ bỏ qua.")
        else:
            print("[INFO] Không tìm thấy checkpoint Delta+Hadamard, chỉ so sánh 2 models.")
    
    # Save results
    print("\n[SAVE] Saving results...")
    save_results_csv(results, classes, os.path.join(args.output_dir, 'comparison.csv'))
    save_results_markdown(results, classes, os.path.join(args.output_dir, 'comparison.md'))
    save_results_json(results, classes, os.path.join(args.output_dir, 'comparison.json'))
    
    print("\n[RESULTS]")
    for result in results:
        print(f"{result.model_name}: Accuracy={result.accuracy:.4f}, Macro F1={result.macro_f1:.4f}, FAR={result.far:.4f}")
    print(f"\n[OUTPUT] Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

