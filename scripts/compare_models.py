"""
Side-by-side evaluation of Original vs Rebalanced EfficientNet-B0 model on image_val.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from image_branch.preprocessing.dataset import FaceForensicsDataset

def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = FaceForensicsDataset(split_input=str(PROJECT_ROOT / 'data' / 'splits' / 'image_val.csv'), split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=(device.type == 'cuda'))

    models = {
        'Original Model': PROJECT_ROOT / 'models' / 'efficientnet_b0_best.pth',
        'Rebalanced Model': PROJECT_ROOT / 'models' / 'efficientnet_b0_rebalanced.pth'
    }

    results = {}

    for name, ckpt_path in models.items():
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 6-class metrics
        acc_6class = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        # Binary metrics: Real = 0, Fake = 1..5
        bin_targets = (all_targets != 0).astype(int)
        bin_preds = (all_preds != 0).astype(int)
        bin_acc = accuracy_score(bin_targets, bin_preds)

        # Original class (Real) metrics where Real is positive label (1)
        real_targets = (all_targets == 0).astype(int)
        real_preds = (all_preds == 0).astype(int)
        real_recall = recall_score(real_targets, real_preds, pos_label=1, zero_division=0)
        real_prec = precision_score(real_targets, real_preds, pos_label=1, zero_division=0)
        real_f1 = f1_score(real_targets, real_preds, pos_label=1, zero_division=0)

        # Fake class metrics where Fake is positive label (1)
        fake_recall = recall_score(bin_targets, bin_preds, pos_label=1, zero_division=0)
        fake_prec = precision_score(bin_targets, bin_preds, pos_label=1, zero_division=0)

        cm = confusion_matrix(all_targets, all_preds)

        results[name] = {
            'acc_6class': acc_6class,
            'f1_macro': f1_macro,
            'bin_acc': bin_acc,
            'real_recall': real_recall,
            'real_prec': real_prec,
            'real_f1': real_f1,
            'fake_recall': fake_recall,
            'fake_prec': fake_prec,
            'cm': cm
        }

    print('\n' + '=' * 82)
    print('  SIDE-BY-SIDE MODEL EVALUATION (VALIDATION SET: 4,500 SAMPLES)')
    print('=' * 82)
    col1 = "Metric"
    col2 = "Original Model"
    col3 = "Rebalanced Model"
    col4 = "Delta"
    print(f'{col1:<32} | {col2:<16} | {col3:<16} | {col4:<10}')
    print('-' * 82)

    metrics_to_show = [
        ('6-Class Validation Accuracy', 'acc_6class', '%'),
        ('6-Class Macro F1', 'f1_macro', ''),
        ('Binary Validation Accuracy', 'bin_acc', '%'),
        ('Original-Class Recall (Real)', 'real_recall', '%'),
        ('Original-Class Precision (Real)', 'real_prec', '%'),
        ('Original-Class F1 (Real)', 'real_f1', ''),
        ('Fake-Class Recall (Manipulated)', 'fake_recall', '%'),
        ('Fake-Class Precision (Manipulated)', 'fake_prec', '%')
    ]

    for label, key, unit in metrics_to_show:
        orig_val = results['Original Model'][key]
        rebal_val = results['Rebalanced Model'][key]
        if unit == '%':
            delta = (rebal_val - orig_val) * 100
            print(f'{label:<32} | {orig_val*100:>15.2f}% | {rebal_val*100:>15.2f}% | {delta:>+9.2f}%')
        else:
            delta = rebal_val - orig_val
            print(f'{label:<32} | {orig_val:>16.4f} | {rebal_val:>16.4f} | {delta:>+10.4f}')

    print('\nOriginal Model 6-Class Confusion Matrix (Row=True, Col=Pred):')
    print(results['Original Model']['cm'])
    print('\nRebalanced Model 6-Class Confusion Matrix (Row=True, Col=Pred):')
    print(results['Rebalanced Model']['cm'])

if __name__ == '__main__':
    evaluate_models()
