"""
DeepGuard X — Phase 10 Explainability Engine
Implements:
1. Grad-CAM++ for CNN models (Xception)
2. Attention Rollout / Token Attribution for Vision Transformers (ViT-B/16)
3. Error Level Analysis (ELA) as forensic supporting evidence
Generates side-by-side diagnostic panels:
[Original RGB] | [Grad-CAM++ / Attention Attribution] | [Forensic ELA Map]
for:
- Correctly classified REAL (True Real)
- Correctly classified FAKE (True Fake)
- Incorrectly classified FAKE as REAL (False Negative - DeepGuard X focus)
- Incorrectly classified REAL as FAKE (False Positive)
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = Path("E:/ADL_PROJECT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe patch for Windows App Control policy blocking optree._C in torch._dynamo
import types
mock_pytree = types.ModuleType('torch._dynamo.polyfills.pytree')
mock_pytree.__all__ = []
sys.modules['torch._dynamo.polyfills.pytree'] = mock_pytree

import timm
import torchvision.models as tvm
from image_branch.experiments.train_controlled import create_model, TARGET_SIZE, IMAGENET_MEAN, IMAGENET_STD

REPORTS_DIR = PROJECT_ROOT / "image_branch" / "experiments" / "reports"
EXPLAIN_DIR = REPORTS_DIR / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Forensic ELA (Error Level Analysis)
# -------------------------------------------------------------
def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Computes Error Level Analysis by compressing image to JPEG and measuring pixel diff.
    """
    temp_jpeg = EXPLAIN_DIR / "temp_ela.jpg"
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_rgb.save(temp_jpeg, "JPEG", quality=quality)
        with Image.open(temp_jpeg) as resaved_img:
            diff = ImageChops.difference(img_rgb, resaved_img)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale_factor = min(255.0 / max_diff, scale)
            enhancer = ImageEnhance.Brightness(diff)
            ela_img = enhancer.enhance(scale_factor)
            ela_arr = np.array(ela_img)
            
    if temp_jpeg.exists():
        temp_jpeg.unlink()
    return ela_arr

# -------------------------------------------------------------
# 2. Grad-CAM++ for Xception
# -------------------------------------------------------------
class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_fwd = target_layer.register_forward_hook(self._save_activations)
        self.hook_bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        grads = self.gradients[0]  # (C, H, W)
        acts = self.activations[0]  # (C, H, W)
        
        # Grad-CAM++ weight formula
        grads_pow2 = grads.pow(2)
        grads_pow3 = grads.pow(3)
        sum_acts = acts.sum(dim=(1, 2), keepdim=True)
        
        alpha_denom = 2.0 * grads_pow2 + sum_acts * grads_pow3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = grads_pow2 / alpha_denom
        
        positive_grads = F.relu(score.exp() * grads)
        weights = (alpha * positive_grads).sum(dim=(1, 2), keepdim=True)
        
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_np = cam.detach().cpu().numpy()
        return cam_np

    def remove_hooks(self):
        self.hook_fwd.remove()
        self.hook_bwd.remove()

# -------------------------------------------------------------
# 3. ViT Attention Attribution
# -------------------------------------------------------------
def get_vit_attention_map(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Extracts class token attention from ViT-B/16 across 14x14 spatial patches.
    """
    model.eval()
    with torch.no_grad():
        x = model._process_input(input_tensor)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder.dropout(x + model.encoder.pos_embedding)
        
        # Forward through layers up to last block
        for layer in model.encoder.layers[:-1]:
            x = layer(x)
            
        last_block = model.encoder.layers[-1]
        x_norm = last_block.ln_1(x)
        # Compute self-attention query-key weights
        attn_module = last_block.self_attention
        B, N, C = x_norm.shape
        qkv = attn_module.in_proj_weight @ x_norm.squeeze(0).T
        if attn_module.in_proj_bias is not None:
            qkv = qkv + attn_module.in_proj_bias.unsqueeze(-1)
        qkv = qkv.T.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, heads, N, N)
        cls_attn = attn_weights[0, :, 0, 1:].mean(dim=0) # average across heads, shape (196,)
        cls_map = cls_attn.reshape(14, 14).cpu().numpy()
        cls_map = (cls_map - cls_map.min()) / (cls_map.max() - cls_map.min() + 1e-8)
        return cls_map

# -------------------------------------------------------------
# 4. Composite Panel Generator
# -------------------------------------------------------------
def create_explanation_panel(
    image_path: str,
    orig_rgb: np.ndarray,
    heatmap: np.ndarray,
    ela_rgb: np.ndarray,
    title_text: str,
    meta_text: str,
    output_path: Path,
    method_name: str = "Grad-CAM++"
):
    h, w, _ = orig_rgb.shape
    # Resize heatmap to original image resolution
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat_resized), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap onto original RGB
    overlay = np.uint8(0.55 * orig_rgb + 0.45 * heat_color)
    
    # Resize ELA if needed
    ela_resized = cv2.resize(ela_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Image (RGB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"{method_name} Attribution Overlay", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(ela_resized)
    axes[2].set_title("Forensic Error Level Analysis (ELA)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    fig.suptitle(f"{title_text}\n{meta_text}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Explanation panel saved: {output_path}")

def main():
    print("=" * 80)
    print("DEEPGUARD X — PHASE 10: VERIFIED EXPLAINABILITY & FORENSICS")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Xception (Top CNN Detector)
    xception_ckpt = PROJECT_ROOT / "image_branch" / "experiments" / "xception" / "best_model.pth"
    model_xc = create_model("xception").to(device)
    model_xc.load_state_dict(torch.load(xception_ckpt, map_location=device)["model_state_dict"])
    model_xc.eval()
    if hasattr(model_xc, "act4"):
        model_xc.act4.inplace = False
    
    # Target layer for Grad-CAM++ in legacy_xception: conv4.pointwise (2048-dim feature map)
    target_layer_xc = model_xc.conv4.pointwise
    gradcam_pp = GradCAMPlusPlus(model_xc, target_layer_xc)
    
    # 2. Load ViT-B/16
    vit_ckpt = PROJECT_ROOT / "image_branch" / "experiments" / "vit" / "best_model.pth"
    model_vit = create_model("vit").to(device)
    model_vit.load_state_dict(torch.load(vit_ckpt, map_location=device)["model_state_dict"])
    model_vit.eval()
    
    # Load Test Predictions for Xception
    df_preds = pd.read_csv(PROJECT_ROOT / "image_branch" / "experiments" / "xception" / "test_predictions.csv")
    
    # Select Case Studies
    # Case 1: True Real (Authentic correctly classified)
    case_tr = df_preds[(df_preds["true_label"] == 0) & (df_preds["predicted_label"] == 0)].sort_values("real_probability", ascending=False).iloc[0]
    
    # Case 2: True Fake (Deepfake correctly classified)
    case_tf = df_preds[(df_preds["true_label"] == 1) & (df_preds["predicted_label"] == 1)].sort_values("fake_probability", ascending=False).iloc[0]
    
    # Case 3: False Negative (Deepfake/AI incorrectly predicted as Real) — KEY CRITICAL CASE
    case_fn = df_preds[(df_preds["true_label"] == 1) & (df_preds["predicted_label"] == 0)].sort_values("real_probability", ascending=False).iloc[0]
    
    # Case 4: False Positive (Real incorrectly predicted as Fake)
    case_fp = df_preds[(df_preds["true_label"] == 0) & (df_preds["predicted_label"] == 1)].sort_values("fake_probability", ascending=False).iloc[0]
    
    cases = [
        ("true_real", case_tr, "TRUE REAL (Correct Authentic)"),
        ("true_fake", case_tf, "TRUE FAKE (Correct Deepfake Detected)"),
        ("false_negative", case_fn, "FALSE NEGATIVE (Deepfake Predicted as REAL — Failure Case)"),
        ("false_positive", case_fp, "FALSE POSITIVE (Authentic Predicted as FAKE)")
    ]
    
    explain_records = []
    
    for case_id, row, label_desc in cases:
        fp = row["filepath"]
        true_lbl = "Real (0)" if row["true_label"] == 0 else "Fake (1)"
        pred_lbl = "Real (0)" if row["predicted_label"] == 0 else "Fake (1)"
        p_fake = row["fake_probability"]
        p_real = row["real_probability"]
        
        print(f"\nProcessing Explainability for Case: {case_id.upper()} ({fp})")
        
        # Load and preprocess image
        orig_rgb = np.array(Image.open(fp).convert("RGB"), dtype=np.uint8)
        ela_map = compute_ela(fp)
        
        # Prepare normalized tensor for model
        eval_tf = cv2.resize(orig_rgb, (TARGET_SIZE, TARGET_SIZE))
        tensor = (eval_tf.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.tensor(tensor.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        
        # 1. Generate Grad-CAM++ for Xception
        # Explain the predicted class
        pred_class = int(row["predicted_label"])
        cam_xc = gradcam_pp.generate(tensor, class_idx=pred_class)
        
        meta_str = f"True: {true_lbl} | Pred: {pred_lbl} | P(Real): {p_real:.3f} | P(Fake): {p_fake:.3f} | Model: Xception"
        out_xc = EXPLAIN_DIR / f"{case_id}_xception_gradcampp.png"
        create_explanation_panel(
            fp, orig_rgb, cam_xc, ela_map,
            f"DeepGuard X Explainability — {label_desc}",
            meta_str, out_xc, method_name="Grad-CAM++ (Xception)"
        )
        
        # 2. Generate ViT Attention Map
        attn_vit = get_vit_attention_map(model_vit, tensor)
        out_vit = EXPLAIN_DIR / f"{case_id}_vit_attention.png"
        meta_str_vit = f"True: {true_lbl} | Pred: {pred_lbl} | Model: ViT-B/16 Self-Attention Attribution"
        create_explanation_panel(
            fp, orig_rgb, attn_vit, ela_map,
            f"DeepGuard X ViT Attribution — {label_desc}",
            meta_str_vit, out_vit, method_name="Attention Attribution (ViT)"
        )
        
        explain_records.append({
            "case_id": case_id,
            "description": label_desc,
            "filepath": fp,
            "true_label": int(row["true_label"]),
            "predicted_label": int(row["predicted_label"]),
            "real_probability": float(p_real),
            "fake_probability": float(p_fake),
            "gradcampp_image": str(out_xc),
            "vit_attention_image": str(out_vit)
        })
        
    gradcam_pp.remove_hooks()
    
    # Save explainability report
    with open(EXPLAIN_DIR / "explainability_summary.json", "w") as f:
        json.dump(explain_records, f, indent=2)
    print(f"\nPhase 10 Explainability complete! Summary saved to {EXPLAIN_DIR / 'explainability_summary.json'}")

if __name__ == "__main__":
    main()
