import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import pytorch_mask_rcnn as pmr

# =============================
# 설정
# =============================
CKPT_PATH = "checkpoints_old/maskrcnn_coco-10.pth"

IMG_DIR = "data/chncxr/CXR_png"
GT_DIR  = "data/chncxr/masks"

OUT_DIR = "inference_results_final"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_IMAGES = 5        # 🔹 발표용: 5 / 내부 확인용: 10~20
SCORE_TH = 0.5

# =============================
# device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =============================
# model
# =============================
num_classes = 2  # background + lung
model = pmr.maskrcnn_resnet50(pretrained=False, num_classes=num_classes)

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print("✔ Model loaded")

# =============================
# transform
# =============================
tf = transforms.ToTensor()

# =============================
# util functions
# =============================
def load_gt_mask(gt_path):
    gt = Image.open(gt_path).convert("L")
    gt = np.array(gt)
    gt = (gt > 0).astype(np.uint8)   # 0/255 → 0/1
    return gt

def compute_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union > 0 else 0.0

def compute_dice(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    return (2 * intersection) / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 0.0

# =============================
# inference
# =============================
image_files = sorted([
    f for f in os.listdir(IMG_DIR) if f.endswith(".png")
])[:NUM_IMAGES]

ious, dices = [], []

for idx, fname in enumerate(image_files):
    img_path = os.path.join(IMG_DIR, fname)
    base = os.path.splitext(fname)[0]
    gt_path = os.path.join(GT_DIR, base + "_mask.png")

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = tf(img_pil).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    masks  = output["masks"].squeeze(1)   # [N, H, W]
    scores = output["scores"]

    keep = scores > SCORE_TH
    masks = masks[keep]

    if len(masks) == 0:
        print(f"[{fname}] no detection")
        continue

    # -----------------------------
    # 좌/우 폐 분리 (x 중심 기준)
    # -----------------------------
    masks_np = masks.cpu().numpy()
    centers = []

    for m in masks_np:
        ys, xs = np.where(m > 0.5)
        centers.append(xs.mean() if len(xs) > 0 else np.inf)

    order = np.argsort(centers)

    left_mask  = masks_np[order[0]]
    right_mask = masks_np[order[1]] if len(order) > 1 else None

    # -----------------------------
    # 전체 폐 Pred 마스크
    # -----------------------------
    pred_mask = (left_mask > 0.5).astype(np.uint8)
    if right_mask is not None:
        pred_mask = np.logical_or(pred_mask, (right_mask > 0.5)).astype(np.uint8)

    # -----------------------------
    # GT 마스크
    # -----------------------------
    gt_mask = load_gt_mask(gt_path)

    # -----------------------------
    # Metrics
    # -----------------------------
    iou  = compute_iou(gt_mask, pred_mask)
    dice = compute_dice(gt_mask, pred_mask)

    ious.append(iou)
    dices.append(dice)

    print(f"[{fname}] IoU = {iou:.4f}, Dice = {dice:.4f}")

    # =============================
    # Visualization (4-panel)
    # =============================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Original
    axes[0].imshow(img_pil, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # GT Mask (Green)
    gt_vis = np.zeros((*gt_mask.shape, 3))
    gt_vis[..., 1] = gt_mask
    axes[1].imshow(gt_vis)
    axes[1].set_title("GT Mask (Green)")
    axes[1].axis("off")

    # Pred Mask (Red)
    pred_vis = np.zeros((*pred_mask.shape, 3))
    pred_vis[..., 0] = pred_mask
    axes[2].imshow(pred_vis)
    axes[2].set_title("Pred Mask (Red)")
    axes[2].axis("off")

    # Overlay (Yellow = TP)
    overlay = np.zeros((*gt_mask.shape, 3))
    overlay[..., 0] = pred_mask        # Red
    overlay[..., 1] = gt_mask          # Green

    axes[3].imshow(img_pil, cmap="gray")
    axes[3].imshow(overlay, alpha=0.4)
    axes[3].set_title(f"Overlay\nIoU={iou:.3f}, Dice={dice:.3f}")
    axes[3].axis("off")

    out_path = os.path.join(OUT_DIR, f"result_{idx}_{fname}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"✔ saved: {out_path}")

# =============================
# Mean metrics
# =============================
if len(ious) > 0:
    print("\n📌 Performance Summary")
    print(f"Mean IoU  : {np.mean(ious):.4f}")
    print(f"Mean Dice : {np.mean(dices):.4f}")

print("🎉 inference + GT + IoU + Dice done")