import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pytorch_mask_rcnn as pmr

# -----------------------------
# 설정
# -----------------------------
CKPT_PATH = "./maskrcnn_coco-10.pth"
IMG_DIR = "data/chncxr/CXR_png"
GT_DIR  = "data/chncxr/masks"

SCORE_TH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dice / IoU 함수
# -----------------------------
def dice_coef(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)

def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-8)

# -----------------------------
# 모델 로드
# -----------------------------
model = pmr.maskrcnn_resnet50(pretrained=False, num_classes=2)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

tf = transforms.ToTensor()

dice_total_list = []
dice_left_list = []
dice_right_list = []

# -----------------------------
# Evaluation loop
# -----------------------------
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])

for fname in img_files:
    img_path = os.path.join(IMG_DIR, fname)

    base = os.path.splitext(fname)[0]
    gt_name = base + "_mask.png"
    gt_path = os.path.join(GT_DIR, gt_name)

    if not os.path.exists(gt_path):
        continue

    # --- load image ---
    image_pil = Image.open(img_path).convert("RGB")
    image = tf(image_pil).to(DEVICE)

    H, W = image.shape[-2:]

    # --- inference ---
    with torch.no_grad():
        output = model(image)

    if len(output["masks"]) == 0:
        continue

    masks = output["masks"].squeeze(1)   # [N,H,W]
    scores = output["scores"]
    boxes  = output["boxes"]

    keep = scores > SCORE_TH
    masks = masks[keep]
    boxes = boxes[keep]

    if len(masks) == 0:
        continue

    # -----------------------------
    # instance → 좌/우 분리
    # -----------------------------
    left_pred = np.zeros((H, W), dtype=np.bool_)
    right_pred = np.zeros((H, W), dtype=np.bool_)

    for mask, box in zip(masks, boxes):
        cx = (box[0] + box[2]) / 2
        mask_np = (mask.cpu().numpy() > 0.5)

        if cx < W / 2:
            left_pred |= mask_np
        else:
            right_pred |= mask_np

    total_pred = left_pred | right_pred

    # -----------------------------
    # GT mask 로드 및 좌/우 분리
    # -----------------------------
    gt = np.array(Image.open(gt_path).convert("L")) > 0

    gt_left = np.zeros_like(gt)
    gt_right = np.zeros_like(gt)

    mid = W // 2
    gt_left[:, :mid] = gt[:, :mid]
    gt_right[:, mid:] = gt[:, mid:]

    # -----------------------------
    # Dice 계산
    # -----------------------------
    dice_total_list.append(dice_coef(total_pred, gt))
    dice_left_list.append(dice_coef(left_pred, gt_left))
    dice_right_list.append(dice_coef(right_pred, gt_right))

# -----------------------------
# 결과 출력
# -----------------------------
def summarize(name, values):
    values = np.array(values)
    print(f"{name}: {values.mean():.4f} ± {values.std():.4f}")

print("\n===== Evaluation Result =====")
summarize("Dice_total", dice_total_list)
summarize("Dice_left ", dice_left_list)
summarize("Dice_right", dice_right_list)
