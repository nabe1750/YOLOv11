import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO

def create_increment_dir(parent="./output", base="vector_results"):
    os.makedirs(parent, exist_ok=True)
    n = 0
    while True:
        name = base if n == 0 else f"{base}{n}"
        path = os.path.join(parent, name)
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        n += 1


def get_short_side_midpoints_torch(xyxyxyxy):
    if isinstance(xyxyxyxy, torch.Tensor):
        xyxyxyxy = xyxyxyxy.detach().cpu()

    pts = xyxyxyxy.reshape(4, 2).float()
    edges = torch.norm(torch.roll(pts, shifts=-1, dims=0) - pts, dim=1)
    short_edges_idx = torch.argsort(edges)[:2]

    midpoints = []
    for idx in short_edges_idx:
        p1 = pts[idx]
        p2 = pts[(idx + 1) % 4]
        midpoints.append((p1 + p2) / 2)

    return torch.stack(midpoints)


def select_non_masked_midpoint(midpoints, masks, orig_h, orig_w):
    if masks is None:
        return midpoints

    mask_tensor = masks.data.float().cpu()
    mask_resized = F.interpolate(
        mask_tensor.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="nearest"
    ).squeeze(1)

    selected = []
    for mp in midpoints:
        x, y = int(mp[0]), int(mp[1])
        inside = False

        for m in mask_resized:
            if 0 <= y < orig_h and 0 <= x < orig_w:
                if m[y, x] > 0.5:
                    inside = True
                    break
        if not inside:
            selected.append(mp)

    return selected


# ============================================================
#                   推論 + 描画 + 保存
# ============================================================

input_folder = "../../../data/shiba_data/nanox/seijo/images"
output_dir = create_increment_dir(parent="../output", base="vector_results")

obb_model = YOLO("../runs/obb/train9/weights/best.pt")
seg_model = YOLO("../runs/segment/train27/weights/best.pt")

for obb_result, seg_result in zip(
    obb_model.predict(source=input_folder, save=True, stream=True),
    seg_model.predict(source=input_folder, save=True, stream=True),
):
    img = obb_result.orig_img.copy()
    h, w = img.shape[:2]

    # ============================
    # ① セグメンテーションマスクを「青」で重ねる
    # ============================
    if seg_result.masks is not None:
        masks = seg_result.masks.data.cpu().numpy()
        masks_resized = np.zeros((len(masks), h, w), dtype=np.uint8)

        for i, m in enumerate(masks):
            masks_resized[i] = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        blue = np.array([255, 0, 0], dtype=np.uint8)
        for m in masks_resized:
            color_img = np.zeros_like(img)
            color_img[m == 1] = blue
            img = cv2.addWeighted(img, 1.0, color_img, 0.5, 0)

    # ============================
    # ② midpoint のみ描画 + 矢印追加
    # ============================
    for box in obb_result:
        xyxyxyxy = box.obb.xyxyxyxy
        midpoints = get_short_side_midpoints_torch(xyxyxyxy)

        # midpoint（元の2点＝青）
        for mp in midpoints:
            x, y = int(mp[0]), int(mp[1])
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

        # マスク外の midpoint（緑）＋矢印
        valid_pts = select_non_masked_midpoint(midpoints, seg_result.masks, h, w)

        for green_pt in valid_pts:
            gx, gy = int(green_pt[0]), int(green_pt[1])
            cv2.circle(img, (gx, gy), 6, (0, 255, 0), -1)

            # --- 近い青 midpoint に向けて矢印を描画 ---
            dists = [torch.norm(green_pt - blue_pt).item() for blue_pt in midpoints]
            nearest_blue = midpoints[int(np.argmin(dists))]
            bx, by = int(nearest_blue[0]), int(nearest_blue[1])

            cv2.arrowedLine(
                img,
                (gx, gy), (bx, by),
                (0, 255, 0),
                3,
                tipLength=0.3
            )

    # ============================
    # 保存
    # ============================
    save_path = os.path.join(output_dir, os.path.basename(obb_result.path))
    cv2.imwrite(save_path, img)
    print(f"保存しました: {save_path}")
