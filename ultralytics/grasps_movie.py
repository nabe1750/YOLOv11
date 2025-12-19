import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
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
            if 0 <= y < orig_h and 0 <= x < orig_w and m[y, x] > 0.5:
                inside = True
                break

        if not inside:
            selected.append(mp)

    return selected


# =============================== 推論処理 ===============================

DRAW_SEG_MASK = False     # セグメンテーションを描画するか

input_video = "../../../data/movie/6_30fps_rotate.mov"
output_dir = create_increment_dir(parent="../output", base="vector_results")

obb_model = YOLO("../runs/obb/train14/weights/best.pt")
seg_model = YOLO("../runs/segment/train30/weights/best.pt")

obb_stream = obb_model.predict(source=input_video, stream=True)
seg_stream = seg_model.predict(source=input_video, stream=True)

cap = cv2.VideoCapture(input_video)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
cap.release()

input_filename = os.path.basename(input_video)
output_path = os.path.join(output_dir, input_filename)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0

start_time = time.time()

for obb_result, seg_result in zip(obb_stream, seg_stream):

    img = obb_result.orig_img.copy()
    masks = seg_result.masks

    # ====================== 追加：セグメンテーション描画 ======================
    if DRAW_SEG_MASK and masks is not None:
        mask_tensor = masks.data.cpu().numpy()  # (N, Hm, Wm)
        mask_tensor = np.max(mask_tensor, axis=0)  # 全マスク結合
        mask_resized = cv2.resize(mask_tensor, (img.shape[1], img.shape[0]))
        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        color = np.array([255, 0, 0], dtype=np.uint8)  # 赤
        overlay = img.copy()
        overlay[mask_bin == 1] = color

        alpha = 0.4
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


    # ====================== OBB 結果の描画 ======================
    for box in obb_result:
        xyxyxyxy = box.obb.xyxyxyxy
        midpoints = get_short_side_midpoints_torch(xyxyxyxy)

        for mp in midpoints:
            x, y = int(mp[0]), int(mp[1])
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

        h, w = img.shape[:2]
        valid_pts = select_non_masked_midpoint(midpoints, masks, h, w)

        for gp in valid_pts:
            xg, yg = int(gp[0]), int(gp[1])
            cv2.circle(img, (xg, yg), 6, (0, 255, 0), -1)
            
            for mp in midpoints:
                if torch.allclose(gp, mp, atol=1e-3):
                    continue
                
                xb, yb = int(mp[0]), int(mp[1])
                
                cv2.arrowedLine(
                    img,
                    (xg, yg),      # 選択点 → 
                    (xb, yb),      # 非選択点
                    (0, 255, 255),
                    2,
                    tipLength=0.3
                )

    writer.write(img)
    print(f"Frame {frame_idx} written")
    frame_idx += 1


writer.release()

elapsed = time.time() - start_time
print(f"動画を書き出しました: {output_path}")
print(f"総処理時間: {elapsed:.2f} 秒")
print(f"1フレーム平均: {elapsed / frame_idx:.4f} 秒/フレーム")
