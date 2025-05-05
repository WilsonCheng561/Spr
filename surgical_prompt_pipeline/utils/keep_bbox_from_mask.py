"""
Keep only pixels inside SAM2 bounding‑boxes; everything else is black.
SAM2 mask → bounding boxes → crops → GPT‑4o → triplet JSON.

Author : Wenzheng Cheng
Date   : 2025‑04‑20

run:python /home/haoding/Wenzheng/surgical_prompt_pipeline/utils/keep_bbox_from_mask.py
"""
#这个是把一帧几个物体的 mask 提取出来，并还是保存为一帧，让其他的地方黑

import os
import cv2
import numpy as np
from typing import List

# ---------------------------- Paths ----------------------------------

MASK_ROOT = "/mnt/disk0/haoding/cholec80_dt/masks"
IMG_ROOT  = "/mnt/disk0/haoding/cholec80/annotated_data"
OUT_ROOT  = "/mnt/disk0/haoding/cholec80/box_only"

PAD_RATIO  = 0.05     # 5 % extra context
MIN_AREA   = 500      # ignore tiny blobs

# -------------------- bbox utilities --------------------------------
# def extract_boxes(mask: np.ndarray) -> List[List[int]]:
#     """Return list of [x1,y1,x2,y2] boxes from binary/label mask."""
#     if mask.ndim == 3:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
#     num_lbl, labels = cv2.connectedComponents(bin_mask)

#     boxes = []
#     for lbl in range(1, num_lbl):
#         ys, xs = np.where(labels == lbl)
#         if ys.size < MIN_AREA_PX:
#             continue
#         boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

#     # greedy merge for overlapping boxes
#     merged = True
#     while merged:
#         merged, new_boxes = False, []
#         while boxes:
#             a = boxes.pop(0)
#             ax1, ay1, ax2, ay2 = a
#             tmp = []
#             for b in boxes:
#                 bx1, by1, bx2, by2 = b
#                 ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#                 ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#                 if ix1 < ix2 and iy1 < iy2:     # overlap
#                     ax1, ay1 = min(ax1, bx1), min(ay1, by1)
#                     ax2, ay2 = max(ax2, bx2), max(ay2, by2)
#                     merged = True
#                 else:
#                     tmp.append(b)
#             new_boxes.append([ax1, ay1, ax2, ay2])
#             boxes = tmp
#         boxes = new_boxes
#     return boxes


def extract_boxes(mask: np.ndarray, min_area: int = 500) -> List[List[int]]:
    """
    One bounding‑box per semantic label.
    Works with single‑channel (class index) or RGB masks.
    """
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask_flat = (mask[:,:,0].astype(np.uint32) << 16) + \
                    (mask[:,:,1].astype(np.uint32) << 8)  + \
                     mask[:,:,2].astype(np.uint32)
    else:
        mask_flat = mask.astype(np.uint32)

    boxes = []
    for lbl in np.unique(mask_flat):
        if lbl == 0:             # background
            continue
        ys, xs = np.where(mask_flat == lbl)
        if ys.size < min_area:
            continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return boxes

def pad_box(box, w, h, ratio=PAD_RATIO):
    x1,y1,x2,y2 = box
    dx,dy = int((x2-x1)*ratio), int((y2-y1)*ratio)
    return [max(0,x1-dx), max(0,y1-dy),
            min(w-1,x2+dx), min(h-1,y2+dy)]

# ------------------- per‑video processing ---------------------------

def keep_bbox_for_video(vnum: int):
    vid       = f"video{vnum:02d}"
    mask_dir  = os.path.join(MASK_ROOT, vid)
    img_dir   = os.path.join(IMG_ROOT, vid, "ws_0", "images")
    out_dir   = os.path.join(OUT_ROOT, vid)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(mask_dir):
        print(f"[Skip] {mask_dir} not found.")
        return

    for mname in sorted(os.listdir(mask_dir)):
        stem, ext = os.path.splitext(mname)
        if ext.lower() != ".png" or not stem.isdigit():
            continue                        # 跳过非数字 mask
        idx       = int(stem)               # 00000 -> 0
        img_name  = f"{idx*25:07d}.jpg"     # 0 -> 0000000.jpg

        mpath = os.path.join(mask_dir, mname)
        ipath = os.path.join(img_dir, img_name)
        if not os.path.isfile(ipath):
            print(f"[Warn] {ipath} missing")
            continue

        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        img  = cv2.imread(ipath)
        if mask is None or img is None:
            continue

        h, w = img.shape[:2]
        boxes = [pad_box(b,w,h) for b in extract_boxes(mask, MIN_AREA)]
        if not boxes:
            # #没有任何有效的 bbox，保存整个黑图
            # blank = np.zeros_like(img)
            # cv2.imwrite(os.path.join(out_dir, img_name), blank)
            # continue
            #没有任何有效的 bbox，保存整个原图
            cv2.imwrite(os.path.join(out_dir, img_name), img)
            continue


        kept = np.zeros_like(img)
        for x1,y1,x2,y2 in boxes:
            kept[y1:y2+1, x1:x2+1] = img[y1:y2+1, x1:x2+1]

        cv2.imwrite(os.path.join(out_dir, img_name), kept)

    print(f"[Done] {vid} -> {out_dir}")

# ------------------------------ main --------------------------------

if __name__ == "__main__":
    VIDEO_RANGE = range(1, 20)     
    for v in VIDEO_RANGE:
        keep_bbox_for_video(v)
