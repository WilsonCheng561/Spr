"""
SAM2 mask → bounding boxes → crops → GPT‑4o → triplet JSON.
"""


import os
import cv2
import json
import time
import uuid
import numpy as np
from typing import List, Tuple
from openai import OpenAI
from modules.prompt_builder import build_pdf_prompt, build_text_prompt

# python /home/haoding/Wenzheng/surgical_prompt_pipeline/draw_box_from_mask.py
#这个是把一帧几个物体的 mask 在原来帧上框出来

# ----------------------------- Config ---------------------------------

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

MASK_ROOT   = "/mnt/disk0/haoding/cholec80_dt/masks"
IMG_ROOT    = "/mnt/disk0/haoding/cholec80/annotated_data"
OUT_ROOT    = "/mnt/disk0/haoding/cholec80/box_only"

PAD_RATIO    = 0.05       # 对每个 box 扩展 5% 的边距
MERGE_IOU_T  = 0.0        # 合并框时的 IOU 阈值（此处使用简单重叠检测）
MIN_AREA_PX  = 500        # 最小区域像素数，过滤小噪声

LLM_MODEL      = "gpt-4o"
MAX_RETRIES    = 1
TEMPERATURE    = 0.0
MAX_TOKENS     = 800

# ---------------------- Helper – bounding boxes ------------------------

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


def extract_boxes(mask: np.ndarray, min_area_px: int = 500) -> List[List[int]]:
    """
    One bounding‑box per semantic label.
    Works for:
      • single‑channel integer masks (0 = bg, 1/2/3 … = object id)
      • RGB masks where each label has an unique (R,G,B) tuple
    """
    # --- step 1: turn RGB into single int if needed  ------------------
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask_flat = (mask[:, :, 0].astype(np.uint32) << 16) + \
                    (mask[:, :, 1].astype(np.uint32) << 8)  + \
                     mask[:, :, 2].astype(np.uint32)
    else:                              # already single channel
        mask_flat = mask.astype(np.uint32)

    labels = np.unique(mask_flat)
    labels = labels[labels != 0]       # drop background

    boxes = []
    for lbl in labels:
        ys, xs = np.where(mask_flat == lbl)
        if ys.size < min_area_px:      # too small → noise
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes


def pad_box(box: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * PAD_RATIO), int((y2 - y1) * PAD_RATIO)
    return [max(0, x1 - dx), max(0, y1 - dy),
            min(w-1, x2 + dx), min(h-1, y2 + dy)]


# -------------------- per‑video processing --------------------------

def draw_boxes_for_video(vnum: int):
    vid = f"video{vnum:02d}"
    mask_dir = os.path.join(MASK_ROOT, vid)
    img_dir  = os.path.join(IMG_ROOT, vid, "ws_0", "images")
    out_dir  = os.path.join(OUT_ROOT, vid)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(mask_dir):
        print(f"[Skip] {mask_dir} not found.")
        return

    for mname in sorted(os.listdir(mask_dir)):
        try:
            idx = int(os.path.splitext(mname)[0])# 00000 -> 0
            img_name = f"{idx*25:07d}.jpg"                 # 0 -> 0000000.jpg
        except ValueError:
            print(f"[Skip] non‑numeric mask file: {mname}")
            continue
        mpath = os.path.join(mask_dir, mname)
        ipath = os.path.join(img_dir, img_name)
        if not os.path.isfile(ipath):
            print(f"[Warn] raw image {ipath} missing")
            continue

        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        img  = cv2.imread(ipath)
        if mask is None or img is None:
            continue

        h, w = img.shape[:2]
        for x1,y1,x2,y2 in (pad_box(b,w,h) for b in extract_boxes(mask)):
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

        cv2.imwrite(os.path.join(out_dir, img_name), img)
    print(f"[Done] {vid} → {out_dir}")

# ----------------------- Helper – strip JSON --------------------------

def strip_JSON(text: str) -> dict:
    """Robustly parse GPT JSON output."""
    cleaned = text.strip("```").strip().lstrip("json").strip()
    if not cleaned:
        return {"triplets": [], "analysis": "Empty response"}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[JSON error] {e} | content: {cleaned[:120]}...")
        return {"triplets": [], "analysis": "Invalid JSON"}

# ----------------------- Helper – upload crop -------------------------

def upload_image(local_path: str) -> str:
    """
    Upload local image to a public URL that OpenAI can fetch.
    Implement e.g. S3/GitHub‑Pages; here we fall back to file:// for testing.
    """
    # TODO: replace with real uploader
    return "file://" + local_path   # <‑‑ Change to your uploader result URL

# ----------------------- GPT call for one crop ------------------------

def llm_triplet(base_msgs: list, crop_url: str,
                vid: str, frame_idx: str, crop_idx: int) -> dict:
    msgs = base_msgs + [{
        "role": "user",
        "content": [
            {"type": "text",
             "text": f"Video {vid}, frame {frame_idx}, crop {crop_idx}. Please analyze."},
            {"type": "image_url", "image_url": {"url": crop_url}}
        ]
    }]
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=msgs,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            parsed = strip_JSON(resp.choices[0].message.content)
            if parsed.get("triplets"):
                return parsed
            print(f"[Warn] empty triplets, retry {attempt}")
        except Exception as e:
            print(f"[Error] GPT call {attempt}: {e}")
        time.sleep(1)
    return {"triplets": [], "analysis": f"Failed after {MAX_RETRIES} retries"}

# ----------------------------- Main per video ------------------------

def process_video(video_id: str):
    mask_dir  = os.path.join(MASK_DIR,  video_id)
    frame_dir = os.path.join(FRAME_DIR, video_id)
    out_dir   = os.path.join(RESULT_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) build base prompts only once per video ----------
    base_messages = [
        {"role": "system",
         "content": "You are an advanced AI with expertise in laparoscopic surgery."},
        {"role": "user", "content": build_pdf_prompt()},
    ]
    # let GPT "read" PDFs
    try:
        resp1 = client.chat.completions.create(
            model=LLM_MODEL, messages=base_messages,
            temperature=0.0, max_tokens=800)
        base_messages.append({"role": "assistant",
                              "content": resp1.choices[0].message.content.strip()})
    except Exception as e:
        print(f"[PDF] error: {e}")

    base_messages.append({"role": "user", "content": build_text_prompt()})
    try:
        resp2 = client.chat.completions.create(
            model=LLM_MODEL, messages=base_messages,
            temperature=0.0, max_tokens=300)
        base_messages.append({"role": "assistant",
                              "content": resp2.choices[0].message.content.strip()})
    except Exception as e:
        print(f"[Text prompt] error: {e}")

    # ---------- 2) iterate frames ----------
    for fname in sorted(os.listdir(mask_dir)):
        frame_idx = os.path.splitext(fname)[0]          # '000123'
        mask_path  = os.path.join(mask_dir, fname)
        frame_path = os.path.join(frame_dir, fname).replace(".png", ".jpg")

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        img  = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if mask is None or img is None:
            print(f"[Skip] {fname}")
            continue

        h, w = img.shape[:2]
        boxes = [pad_box(b, w, h) for b in extract_boxes(mask)]

        frame_triplets, analyses, crop_urls = [], [], []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = img[y1:y2+1, x1:x2+1]
            crop_name = f"{frame_idx}_{i}.jpg"
            local_crop = os.path.join(CROP_DIR, video_id, crop_name)
            os.makedirs(os.path.dirname(local_crop), exist_ok=True)
            cv2.imwrite(local_crop, crop)

            url = upload_image(local_crop)
            crop_urls.append(url)

            result = llm_triplet(base_messages, url, video_id, frame_idx, i)
            frame_triplets.extend(result.get("triplets", []))
            if result.get("analysis"):
                analyses.append(result["analysis"])

        # deduplicate triplets (tuple‑hashable)
        uniq_triplets = sorted(list({tuple(t) for t in frame_triplets}))

        frame_json = {
            "boxes": boxes,
            "crop_urls": crop_urls,
            "triplets": uniq_triplets,
            "analysis": analyses
        }
        with open(os.path.join(out_dir, f"{frame_idx}.json"), "w") as f:
            json.dump(frame_json, f, indent=2, ensure_ascii=False)

        print(f"[{video_id}] frame {frame_idx}: "
              f"{len(boxes)} crops → {len(uniq_triplets)} triplets")



# if __name__ == "__main__":
    # videos = sorted(os.listdir(MASK_DIR))
    # for vid in videos:
    #     process_video(vid)

if __name__ == "__main__":
    # choose videos to test  (01‑05)
    VIDEO_RANGE = range(1, 2)             # e.g. range(1,81) for all
    for v in VIDEO_RANGE:
        draw_boxes_for_video(v)
