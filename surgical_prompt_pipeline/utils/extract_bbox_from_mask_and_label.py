"""
Crop objects from SAM2 masks ➜ save crops
+ generate per‑frame ground‑truth tables (tool / all).

Author : Wenzheng Cheng
Date   : 2025‑04‑22

python /home/haoding/Wenzheng/surgical_prompt_pipeline/utils/extract_bbox_from_mask_and_label.py
"""
#这个是把一帧几个物体的 mask 提取出来单独保存为几张图片

import os, cv2, numpy as np
from typing import List, Tuple
from tqdm import tqdm        

# -------------------------------config-------------------------------------
MASK_ROOT = "/mnt/disk0/haoding/cholec80_dt/masks"#输入mask
IMG_ROOT  = "/mnt/disk0/haoding/cholec80/annotated_data"#输入原图
CROP_ROOT = "/mnt/disk0/haoding/cholec80/extracted_frames_bbox_clean"#输出保存的位置clean没有语义标签，否则有
ANN_ROOT  = "/mnt/disk0/haoding/cholec80/tool_annotations_sam2"#输出同时提取的sam2的标签

PAD_RATIO  = 0.05
MIN_AREA   = 300
MERGE_DIST = 25      # px

LABELS = {
    "Gallbladder"   : (  0,   0, 255),
    "LeftGrasper"   : (  0, 255,   0),
    "TOPGrasper"    : (255,   0,   0),
    "RightGrasper"  : (  0, 225, 225),
    "Bipolar"       : (255,   0, 255),
    "Hook"          : (255, 255,   0),
    "Scissors"      : (128, 128, 128),
    "Clipper"       : (  0,   0, 128),
    "Irrigator"     : (  0, 128,   0),
    "SpecimenBag"   : (128,   0,   0),
}

LABEL_IDS = {
    "Gallbladder":    "01",
    "LeftGrasper":    "02",
    "TOPGrasper":     "03",
    "RightGrasper":   "04",
    "Bipolar":        "05",
    "Hook":           "06",
    "Scissors":       "07",
    "Clipper":        "08",
    "Irrigator":      "09",
    "SpecimenBag":    "10",
}


COL_TOOL = ["Grasper","Bipolar","Hook","Scissors","Clipper","Irrigator","SpecimenBag"]
COL_ALL  = ["LeftGrasper","RightGrasper","TOPGrasper","Bipolar","Hook",
            "Scissors","Clipper","Irrigator","SpecimenBag"]



# ---------------- merge nearby boxes --------------------------------
def merge_boxes(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    out = []
    while boxes:
        x1,y1,x2,y2 = boxes.pop(0)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        keep = []
        for bx1,by1,bx2,by2 in boxes:
            bcx,bcy = (bx1+bx2)//2, (by1+by2)//2
            if abs(bcx-cx)<=MERGE_DIST and abs(bcy-cy)<=MERGE_DIST:
                x1,y1 = min(x1,bx1), min(y1,by1)
                x2,y2 = max(x2,bx2), max(y2,by2)
            else:
                keep.append((bx1,by1,bx2,by2))
        out.append((x1,y1,x2,y2))
        boxes = keep
    return out

# ------------- iterate objects in one mask --------------------------
def crop_objects(mask: np.ndarray, img: np.ndarray):
    """yield (label, crop_img)"""
    if mask.ndim != 3 or mask.shape[2] not in (3,4):
        return

    b_chan, g_chan, r_chan = mask[:,:,0], mask[:,:,1], mask[:,:,2]
    h, w = img.shape[:2]

    for label, (b_val, g_val, r_val) in LABELS.items():
        pix = (b_chan==b_val)&(g_chan==g_val)&(r_chan==r_val)
        if not pix.any(): continue

        comp_map = pix.astype(np.uint8)*255
        num, comp = cv2.connectedComponents(comp_map)

        boxes=[]
        for cid in range(1,num):
            ys,xs = np.where(comp==cid)
            if ys.size < MIN_AREA: continue
            boxes.append((xs.min(),ys.min(),xs.max(),ys.max()))
        if not boxes: continue

        for x1,y1,x2,y2 in merge_boxes(boxes):
            dx,dy=int((x2-x1)*PAD_RATIO), int((y2-y1)*PAD_RATIO)
            x1p,y1p=max(0,x1-dx), max(0,y1-dy)
            x2p,y2p=min(w-1,x2+dx), min(h-1,y2+dy)
            yield label, img[y1p:y2p+1, x1p:x2p+1]

# ------------------------- per‑video ---------------------------------
def process_video(vnum:int):
    vid=f"video{vnum:02d}"
    print(f"\n⚙️  Start {vid}")

    mask_dir=os.path.join(MASK_ROOT,vid)
    img_dir =os.path.join(IMG_ROOT,vid,"ws_0","images")
    crop_dir=os.path.join(CROP_ROOT,vid)
    ann_dir =os.path.join(ANN_ROOT,vid)
    os.makedirs(crop_dir,exist_ok=True)
    os.makedirs(ann_dir ,exist_ok=True)

    frame_labels={}
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png") and f[:-4].isdigit()])

    for mname in tqdm(mask_files, desc=vid, ncols=80):
        idx   = int(os.path.splitext(mname)[0])
        frame7= f"{idx*25:07d}"
        mpath = os.path.join(mask_dir,mname)
        ipath = os.path.join(img_dir,f"{frame7}.jpg")
        if not os.path.isfile(ipath):
            continue

        mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[Warn] {vid} frame {frame7}: mask read failed")
            continue
        if mask.ndim==2:                 # 异常的 (480, 854)palette → 再读成 BGR
            mask = cv2.imread(mpath, cv2.IMREAD_COLOR)

        img  = cv2.imread(ipath)
        if img is None:
            print(f"[Warn] {vid} frame {frame7}: image missing")
            continue

        if mask.ndim!=3 or mask.shape[2] not in (3,4):
            print(f"[Warn] {vid} frame {frame7}: invalid mask shape {mask.shape}")
            continue

        present=set()
        for label,crop in crop_objects(mask,img):
            present.add(label)
            # cv2.imwrite(os.path.join(crop_dir,f"{frame7}_{label}.jpg"),crop)#给的语义标签，保存在"/mnt/disk0/haoding/cholec80/extracted_frames_bbox"

            label_id = LABEL_IDS.get(label, "00")  # 默认00代表没有标签，这里改成了数字编号掩盖语义信息
            cv2.imwrite(os.path.join(crop_dir, f"{frame7}_{label_id}.jpg"), crop)


        frame_labels[frame7]=present

    # ---------------- write annotation tables ------------------------
    if not frame_labels:
        print(f"[Skip] {vid} empty")
        return

    tool_lines=["Frame\t"+"\t".join(COL_TOOL)]
    for f in sorted(frame_labels,key=lambda x:int(x)):
        p=frame_labels[f]
        row=[f]
        row.append("1" if p & {"LeftGrasper","RightGrasper","TOPGrasper"} else "0")
        for t in COL_TOOL[1:]:
            row.append("1" if t in p else "0")
        tool_lines.append("\t".join(row))
    with open(os.path.join(ann_dir,f"{vid}-tool.txt"),"w") as fp:
        fp.write("\n".join(tool_lines))

    all_lines=["Frame\t"+"\t".join(COL_ALL)]
    for f in sorted(frame_labels,key=lambda x:int(x)):
        p=frame_labels[f]
        all_lines.append("\t".join([f]+["1" if t in p else "0" for t in COL_ALL]))
    with open(os.path.join(ann_dir,f"{vid}-all.txt"),"w") as fp:
        fp.write("\n".join(all_lines))

    print(f"✅  Done  {vid}: crops→{crop_dir} | ann→{ann_dir}")

# ------------------------------ main ---------------------------------
if __name__=="__main__":
    VIDEO_RANGE = range(1, 21)  
    for v in VIDEO_RANGE:
        process_video(v)