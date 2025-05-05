import os
import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# 所有视频编号（01 - 80）
#python /home/haoding/Wenzheng/surgical_prompt_pipeline/test/test.py

# ===============================================================
# >>>在这⾥切换评测模式
#     "triplet"  : 整帧三元组 JSON
#     "bbox"   : BBox-tool JSON
EVAL_MODE = "triplet"     
# ===============================================================

# 所有视频编号
video_ids = [f"{i:02d}" for i in range(1, 6)]

# >>> 新增/修改 ②：根据模式选取不同路径
if EVAL_MODE == "triplet":
    triplet_base = "/mnt/disk0/haoding/cholec80_dt/gpt_response_all"
    gt_base      = "/mnt/disk0/haoding/cholec80/tool_annotations"
else: 
    triplet_base = "/mnt/disk0/haoding/cholec80_dt/gpt_response_bbox_clean"
    gt_base      = "/mnt/disk0/haoding/cholec80/tool_annotations_sam2"

output_path = f"/home/haoding/Wenzheng/surgical_prompt_pipeline/test/test_{EVAL_MODE}.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 工具/目标列表
tool_list   = ["Grasper", "Bipolar", "Hook", "Irrigator"]
target_list = ["Specimen Bag"]
tool_all    = tool_list + ["ScissorOrClipper"] + [t.replace(" ", "") for t in target_list]

def extract_tools_safe(triplets, video_id, frame_id, error_log):
    detected = set()
    for idx, triplet in enumerate(triplets):
        if not isinstance(triplet, list) or len(triplet) < 2:
            error_log.append(f"[Triplet Error] Video {video_id}, Frame {frame_id}, Triplet Idx {idx}, Val: {triplet}")
            continue
        for item in triplet[:2]:
            _map_and_add(item, detected)
    return detected

def _map_and_add(label, bucket: set):
    if label in tool_list:
        bucket.add(label)
    elif label in ["Scissors", "Clipper"]:
        bucket.add("ScissorOrClipper")
    elif label in target_list:
        bucket.add(label.replace(" ", ""))

# ---------- 主循环 ----------
all_results, triplet_errors = [], []
for vid in video_ids:
    pred_path = os.path.join(triplet_base, f"triplets_{vid}.json")

    if EVAL_MODE == "triplet":
        gt_path = os.path.join(gt_base, f"video{vid}-tool.txt")
    else:
        gt_path = os.path.join(gt_base, f"video{vid}", f"video{vid}-tool.txt")

    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        print(f"[Warning] Missing file for video {vid}, skip")
        continue

    try:
        with open(pred_path) as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"[Error] Cannot parse JSON for video {vid}: {e}")
        continue

    predicted_tools = defaultdict(set)  # real_frame → {tools}
    for item in pred_data:
        frame_id = item.get("frame_id")
        if frame_id is None: continue
        real_frame = frame_id * 25

        # >>> 两种解析分支
        if EVAL_MODE == "triplet":
            det = extract_tools_safe(item.get("triplets", []), vid, real_frame, triplet_errors)
            predicted_tools[real_frame] |= det
        else:  # single
            label = str(item.get("label", "")).strip().strip('"').strip("'")
            _map_and_add(label, predicted_tools[real_frame])

    # 读取 GT
    try:
        gt_df = pd.read_csv(gt_path, sep='\t')
    except Exception as e:
        print(f"[Error] Cannot read GT for video {vid}: {e}")
        continue
    gt_dict = gt_df.set_index("Frame").to_dict(orient="index")

    # 生成逐帧记录
    records = []
    for real_frame, pred_set in predicted_tools.items():
        if real_frame not in gt_dict: continue
        row = gt_dict[real_frame]

        gt_set = set()
        for t in tool_list:
            if row.get(t, 0) == 1: gt_set.add(t)
        if row.get("Scissors", 0)==1 or row.get("Clipper",0)==1:
            gt_set.add("ScissorOrClipper")
        for tg in target_list:
            k = tg.replace(" ", "")
            if row.get(k, 0)==1: gt_set.add(k)

        for t in tool_all:
            records.append({
                "frame": real_frame, "tool": t,
                "gt": int(t in gt_set), "pred": int(t in pred_set),
                "video": vid
            })

    if not records:
        print(f"[Warning] No valid frame for video {vid}")
        continue

    results_df = pd.DataFrame(records)

    for tool in tool_all:
        sub_df = results_df[results_df['tool'] == tool]
        if len(sub_df) == 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            sub_df["gt"], sub_df["pred"], average='binary', zero_division=0
        )
        all_results.append({
            "Video": vid,
            "Tool": tool,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3)
        })

# 写入结果
valid_results = []
with open(output_path, "w") as f:
    for video_id in sorted(set(r["Video"] for r in all_results)):
        video_rows = list(filter(lambda x: x["Video"] == video_id, all_results))
        if all(row["Precision"] == 0 and row["Recall"] == 0 and row["F1"] == 0 for row in video_rows):
            print(f"[Skip] Video {video_id} is invalid (all-zero metrics), skipping in total summary.")
            continue

        f.write(f"video{video_id}：\n")
        f.write(f"{'Tool':<20}{'Precision':>10}  {'Recall':>6}  {'F1':>6}\n")
        for row in video_rows:
            tool = row["Tool"]
            precision = f"{row['Precision']:.3f}"
            recall = f"{row['Recall']:.3f}"
            f1 = f"{row['F1']:.3f}"
            f.write(f"{tool:<20}{precision:>10}  {recall:>6}  {f1:>6}\n")
            valid_results.append(row)
        f.write("\n")

    f.write("=" * 40 + "\n")
    f.write(" Overall summary (excluding invalid videos):\n")

    if valid_results:
        total_precision = sum(r["Precision"] for r in valid_results) / len(valid_results)
        total_recall = sum(r["Recall"] for r in valid_results) / len(valid_results)
        total_f1 = sum(r["F1"] for r in valid_results) / len(valid_results)

        f.write(f"{'Average Precision':<20}: {total_precision:.3f}\n")
        f.write(f"{'Average Recall':<20}: {total_recall:.3f}\n")
        f.write(f"{'Average F1-Score':<20}: {total_f1:.3f}\n")
    else:
        f.write("No valid results found.\n")

    f.write("\n Per-tool average metrics (excluding invalid videos):\n")
    f.write(f"{'Tool':<20}{'Precision':>10}  {'Recall':>6}  {'F1':>6}\n")

    tool_metrics = defaultdict(lambda: {"P": [], "R": [], "F": []})
    for row in valid_results:
        tool = row["Tool"]
        tool_metrics[tool]["P"].append(row["Precision"])
        tool_metrics[tool]["R"].append(row["Recall"])
        tool_metrics[tool]["F"].append(row["F1"])

    for tool in tool_all:
        if tool in tool_metrics:
            p_avg = sum(tool_metrics[tool]["P"]) / len(tool_metrics[tool]["P"])
            r_avg = sum(tool_metrics[tool]["R"]) / len(tool_metrics[tool]["R"])
            f_avg = sum(tool_metrics[tool]["F"]) / len(tool_metrics[tool]["F"])
            f.write(f"{tool:<20}{p_avg:>10.3f}  {r_avg:>6.3f}  {f_avg:>6.3f}\n")
        else:
            f.write(f"{tool:<20}{'N/A':>10}  {'N/A':>6}  {'N/A':>6}\n")

# 写入错误日志
error_log_path = output_path.replace(".txt", "_errors.txt")
with open(error_log_path, "w") as f:
    for line in triplet_errors:
        f.write(line + "\n")

print(f"✅ 所有视频处理完成，格式化结果保存到: {output_path}")
print(f"⚠️ 异常 triplet 日志保存到: {error_log_path}")
