import os
import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# 所有视频编号（01 - 80）
#python home/haoding/Wenzheng/surgical_prompt_pipeline/test/test_split_scissor_clipper.py
video_ids = [f"{i:02d}" for i in range(1, 20)]

# 路径定义
triplet_base = "/mnt/disk0/haoding/cholec80_dt/gpt_response"
gt_base = "/mnt/disk0/haoding/cholec80/tool_annotations"
output_path = "/home/haoding/DT_SPR_utils/SAM2/automate_GPT/test.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 工具和目标列表
tool_list = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator"]
target_list = ["Specimen Bag"]
tool_all = tool_list + [t.replace(" ", "") for t in target_list]  # e.g. SpecimenBag

# 提取 tools 函数
def extract_tools_safe(triplets, video_id, frame_id, error_log):
    detected = set()
    for idx, triplet in enumerate(triplets):
        if not isinstance(triplet, list) or len(triplet) < 2:
            error_log.append(f"[Triplet Error] Video {video_id}, Frame {frame_id}, Triplet Index {idx}, Value: {triplet}")
            continue
        subj, obj = triplet[0], triplet[1]
        for item in [subj, obj]:
            if item in tool_list:
                detected.add(item)
            elif item in target_list:
                mapped = item.replace(" ", "")
                detected.add(mapped)
    return detected

all_results = []
triplet_errors = []

for video_id in video_ids:
    triplet_path = os.path.join(triplet_base, f"triplets_{video_id}.json")
    gt_path = os.path.join(gt_base, f"video{video_id}-tool.txt")

    if not os.path.exists(triplet_path) or not os.path.exists(gt_path):
        print(f"[Warning] Missing file for video {video_id}, skipping...")
        continue

    try:
        with open(triplet_path, 'r') as f:
            triplet_data = json.load(f)
    except Exception as e:
        print(f"[Error] Cannot parse JSON for video {video_id}: {e}")
        continue

    predicted_tools = {}
    for item in triplet_data:
        frame_id = item.get("frame_id")
        triplets = item.get("triplets", [])
        if frame_id is None:
            continue
        real_frame = frame_id * 25
        detected = extract_tools_safe(triplets, video_id, real_frame, triplet_errors)
        predicted_tools[real_frame] = detected

    try:
        gt_df = pd.read_csv(gt_path, sep='\t')
    except Exception as e:
        print(f"[Error] Cannot read GT file for video {video_id}: {e}")
        continue

    gt_dict = gt_df.set_index("Frame").to_dict(orient="index")
    records = []
    for real_frame, pred_present in predicted_tools.items():
        if real_frame not in gt_dict:
            print(f"[Warning] Frame {real_frame} not found in GT for video {video_id}, skipping...")
            continue
        gt_row = gt_dict[real_frame]
        gt_present = {tool for tool in tool_all if gt_row.get(tool, 0) == 1}

        for tool in tool_all:
            records.append({
                "frame": real_frame,
                "tool": tool,
                "gt": int(tool in gt_present),
                "pred": int(tool in pred_present),
                "video": video_id
            })

    results_df = pd.DataFrame(records)

    for tool in tool_all:
        sub_df = results_df[results_df['tool'] == tool]
        if len(sub_df) == 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            sub_df["gt"], sub_df["pred"], average='binary', zero_division=0
        )
        all_results.append({
            "Video": video_id,
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
        f.write(f"{'Tool':<15}{'Precision':>10}  {'Recall':>6}  {'F1':>6}\n")
        for row in video_rows:
            tool = row["Tool"]
            precision = f"{row['Precision']:.3f}"
            recall = f"{row['Recall']:.3f}"
            f1 = f"{row['F1']:.3f}"
            f.write(f"{tool:<15}{precision:>10}  {recall:>6}  {f1:>6}\n")
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
    f.write(f"{'Tool':<15}{'Precision':>10}  {'Recall':>6}  {'F1':>6}\n")

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
            f.write(f"{tool:<15}{p_avg:>10.3f}  {r_avg:>6.3f}  {f_avg:>6.3f}\n")
        else:
            f.write(f"{tool:<15}{'N/A':>10}  {'N/A':>6}  {'N/A':>6}\n")

# 写入错误日志
error_log_path = output_path.replace(".txt", "_errors.txt")
with open(error_log_path, "w") as f:
    for line in triplet_errors:
        f.write(line + "\n")

print(f"✅ 所有视频处理完成，格式化结果保存到: {output_path}")
print(f"⚠️ 异常 triplet 日志保存到: {error_log_path}")
