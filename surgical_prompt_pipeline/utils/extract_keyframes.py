import os
import json
import shutil
from datetime import datetime

#python home/haoding/DT_SPR_utils/SAM2/automate_GPT/extract_keyframes.py
#python /home/haoding/Wenzheng/surgical_prompt_pipeline/utils/extract_keyframes.py

#从prompt.json获取key frame id，超过64则再插入伪id，把这些id对应的图片存入extracted_frames并保存到/mnt/disk0/haoding/cholec80/extracted_frames/videoxx/keyframes.txt
#这一步以后要从extracted_frames使用linux命令行cp到github本地仓库/mnt/disk0/haoding/surgical-images
#rsync -av --ignore-existing /mnt/disk0/haoding/cholec80/extracted_frames/ /mnt/disk0/haoding/surgical-images/
#rsync -av --ignore-existing /mnt/disk0/haoding/cholec80/extracted_frames_20_5/ /mnt/disk0/haoding/surgical-images/
#rsync -av --ignore-existing /mnt/disk0/haoding/cholec80/extracted_frames_all/ /mnt/disk0/haoding/surgical-images/
#rsync -av --progress /mnt/disk0/haoding/surgical-images-pertool-clean/ /mnt/disk0/haoding/surgical-images-pertool/


#在进入到generate_url

# 设置输入 & 输出根目录
input_base_dir = "/mnt/disk0/haoding/cholec80/annotated_data"
#output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames"
# output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_20_5"
# output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_all"
output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_bbox"

# 记录日志
# log_file_path = "log_extract_keyframes.txt"
# log_file_path = "home/haoding/Wenzheng/surgical_prompt_pipeline/utils/log_extract_keyframes_20_5.txt"
# log_file_path = "/home/haoding/Wenzheng/surgical_prompt_pipeline/utils/log_extract_allframes_5.txt"
log_file_path = "/home/haoding/Wenzheng/surgical_prompt_pipeline/utils/log_surgical-images-pertool_20.txt"

# 处理的视频范围1-80 , 81-97 ,101-105
video_start, video_end = 1, 20

def load_keyframes(json_path):
    """读取并解析 `prompts.json`"""
    with open(json_path, 'r') as f:
        keyframes = json.load(f)
    return keyframes

def extract_and_save_keyframes_ids(input_dir, keyframes_json, output_dir, keyframes_txt):
    """
    读取 `prompts.json`，提取关键帧图片，保存到新目录，并生成 `keyframes.txt`
    若相邻关键帧 ID 差值 >=64，则在两者之间按64的步长插入额外的 frame_id(及其对应文件名)，
    以防关键帧间隔过大。
    """

    # 1) 读取原始 keyframes
    raw_keyframes = load_keyframes(keyframes_json)
    if not raw_keyframes:
        print(f"⚠️ Warning: No keyframes found in {keyframes_json}, skipping.")
        return

    # 2) 按 frame_id 升序排序
    raw_keyframes.sort(key=lambda x: x["frame_id"])

    # 3) 生成“扩展”后的关键帧列表 (考虑差值>=64 的情况)
    expanded_keyframes = []
    expanded_keyframes.append(raw_keyframes[0])  # 第一个必然保留

    for i in range(1, len(raw_keyframes)):
        prev_frame = expanded_keyframes[-1]
        cur_frame = raw_keyframes[i]

        prev_id = prev_frame["frame_id"]
        cur_id = cur_frame["frame_id"]

        # 若相邻ID差值>=64，就在中间插入 (prev_id +64, +128, ...)
        diff = cur_id - prev_id

        while diff >= 64:
            prev_id += 64
            diff = cur_id - prev_id

            # 这里构造一个“伪”keyframe对象，文件名按 usual pattern
            # 若实际不存在，就后面复制时会 skip
            pseudo_item = {
                "frame_id": prev_id,
                "frame_file": f"{prev_id * 25:07d}.jpg"  
            }
            expanded_keyframes.append(pseudo_item)

        # 最后把当前帧也放进去
        expanded_keyframes.append(cur_frame)

    # 4) 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    keyframe_ids = []

    # 5) 依次复制关键帧 & 收集 frame_id
    # expanded_keyframes 里可能有重复ID或不严格升序 => 我们处理时过滤一下
    seen_ids = set()
    final_list = []

    for item in expanded_keyframes:
        fid = item["frame_id"]
        if fid not in seen_ids:
            seen_ids.add(fid)
            final_list.append(item)

    # 按照 frame_id 再次排序
    final_list.sort(key=lambda x: x["frame_id"])

    for item in final_list:
        frame_id = item["frame_id"]
        frame_file = item["frame_file"]
        image_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)

        # 若文件不存在, skip
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Frame {frame_file} not found, skipping...")
            continue

        # 记录 frame_id
        keyframe_ids.append(str(frame_id))

    # 6) 保存 frame_id 到 txt
    with open(keyframes_txt, "w") as f:
        f.write("\n".join(keyframe_ids))

    print(f"📄 Keyframe IDs saved to {keyframes_txt}")


def extract_and_save_keyframes(input_dir, keyframes_json, output_dir, keyframes_txt):
    """
    读取 `prompts.json`，提取关键帧图片，保存到新目录，并生成 `keyframes.txt`
    若相邻关键帧 ID 差值 >=64，则在两者之间按64的步长插入额外的 frame_id(及其对应文件名)，
    以防关键帧间隔过大。
    """

    # 1) 读取原始 keyframes
    raw_keyframes = load_keyframes(keyframes_json)
    if not raw_keyframes:
        print(f"⚠️ Warning: No keyframes found in {keyframes_json}, skipping.")
        return

    # 2) 按 frame_id 升序排序
    raw_keyframes.sort(key=lambda x: x["frame_id"])

    # 3) 生成“扩展”后的关键帧列表 (考虑差值>=64 的情况)
    expanded_keyframes = []
    expanded_keyframes.append(raw_keyframes[0])  # 第一个必然保留

    for i in range(1, len(raw_keyframes)):
        prev_frame = expanded_keyframes[-1]
        cur_frame = raw_keyframes[i]

        prev_id = prev_frame["frame_id"]
        cur_id = cur_frame["frame_id"]

        # 若相邻ID差值>=64，就在中间插入 (prev_id +64, +128, ...)
        diff = cur_id - prev_id

        while diff >= 64:
            prev_id += 64
            diff = cur_id - prev_id

            # 这里构造一个“伪”keyframe对象，文件名按 usual pattern
            # 若实际不存在，就后面复制时会 skip
            pseudo_item = {
                "frame_id": prev_id,
                "frame_file": f"{prev_id * 25:07d}.jpg"  
            }
            expanded_keyframes.append(pseudo_item)

        # 最后把当前帧也放进去
        expanded_keyframes.append(cur_frame)

    # 4) 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    keyframe_ids = []

    # 5) 依次复制关键帧 & 收集 frame_id
    # expanded_keyframes 里可能有重复ID或不严格升序 => 我们处理时过滤一下
    seen_ids = set()
    final_list = []

    for item in expanded_keyframes:
        fid = item["frame_id"]
        if fid not in seen_ids:
            seen_ids.add(fid)
            final_list.append(item)

    # 按照 frame_id 再次排序
    final_list.sort(key=lambda x: x["frame_id"])

    for item in final_list:
        frame_id = item["frame_id"]
        frame_file = item["frame_file"]
        image_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)

        # 若文件不存在, skip
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Frame {frame_file} not found, skipping...")
            continue

        # 复制关键帧图片
        shutil.copy(image_path, output_path)

        # 记录 frame_id
        keyframe_ids.append(str(frame_id))

    # 6) 保存 frame_id 到 txt
    with open(keyframes_txt, "w") as f:
        f.write("\n".join(keyframe_ids))

    print(f"📄 Keyframe IDs saved to {keyframes_txt}")


def extract_and_save_keyframes_expand_neighbor(input_dir, keyframes_json, output_dir, keyframes_txt):
    """
    扩展版：提取 prompts.json 中关键帧，间隔 >=64 插值伪帧，每帧再往前后拓展2帧，共5帧组。
    最终保存两份文件：
    - keyframes.txt：原始中心关键帧
    - keyframes_all.txt：所有扩展帧，用于训练使用
    """

    raw_keyframes = load_keyframes(keyframes_json)
    if not raw_keyframes:
        print(f"⚠️ Warning: No keyframes found in {keyframes_json}, skipping.")
        return

    raw_keyframes.sort(key=lambda x: x["frame_id"])

    # Step 1: 插值
    expanded_keyframes = [raw_keyframes[0]]
    for i in range(1, len(raw_keyframes)):
        prev = expanded_keyframes[-1]["frame_id"]
        curr = raw_keyframes[i]["frame_id"]
        while curr - prev >= 64:
            prev += 64
            pseudo = {
                "frame_id": prev,
                "frame_file": f"{prev * 25:07d}.jpg"
            }
            expanded_keyframes.append(pseudo)
        expanded_keyframes.append(raw_keyframes[i])

    # Step 2: 去重 + 排序
    seen = set()
    final_list = []
    for item in expanded_keyframes:
        fid = item["frame_id"]
        if fid not in seen:
            seen.add(fid)
            final_list.append(item)
    final_list.sort(key=lambda x: x["frame_id"])
    center_frame_ids = [item["frame_id"] for item in final_list]

    # Step 3: 拓展每帧 ±2
    neighbor_ids = set()
    for fid in center_frame_ids:
        for offset in range(-2, 3):
            nid = fid + offset
            if nid >= 0:
                neighbor_ids.add(nid)
    neighbor_ids = sorted(neighbor_ids)

    # Step 4: 复制图像
    os.makedirs(output_dir, exist_ok=True)
    saved_ids = []

    for fid in neighbor_ids:
        fname = f"{fid * 25:07d}.jpg"
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            saved_ids.append(str(fid))
        else:
            print(f"⚠️ Frame {fname} not found, skipping.")

    # Step 5: 保存文件
    with open(os.path.join(output_dir, "keyframes_all.txt"), "w") as f:
        f.write("\n".join(saved_ids))
    with open(keyframes_txt, "w") as f:
        f.write("\n".join([str(fid) for fid in center_frame_ids]))

    print(f"✅ {len(center_frame_ids)} center frames, {len(saved_ids)} total saved.")


def extract_and_save_frames_all(input_dir, output_dir, keyframes_txt):
    """
    提取并保存 input_dir 下的所有 jpg 图像帧到 output_dir，
    同时生成 keyframes_txt，记录所有 frame_id（按升序）。
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    keyframe_ids = []
    seen_ids = set()

    # 遍历所有 jpg 文件
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            try:
                frame_id = int(os.path.splitext(file)[0]) // 25
            except ValueError:
                print(f"⚠️ Warning: 无法解析文件名 {file}，跳过...")
                continue

            if frame_id in seen_ids:
                continue

            seen_ids.add(frame_id)
            keyframe_ids.append((frame_id, file))

    # 按 frame_id 升序排列
    keyframe_ids.sort(key=lambda x: x[0])

    final_ids = []

    for frame_id, file in keyframe_ids:
        image_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        # 复制图片
        shutil.copy(image_path, output_path)

        final_ids.append(str(frame_id))

    # 保存 frame_id 到 txt
    with open(keyframes_txt, "w") as f:
        f.write("\n".join(final_ids))

    print(f"✅ 所有帧已保存至 {output_dir}，frame IDs 写入 {keyframes_txt}")


import glob
import shutil
import os

def copy_bbox_keyframes_by_ids(id_txt_path, bbox_dir, output_dir):
    """
    从 bbox_dir 中根据 keyframe_ids 复制 0000000_*.jpg 的所有匹配图片到 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取 txt 中保存的 frame_id
    with open(id_txt_path, "r") as f:
        ids = [int(line.strip()) for line in f if line.strip().isdigit()]

    count = 0
    for fid in ids:
        prefix = f"{fid * 25:07d}_"
        matches = glob.glob(os.path.join(bbox_dir, f"{prefix}*.jpg"))
        if not matches:
            print(f"⚠️ No images found for prefix {prefix} in {bbox_dir}")
            continue

        for path in matches:
            shutil.copy(path, os.path.join(output_dir, os.path.basename(path)))
            count += 1

    # 也拷贝 txt 本身
    shutil.copy(id_txt_path, os.path.join(output_dir, "keyframes.txt"))

    print(f"✅ Copied {count} bbox images and keyframes.txt to {output_dir}")


# 打开日志文件
with open(log_file_path, "a") as log_file:
    log_file.write(f"\n===== Batch Processing Started at {datetime.now()} =====\n")

    # 遍历所有视频
    for i in range(video_start, video_end + 1):
        video_name = f"video{i:02d}"
        input_dir = os.path.join(input_base_dir, video_name, "ws_0", "images")
        keyframes_json = os.path.join(input_base_dir, video_name, "ws_0", "prompts.json")
        output_dir = os.path.join(output_base_dir, video_name)
        keyframes_txt = os.path.join(output_dir, "keyframes.txt")
        bbox_dir = f"/mnt/disk0/haoding/cholec80/extracted_frames_bbox_clean/{video_name}"
        output_dir = f"/mnt/disk0/haoding/surgical-images-pertool-clean/{video_name}"

        # 检查 `images` 目录 & `prompts.json` 是否存在
        if not os.path.exists(input_dir) or not os.path.exists(keyframes_json):
            message = f"[{datetime.now()}] ❌ Skipping {video_name}: Missing images or prompts.json\n"
            print(message.strip())
            log_file.write(message)
            continue

        message = f"\n[{datetime.now()}] 🚀 Processing {video_name}...\n"
        print(message.strip())
        log_file.write(message)

        try:
            # 提取关键帧
            # extract_and_save_keyframes(input_dir, keyframes_json, output_dir, keyframes_txt)
            # extract_and_save_keyframes_expand_neighbor(input_dir, keyframes_json, output_dir, keyframes_txt)
            # extract_and_save_frames_all(input_dir, output_dir, keyframes_txt)

            extract_and_save_keyframes_ids(input_dir, keyframes_json, output_dir, keyframes_txt)
            copy_bbox_keyframes_by_ids(keyframes_txt, bbox_dir, output_dir)#根据 ID 匹配并复制 bbox 图像

            # 记录成功日志
            success_message = f"[{datetime.now()}] ✅ {video_name} processed successfully\n"
            print(success_message.strip())
            log_file.write(success_message)

        except Exception as e:
            # 记录失败日志
            error_message = f"[{datetime.now()}] ❌ Error processing {video_name}: {e}\n"
            print(error_message.strip())
            log_file.write(error_message)
            log_file.write(f"[{datetime.now()}] Skipping {video_name} and continuing to the next video\n\n")
