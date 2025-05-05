import os
import shutil

#python home/haoding/DT_SPR_utils/SAM2/automate_GPT/move_triple.py
#python /home/haoding/Wenzheng/surgical_prompt_pipeline/utils/move_triple.py
src_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_all"
dst_base_dir = "/mnt/disk0/haoding/cholec80_dt/gpt_response_all"

#1-80 , 81-97 ,101-105
start_video = 1
end_video = 6

os.makedirs(dst_base_dir, exist_ok=True)

for video_num in range(start_video, end_video + 1):
    video_folder = f"video{video_num:02d}"  # 格式化成 video01, video02, ..., videoXX
    src_triplets_path = os.path.join(src_base_dir, video_folder, "triplets.json")
    dst_triplets_path = os.path.join(dst_base_dir, f"triplets_{video_num:02d}.json")

    if not os.path.exists(src_triplets_path):
        print(f"⚠️ {src_triplets_path} 不存在，创建一个空文件")
        with open(src_triplets_path, "w") as f:
            f.write("{}")  # 创建空 JSON

    if os.path.exists(dst_triplets_path):
        print(f"⏩ {dst_triplets_path} 已存在，跳过")
        continue

    # 复制并重命名
    shutil.move(src_triplets_path, dst_triplets_path)
    print(f"✅ 移动完成: {src_triplets_path} -> {dst_triplets_path}")

print("🎯 所有文件处理完毕！")
