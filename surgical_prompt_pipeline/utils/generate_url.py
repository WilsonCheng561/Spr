#python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/generate_url.py
#python /home/haoding/Wenzheng/surgical_prompt_pipeline/utils/generate_url.py
import os
import re

#把/mnt/disk0/haoding/cholec80/extracted_frames/video00/keyframes.txt及其对应的所有伪关键帧图片（减少间隔），转换成url并保存 `image_urls.txt`
#然后就是挂载图片到github了
#cd /mnt/disk0/haoding/surgical-images
#git pull / git add ./git commit -m "video 1-5 every frame"/git push
#那你就可以去generate gpt triplet了



# GitHub Pages 主页 URL
# BASE_URL = "https://wilsoncheng561.github.io/surgical-images/"
BASE_URL = "https://wilsoncheng561.github.io/Surgical-img-pertool/"

# 本地 GitHub 目录
# IMAGE_DIR = "/mnt/disk0/haoding/surgical-images/"
# IMAGE_DIR = "/mnt/disk0/haoding/cholec80/extracted_frames"
# IMAGE_DIR = "/mnt/disk0/haoding/cholec80/extracted_frames_bbox"
IMAGE_DIR = "/mnt/disk0/haoding/surgical-images-pertool-clean"

# 正则匹配帧号，如 frame_001.jpg, img_12.png
FRAME_PATTERN = re.compile(r"(\d+)")

def extract_frame_number(filename):
    """从文件名提取帧序号，默认返回 0（防止异常情况）"""
    match = FRAME_PATTERN.search(filename)
    return int(match.group(1)) if match else 0

# 遍历 `videoXX` 目录
for video_folder in sorted(os.listdir(IMAGE_DIR)):
    video_path = os.path.join(IMAGE_DIR, video_folder)

    # 仅处理 `video` 目录
    if not os.path.isdir(video_path) or not video_folder.startswith("video"):
        continue

    # 存放当前视频的 URL
    image_urls = []
    
    # 遍历当前 `videoXX` 目录下的所有图片
    image_files = []
    for root, _, files in os.walk(video_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_files.append(file)

    # **按帧序号排序**
    image_files.sort(key=extract_frame_number)

    # 生成 URL
    for file in image_files:
        rel_path = os.path.relpath(os.path.join(video_path, file), IMAGE_DIR)
        image_urls.append(f"{BASE_URL}{rel_path}")

    # 保存 `image_urls.txt` 到当前 `videoXX` 目录
    output_txt = os.path.join(video_path, "image_urls.txt")
    with open(output_txt, "w") as f:
        f.write("\n".join(image_urls))

    print(f"✅ URLs for {video_folder} saved in {output_txt}")
