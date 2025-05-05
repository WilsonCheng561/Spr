import os
import json
import copy
from typing import List, Dict
from modules.prompt_builder import build_pdf_prompt, build_text_prompt
from modules.multi_agents import SurgicalAgent
from modules.utils import extract_frame_info, strip_JSON, OpenAIClient

#python /home/haoding/Wenzheng/surgical_prompt_pipeline/main3.py

def read_url_list(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_id_list(path: str) -> List[int]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def build_single_frame_msg(video_no: int, global_idx: int, url: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"[Video {video_no:02d} | Seq {global_idx}] "
                    "请分析这张腹腔镜图像，只返回 JSON："
                    '{"triplets":[...], "analysis":"..."}'
                )
            },
            {"type": "image_url", "image_url": {"url": url}}
        ]
    }


def process_video(video_number: int, input_base_dir: str, output_base_dir: str):
    video_folder = f"video{video_number:02d}"
    path_urls = os.path.join(input_base_dir, video_folder, "image_urls.txt")
    path_all_ids = os.path.join(input_base_dir, video_folder, "keyframes_all.txt")
    path_center_ids = os.path.join(input_base_dir, video_folder, "keyframes.txt")
    output_jsn = os.path.join(output_base_dir, video_folder, "triplets.json")

    image_urls = read_url_list(path_urls)
    all_ids = set(read_id_list(path_all_ids))
    center_ids = set(read_id_list(path_center_ids))

    if not image_urls or not all_ids or not center_ids:
        print(f"[❌] {video_folder} 缺少输入，跳过")
        return

    # 构建 frame_id → (url, idx) 映射
    frameid2url: Dict[int, str] = {}
    frameid2idx: Dict[int, int] = {}
    for i, url in enumerate(image_urls):
        try:
            fid, _ = extract_frame_info(url)
            if fid in all_ids:  # 只处理在 keyframes_all 中的帧
                frameid2url[fid] = url
                frameid2idx[fid] = i
        except:
            continue

    # base prompt
    base_messages = [
        {"role": "system", "content": "You are an advanced AI with laparoscopic cholecystectomy expertise."},
        {"role": "user", "content": build_pdf_prompt()},
        {"role": "assistant", "content": "PDF 学习完毕，已加载解剖与流程知识。"},
        {"role": "user", "content": build_text_prompt()},
        {"role": "assistant", "content": "文本提示已阅读，准备开始逐帧分析。"},
    ]

    surg_agent = SurgicalAgent(OpenAIClient(model="gpt-4o"))
    triplets_data = []

    for k_idx, fid in enumerate(sorted(all_ids)):
        if fid not in frameid2url:
            print(f"⚠️ frame_id {fid} 无法在 URL 列表中找到，跳过")
            continue

        url = frameid2url[fid]
        idx = frameid2idx[fid]
        messages = copy.deepcopy(base_messages)
        messages.append(build_single_frame_msg(video_number, idx, url))

        try:
            raw_json = surg_agent.analyze_frame(messages)
            parsed = strip_JSON(raw_json)
        except Exception as e:
            print(f"[Error] Video {video_number:02d} | frame_id={fid} 推理失败: {e}")
            parsed = {"triplets": [], "analysis": f"Error: {e}"}

        if fid in center_ids:
            frame_id, frame_file = extract_frame_info(url)
            triplets_data.append({
                "frame_id": frame_id,
                "frame_file": frame_file,
                "image_url": url,
                "triplets": parsed.get("triplets", []),
                "analysis": parsed.get("analysis", "")
            })

        if k_idx % 20 == 0:
            print(f"[✅] Video {video_number:02d} | 已处理帧 {k_idx+1}/{len(all_ids)}")

    os.makedirs(os.path.dirname(output_jsn), exist_ok=True)
    with open(output_jsn, "w", encoding="utf-8") as fp:
        json.dump(triplets_data, fp, indent=2, ensure_ascii=False)

    print(f"✅ {video_folder} 完成，写入中心帧 triplets: {len(triplets_data)}")


def main():
    input_base_dir = "/mnt/disk0/haoding/surgical-images/"
    output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_20_5/"
    for vid in range(1, 6):
        process_video(vid, input_base_dir, output_base_dir)


if __name__ == "__main__":
    main()
