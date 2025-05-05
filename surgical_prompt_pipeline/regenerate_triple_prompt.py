import os
import json
import time
from openai import OpenAI
from modules.prompt_builder import build_pdf_prompt, build_text_prompt1

#python /home/haoding/Wenzheng/surgical_prompt_pipeline/regenerate_triple_prompt.py
# OpenAI API Key
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_BASE_DIR  = "/mnt/disk0/haoding/surgical-images"
OUTPUT_BASE_DIR = "/mnt/disk0/haoding/cholec80/extracted_frames_all"

# 需要检查和重跑的 video id 
# VIDEOS_TO_CHECK = [1, 2, 3]  
VIDEOS_TO_CHECK = [3]  

# 最大重试次数
MAX_RETRIES = 1

def strip_JSON(response):
    """清理 GPT 生成的 JSON 响应，确保格式正确"""
    result = response.strip("```").strip("json").strip()
    
    if not result:  # 处理空响应
        print("⚠️ GPT 返回空响应，跳过此帧")
        return {"triplets": [], "analysis": "No valid response"}

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}, 响应内容: {result}")
        return {"triplets": [], "analysis": "Invalid JSON format"}


def extract_frame_info(image_url):
    """从 URL 提取 `frame_file` 和 `frame_id`"""
    frame_file = os.path.basename(image_url)  # 提取文件名
    frame_id = int(os.path.splitext(frame_file)[0]) // 25  # 计算 `frame_id`
    return frame_id, frame_file


def find_empty_frames(triplets_list: list) -> list:
    """
    扫描 triplets_list（已加载的 JSON 数组），
    返回所有 triplets 为空或 analysis 标记异常的索引列表。
    """
    bad_idxs = []
    for idx, entry in enumerate(triplets_list):
        if not entry.get("triplets") or "Invalid" in entry.get("analysis", "") or "Failed" in entry.get("analysis", ""):
            bad_idxs.append(idx)
    return bad_idxs


def rerun_frame(video_number: int, base_messages: list, image_url: str) -> dict:
    """
    对单帧发起请求，最多重试 MAX_RETRIES 次，返回新的 {triplets, analysis} dict。
    """
    local_msgs = base_messages + [{
        "role": "user",
        "content": [
            {"type": "text",    "text": f"Frame {image_url} in video {video_number:02d}, please analyze."},
            {"type": "image_url","image_url": {"url": image_url}}
        ]
    }]
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=local_msgs,
                temperature=0.0,
                max_tokens=1000
            )
            parsed = strip_JSON(resp.choices[0].message.content.strip())
            if parsed.get("triplets"):
                return parsed
            else:
                print(f"    ⚠️ Attempt {attempt}: empty triplets, retrying...")
        except Exception as e:
            print(f"    ❌ Attempt {attempt} error: {e}, retrying...")
        time.sleep(1)
    # 重试失败
    return {"triplets": [], "analysis": f"Failed after {MAX_RETRIES} retries"}


def main():
    for vid in VIDEOS_TO_CHECK:
        folder = f"video{vid:02d}"
        json_path = os.path.join(OUTPUT_BASE_DIR, folder, "triplets.json")
        if not os.path.exists(json_path):
            print(f"❌ {json_path} 不存在，跳过 video{vid:02d}")
            continue

        print(f"\n🔎 Checking video{vid:02d} ...")
        # 加载已有结果
        with open(json_path, "r") as f:
            triplets_list = json.load(f)

        bad_idxs = find_empty_frames(triplets_list)
        if not bad_idxs:
            print(f"✅ video{vid:02d} 没有空帧，跳过")
            continue
        print(f"⚠️ 找到 {len(bad_idxs)} 个空帧，索引: {bad_idxs}")

        # 重建 base_messages（PDF + text prompt），与原 process_video 保持一致
        pdf_prompt  = build_pdf_prompt()
        text_prompt = build_text_prompt1()
        base_messages = [
            {"role": "system", "content": "You are an advanced AI with knowledge in laparoscopic surgery research."},
            {"role": "user",   "content": pdf_prompt}
        ]
        # 先跑一次 PDF，再跑一次 text
        try:
            r1 = client.chat.completions.create(model="gpt-4o", messages=base_messages, temperature=0.0, max_tokens=800)
            base_messages.append({"role":"assistant","content":r1.choices[0].message.content})
        except: pass
        base_messages.append({"role":"user","content": text_prompt})
        try:
            r2 = client.chat.completions.create(model="gpt-4o", messages=base_messages, temperature=0.0, max_tokens=300)
            base_messages.append({"role":"assistant","content":r2.choices[0].message.content})
        except: pass

        # 针对每个空帧单独重跑并替换
        for idx in bad_idxs:
            entry = triplets_list[idx]
            print(f"  🔄 Re-running frame_idx={idx}, file={entry['frame_file']}")
            new_parsed = rerun_frame(vid, base_messages, entry["image_url"])
            # 更新字段
            entry["triplets"] = new_parsed.get("triplets", [])
            entry["analysis"] = new_parsed.get("analysis", "")
            triplets_list[idx] = entry

        # 写回 JSON
        with open(json_path, "w") as f:
            json.dump(triplets_list, f, indent=2)
        print(f"✅ Finished video{vid:02d}, updated {len(bad_idxs)} frames.")

if __name__ == "__main__":
    main()
