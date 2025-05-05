import os, json, time
import requests
from PIL import Image
from openai import OpenAI
from modules.prompt_builder import (
    build_pdf_prompt, build_text_prompt1, build_text_prompt2
)
# python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/generate_triple_prompt.py
# python /home/haoding/Wenzheng/surgical_prompt_pipeline/generate_triple_prompt.py

# ──────────────────────────────────────────────────────────────
# OpenAI API Key
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
ALLOWED_CATEGORIES = {
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "Gallbladder", "Specimen Bag"
}

# ──────────────────────────────────────────────────────────────
def strip_JSON(response: str) -> dict:
    """
    清理 GPT 生成的 JSON（triplet 模式专用）
    """
    result = response.strip("```").strip("json").strip()
    if not result:
        return {"triplets": [], "analysis": "No valid response"}

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e} | 响应: {result}")
        return {"triplets": [], "analysis": "Invalid JSON"}

def parse_label(response: str) -> str:
    """
    提取并规范化单类别输出（single 模式专用）
    """
    label = response.strip().strip('"').strip("'")
    return label

def extract_frame_info(image_url: str):
    """
    提取 frame_id 和 frame_file。
    支持两种命名：
    - 整帧图像：0000975.jpg
    - Bbox图像：0000975_Grasper.jpg 或 0000975_Gallbladder.jpg
    """
    frame_file = os.path.basename(image_url)
    prefix = os.path.splitext(frame_file)[0]  # 去掉 .jpg

    # 只取下划线前的数字部分
    frame_number_str = prefix.split("_")[0]

    try:
        frame_id = int(frame_number_str) // 25
    except ValueError:
        print(f"❌ 无法解析 frame_id: {frame_file}")
        frame_id = -1

    return frame_id, frame_file


# ──────────────────────────────────────────────────────────────
def process_video(
    video_number: int,
    input_base_dir: str,
    output_base_dir: str,
    mode: str = "triplet",          # "triplet" | "single"
    max_retries: int = 1
):
    assert mode in {"triplet", "single"}
    video_folder = f"video{video_number:02d}"
    input_txt  = os.path.join(input_base_dir,  video_folder, "image_urls.txt")
    output_json = os.path.join(output_base_dir, video_folder, "triplets.json")

    if not os.path.exists(input_txt):
        print(f"❌ {input_txt} 不存在，跳过 {video_folder}")
        return

    with open(input_txt) as f:
        image_urls = [l.strip() for l in f if l.strip()]
    if not image_urls:
        print(f"⚠️ {video_folder} 无 URL，跳过")
        return

    # -------- 构建基础对话（PDF + Text Prompt） --------
    base_messages = [
        {"role": "system", "content": "You are an advanced AI with expertise in laparoscopic cholecystectomy."},
        {"role": "user",   "content": build_pdf_prompt()}
    ]
    # 让 GPT 先“读论文”
    try:
        pdf_resp = client.chat.completions.create(
            model="gpt-4o", messages=base_messages, temperature=0, max_tokens=800
        )
        base_messages.append({"role": "assistant", "content": pdf_resp.choices[0].message.content.strip()})
    except Exception as e:
        print(f"⚠️ PDF 预热失败: {e}")

    # 加入对应文本 prompt
    text_prompt = build_text_prompt1() if mode == "triplet" else build_text_prompt2()
    base_messages.append({"role": "user", "content": text_prompt})
    try:
        txt_resp = client.chat.completions.create(
            model="gpt-4o", messages=base_messages, temperature=0, max_tokens=300
        )
        base_messages.append({"role": "assistant", "content": txt_resp.choices[0].message.content.strip()})
    except Exception as e:
        print(f"⚠️ 文本 prompt 预热失败: {e}")

    # -------- 逐帧处理 --------
    outputs = []
    for idx, url in enumerate(image_urls):
        frame_id, frame_file = extract_frame_info(url)
        print(f"🚀 {video_folder} [{idx+1}/{len(image_urls)}] {frame_file}")

        local_messages = base_messages + [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Frame {frame_file}"},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }]

        parsed_success = False
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o", messages=local_messages,
                    temperature=0, max_tokens=1000
                )
                gpt_out = resp.choices[0].message.content.strip()

                if mode == "triplet":
                    parsed = strip_JSON(gpt_out)
                    parsed_success = bool(parsed.get("triplets"))
                else:  # single
                    label = parse_label(gpt_out)
                    parsed_success = label in ALLOWED_CATEGORIES
                    parsed = {"label": label}

                if parsed_success:
                    break
                else:
                    print(f"⚠️ 第 {attempt} 次解析失败，重试…")
            except Exception as e:
                print(f"❌ 第 {attempt} 次请求错误: {e}")
            time.sleep(1)  # 退避

        if not parsed_success:
            parsed = parsed if mode == "single" else {"triplets": [], "analysis": ""}
            print(f"⚠️ Frame {frame_file} 连续 {max_retries} 次失败，记为未知")

        outputs.append({
            "frame_id": frame_id,
            "frame_file": frame_file,
            "image_url": url,
            **({"triplets": parsed.get("triplets", []), "analysis": parsed.get("analysis", "")}
               if mode == "triplet"
               else {"label": parsed.get("label", "Unknown")})
        })

    # 保存
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"✅ {video_folder} 完成 → {output_json}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ===== A：旧任务（整帧三元组） =====
    # input_base_dir = "/mnt/disk0/haoding/surgical-images/"
    # output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames_all"

    # ===== 例 B：新任务（BBox 单类别） =====
    inp = "/mnt/disk0/haoding/surgical-images-pertool"
    out = "/mnt/disk0/haoding/cholec80/extracted_frames_bbox_clean"
    for v in range(1, 21):
        process_video(v, inp, out, mode="single")


