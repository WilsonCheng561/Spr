import os
import json
import copy

from modules.prompt_builder import build_pdf_prompt, build_text_prompt, build_panel_prompt
from modules.multi_agents import SurgicalAgent
from modules.panel_discussion import PanelDiscussion
from modules.memory_agent import MemoryAgent
from modules.utils import extract_frame_info, strip_JSON, OpenAIClient

#python /home/haoding/Wenzheng/surgical_prompt_pipeline/main2.py

def process_video(video_number, input_base_dir, output_base_dir):
    video_folder = f"video{video_number:02d}"
    input_txt = os.path.join(input_base_dir, video_folder, "image_urls.txt")
    output_json = os.path.join(output_base_dir, video_folder, "triplets.json")

    if not os.path.exists(input_txt):
        print(f"❌ {input_txt} 不存在，跳过 {video_folder}")
        return

    with open(input_txt,"r") as f:
        image_urls = [line.strip() for line in f if line.strip()]
    if not image_urls:
        print(f"⚠️ {video_folder} 没有可处理的图片 URL，跳过")
        return
    
    # ========== Create base_messages, one PDF prompt + one text prompt before each video==========
    base_messages = []
    pdf_prompt = build_pdf_prompt()
    text_prompt = build_text_prompt()
    FSM_PROMPT = build_panel_prompt()
    
    base_messages_for_judge = [
        {"role": "system", "content": "你是腹腔镜胆囊切除手术的专家，很擅长 surgical consistency check。"},
        {"role": "user", "content": FSM_PROMPT}
    ]
    
    # ========== Initialize openai client, agent, panel, memory =============
    client = OpenAIClient(model="gpt-4o")
    client.list_valid_models()

    openai_client1 = OpenAIClient(model="gpt-4o")
    judge_client   = OpenAIClient(model="o3-mini")              
    surg_agent     = SurgicalAgent(openai_client1)
    panel          = PanelDiscussion(judge_client, openai_client1, FSM_PROMPT, max_retries=3, base_messages=base_messages_for_judge)
    memory         = MemoryAgent(panel, surg_agent, max_history=3, window_size=5)


    # 1) system + pdf
    base_messages.append({"role":"system","content":"You are an advanced AI with laparoscopic surgery knowledge."})
    base_messages.append({"role":"user","content": pdf_prompt})
    try:
        pdf_response = surg_agent.chat_completions(base_messages)
        base_messages.append({"role":"assistant","content":pdf_response})
        print(f"📖 Video {video_number:02d}: 已做PDF学习\n{pdf_response}\n")
    except Exception as e:
        print(f"⚠️ PDF学习时出现错误: {e}, 继续处理后续...")

    # 2) text prompt
    base_messages.append({"role":"user","content":text_prompt})
    try:
        text_answer = surg_agent.chat_completions(base_messages)
        base_messages.append({"role":"assistant","content":text_answer})
        print(f"💬 Video {video_number:02d}: 已添加文本prompt\n{text_answer}\n")
    except Exception as e:
        print(f"⚠️ 文本Prompt时出现错误: {e}, 继续处理后续...")

    # ===================== 处理video的每一帧 =======================
    triplets_data: list[dict] = []
    frame_idx_map: dict[int, int] = {}   # frame_id -> triplets_data 索引

    for idx, image_url in enumerate(image_urls):
        print(f"🚀 处理 {video_folder} - {idx+1}/{len(image_urls)}: {image_url}")

        frame_id, frame_file = extract_frame_info(image_url)

        #很关键啊，base是每个视频的，local是每一帧的，这样帧之间不会相互影响
        local_messages = copy.deepcopy(base_messages)
        near_history_text = memory.get_near_history_summary()
        if near_history_text:
            local_messages.append({
                "role":"system",
                "content": f"<MEMORY>\n{near_history_text}\n</MEMORY>"
            })

        #  可加 <SUMMARY><CAPTION><REASONING><CONCLUSION> cot
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[Frame {frame_file} in video {video_number:02d}] Please analyze."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
        local_messages.append(user_msg)

        # 调用agent
        try:
            gpt_json  = surg_agent.analyze_frame(local_messages)
            parsed = strip_JSON(gpt_json)
        except Exception as e:
            print(f"❌ OpenAI请求错误: {e}, 跳过 {image_url}")
            parsed = {"triplets": [], "analysis": "Error"}

         # 占位写入
        triplets_data.append({
            "frame_id": frame_id,
            "frame_file": frame_file,
            "image_url": image_url,
            "triplets": parsed.get("triplets", []),
            "analysis": parsed.get("analysis", ""),
            "panel_decision": "Unchecked"
        })
        frame_idx_map[frame_id] = len(triplets_data) - 1

        # --- 交给 MemoryAgent 进入窗口 ---
        memory.add_parsed(frame_id, parsed, local_messages)

        # --- 判断 flush ---
        last = (idx == len(image_urls) - 1)
        refined_pairs = memory.flush_if_needed(last_frame=last)  # [(fid, refined), ...]

        # 写回修正
        for fid, refined in refined_pairs:
            i = frame_idx_map[fid]
            triplets_data[i]["triplets"]       = refined["triplets"]
            triplets_data[i]["analysis"]       = refined["analysis"]
            triplets_data[i]["panel_decision"] = refined["panel_decision"]

    # 保存JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(triplets_data, f, indent=2, ensure_ascii=False)

    print(f"✅ {video_folder} 处理完成，结果保存在 {output_json}")

def main():
    input_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames"
    output_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames"

    for video_number in range(1, 21):
        process_video(video_number, input_base_dir, output_base_dir)

if __name__ == "__main__":
    main()
