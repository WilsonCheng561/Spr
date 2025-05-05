import os
import json
import copy
import time
from json import JSONDecodeError
from typing import List, Dict, Tuple

from modules.prompt_builder import (
    build_pdf_prompt,
    build_text_prompt,
    build_panel_prompt2,
)
from modules.multi_agents import SurgicalAgent
from modules.panel_discussion import PanelDiscussion
from modules.memory_agent import MemoryAgent
from modules.utils import extract_frame_info, strip_JSON, OpenAIClient

#python /home/haoding/Wenzheng/surgical_prompt_pipeline/main4.py

def read_url_list(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def read_id_list(path: str) -> List[int]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def _safe_strip_json(raw: str) -> dict:
    data = strip_JSON(raw)
    if not isinstance(data, dict) or "triplets" not in data:
        raise ValueError("missing triplets")
    return data


def llm_call_with_retry(
    agent: SurgicalAgent,
    msgs: list,
    max_retry: int = 3,
    sleep_sec: int = 3,
) -> dict:
    last_err = None
    for i in range(max_retry):
        try:
            raw = agent.analyze_frame(msgs)
            parsed = _safe_strip_json(raw)
            if parsed.get("triplets"):
                return parsed
            raise ValueError("triplets empty")
        except (JSONDecodeError, ValueError, RuntimeError) as e:
            last_err = e
            if i < max_retry - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": "⚠️ JSON 解析失败或 triplets 为空，"
                        "请只返回有效 JSON（triplets 不得为空）！",
                    }
                )
                time.sleep(sleep_sec)

    print(f"❌ GPT 连续失败: {last_err}")
    return {"triplets": [], "analysis": f"Error: {last_err}"}


def build_user_msg(video_no: int, url: str, seq_idx: int) -> Dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"[Video {video_no:02d} | Seq {seq_idx}] "
                    "请分析该帧并仅返回 JSON："
                    '{"triplets":[...], "analysis":"..."}'
                ),
            },
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def process_video(video_no: int, base_in: str, base_out: str):
    vfolder = f"video{video_no:02d}"
    path_all_ids = os.path.join(base_in, vfolder, "keyframes_all.txt")
    path_key_ids = os.path.join(base_in, vfolder, "keyframes.txt")
    path_urls = os.path.join(base_in, vfolder, "image_urls.txt")
    output_jsn = os.path.join(base_out, vfolder, "triplets.json")

    image_urls = read_url_list(path_urls)
    all_ids = set(read_id_list(path_all_ids))
    key_ids = read_id_list(path_key_ids)

    if not image_urls or not key_ids:
        print(f"❌ {vfolder}: url 或 索引列表为空，跳过")
        return

    # 构建 frame_id → url / index 映射（只保留 keyframes_all 中存在的帧）
    frameid2url: Dict[int, str] = {}
    frameid2idx: Dict[int, int] = {}
    for i, url in enumerate(image_urls):
        try:
            fid, _ = extract_frame_info(url)
            if fid in all_ids:
                frameid2url[fid] = url
                frameid2idx[fid] = i
        except:
            continue

    base_msgs = [
        {
            "role": "system",
            "content": "You are an advanced AI with laparoscopic cholecystectomy expertise.",
        },
        {"role": "user", "content": build_pdf_prompt()},
        {"role": "assistant", "content": "PDF 学习完成，已加载解剖知识。"},
        {"role": "user", "content": build_text_prompt()},
        {"role": "assistant", "content": "文本提示已阅读，准备逐帧分析。"},
    ]

    FSM_PROMPT = build_panel_prompt2()
    judge_base = [
        {
            "role": "system",
            "content": "你是腹腔镜胆囊切除手术的专家，擅长 surgical consistency check。",
        },
        {"role": "user", "content": FSM_PROMPT},
    ]

    oa_main = OpenAIClient(model="gpt-4o")
    oa_judge = OpenAIClient(model="o3-mini")
    surg_agent = SurgicalAgent(oa_main)
    panel = PanelDiscussion(
        judge_client=oa_judge,
        openai_client=oa_main,
        fsm_prompt=FSM_PROMPT,
        max_retries=3,
        base_messages=judge_base,
    )
    memory = MemoryAgent(
        panel=panel,
        surg_agent=surg_agent,
        max_history=5,
        window_size=5,
    )

    triplets_out: List[dict] = []
    frame_idx_map: Dict[int, int] = {}

    for k_idx, center_id in enumerate(key_ids):
        win_ids = [fid for fid in range(center_id - 2, center_id + 3) if fid in frameid2url]
        if not win_ids:
            print(f"🚫 Video{video_no:02d} | Center {k_idx+1}/{len(key_ids)} | 窗口为空，跳过")
            continue

        print(
            f"🚀 Video{video_no:02d} | Center {k_idx+1}/{len(key_ids)} | 窗口 {win_ids}"
        )

        for fid in win_ids:
            url = frameid2url[fid]
            idx = frameid2idx[fid]

            msgs = copy.deepcopy(base_msgs)

            mem_txt = memory.get_near_history_summary()
            if mem_txt:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"<MEMORY>\n(以下为最近 {memory.window_size} 帧摘要)\n{mem_txt}\n</MEMORY>",
                    }
                )

            msgs.append(build_user_msg(video_no, url, seq_idx=idx))
            try:
                parsed = llm_call_with_retry(surg_agent, msgs, max_retry=3, sleep_sec=3)
            except Exception as e:
                print(f"❌ LLM 调用失败，跳过该帧: {e}")
                parsed = {"triplets": [], "analysis": f"Error: {e}"}


            memory.add_parsed(fid, parsed, msgs)

            if fid == center_id:
                frame_id, frame_file = extract_frame_info(url)
                triplets_out.append(
                    {
                        "frame_id": frame_id,
                        "frame_file": frame_file,
                        "image_url": url,
                        "triplets": parsed.get("triplets", []),
                        "analysis": parsed.get("analysis", ""),
                        "panel_decision": "Unchecked",
                    }
                )
                frame_idx_map[frame_id] = len(triplets_out) - 1

        # flush
        refined_pairs = memory.flush_if_needed(last_frame=False)
        for fid, refined in refined_pairs:
            if fid in frame_idx_map:
                i = frame_idx_map[fid]
                triplets_out[i]["triplets"] = refined["triplets"]
                triplets_out[i]["analysis"] = refined["analysis"]
                triplets_out[i]["panel_decision"] = refined["panel_decision"]

    # 最后一批 flush
    refined_pairs = memory.flush_if_needed(last_frame=True)
    for fid, refined in refined_pairs:
        if fid in frame_idx_map:
            i = frame_idx_map[fid]
            triplets_out[i]["triplets"] = refined["triplets"]
            triplets_out[i]["analysis"] = refined["analysis"]
            triplets_out[i]["panel_decision"] = refined["panel_decision"]

    os.makedirs(os.path.dirname(output_jsn), exist_ok=True)
    with open(output_jsn, "w", encoding="utf-8") as fp:
        json.dump(triplets_out, fp, indent=2, ensure_ascii=False)

    print(f"✅ {vfolder} 完成，中心帧={len(triplets_out)}，写入 {output_jsn}")


def main():
    base_in = "/mnt/disk0/haoding/surgical-images/"
    base_out = "/mnt/disk0/haoding/cholec80/extracted_frames_20_5_withMemory/"
    for v in range(2, 6):
        process_video(v, base_in, base_out)


if __name__ == "__main__":
    main()
