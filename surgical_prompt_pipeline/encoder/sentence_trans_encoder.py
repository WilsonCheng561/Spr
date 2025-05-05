import os
import json
import numpy as np
import argparse

import torch
from sentence_transformers import SentenceTransformer

#python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/sentence_trans_encoder.py --json_dir /mnt/disk0/haoding/cholec80_dt/gpt_response --output_dir /mnt/disk0/haoding/cholec80_dt/st_text_embeddings --video_start 1 --video_end 80 --st_model sentence-transformers/all-MiniLM-L6-v2


def generate_surgical_text_embeddings_input(json_dir):
    """
    读取手术视频关键帧的 GPT 生成 JSON 文件，提取手术阶段三元组并转换为文本描述。

    参数:
        json_dir (str): 存放所有 JSON 文件的文件夹路径，每个 JSON 代表一个视频。

    返回:
        dict: {video_id: {frame_id: text_description}}
    """

    video_text_data = {}

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        if json_file.startswith("triplets_"):
            video_id = json_file.replace("triplets_", "").replace(".json", "")
        else:
            continue

        json_path = os.path.join(json_dir, json_file)
        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            triplet_data = json.load(f)

        frame_text_dict = {}
        for entry in triplet_data:
            frame_id = entry["frame_id"]
            triplets = entry["triplets"]
            analysis = entry.get("analysis", "")

            if len(triplets) == 0:
                triplet_str = "No triplets"
            elif isinstance(triplets[0], list):
                triplet_str = " and ".join([" - ".join(t) for t in triplets])
            else:
                triplet_str = " - ".join(triplets)

            text_description = f"Surgical action: {triplet_str}. {analysis}"
            frame_text_dict[frame_id] = text_description

        video_text_data[video_id] = frame_text_dict

    return video_text_data  # {video_id: {frame_id: "文本"}}


def main():
    parser = argparse.ArgumentParser(description="对 GPT 生成的手术视频 JSON 做文本embedding并保存Numpy (Sentence-Transformers版)")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="存放 triplets_{videoID}.json 文件的路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存embedding的输出路径")
    parser.add_argument("--video_start", type=int, default=1,
                        help="起始视频编号")
    parser.add_argument("--video_end", type=int, default=80,
                        help="结束视频编号")
    parser.add_argument("--st_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence Transformers模型名称，如 'sentence-transformers/all-MiniLM-L6-v2'")
    args = parser.parse_args()

    # 1) 加载 Sentence-Transformers 模型
    print(f"🔍 Loading Sentence-Transformers model: {args.st_model}")
    model = SentenceTransformer(args.st_model)

    # 2) 调用函数, 获取 {video_id -> {frame_id -> text}}
    all_text_data = generate_surgical_text_embeddings_input(args.json_dir)

    # 3) 只处理 [video_start, video_end]
    for vid in range(args.video_start, args.video_end + 1):
        video_id_str = f"{vid:02d}"
        if video_id_str not in all_text_data:
            print(f"⚠️ video_id={video_id_str} 不在 JSON 里,跳过")
            continue

        # 输出文件夹
        video_outdir = os.path.join(args.output_dir, video_id_str)
        os.makedirs(video_outdir, exist_ok=True)

        frame_texts = all_text_data[video_id_str]  # {frame_id -> text}
        for frame_id, text in frame_texts.items():
            # **SentenceTransformer** 可以直接 batch encode，但这里逐帧处理也可以
            embedding = model.encode(text)  # -> numpy array
            # 保存npy => {output_dir}/{videoID}/{frame_id}.npy
            out_path = os.path.join(video_outdir, f"{frame_id}.npy")
            np.save(out_path, embedding)
            print(f"✅ Saved ST-embedding for video={video_id_str}, frame={frame_id} => {out_path}, shape={embedding.shape}")#shape=(384,)

if __name__ == "__main__":
    main()
