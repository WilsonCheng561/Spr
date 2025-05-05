import os
import json
from typing import Dict
import numpy as np
import argparse

from transformers import AutoTokenizer, AutoModel
import torch

#处理 JSON 文件 triplets_{01..80}.json 中的 (frame_id -> text)，并用 bert-base-uncased 做嵌入，将结果写到npy

#

def generate_surgical_text_embeddings_input(json_dir):
    """
    读取手术视频关键帧的 GPT 生成 JSON 文件，提取手术阶段三元组并转换为文本描述。
    
    参数:
        json_dir (str): 存放所有 JSON 文件的文件夹路径，每个 JSON 代表一个视频。

    返回:
        dict: {video_id: {frame_id: text_description}}
    """

    video_text_data = {}  # 用于存储所有视频的文本数据

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue
        
        # 假设文件名格式 "triplets_01.json"
        # 提取 "01" 作为 video_id
        # 或者根据需要调整
        if json_file.startswith("triplets_"):
            video_id = json_file.replace("triplets_", "").replace(".json", "")
        else:
            # 如果文件不是这种格式，跳过
            continue

        json_path = os.path.join(json_dir, json_file)

        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            triplet_data = json.load(f)  # 加载 JSON 数据

        frame_text_dict = {}
        for entry in triplet_data:
            frame_id = entry["frame_id"]
            triplets = entry["triplets"]
            analysis = entry.get("analysis", "")  # 有些 JSON 可能没有 analysis 字段

            # 处理 `triplets` 字段
            if len(triplets) == 0:
                # 若三元组列表为空
                triplet_str = "No triplets"
            elif isinstance(triplets[0], list):
                # 处理多组三元组 case
                triplet_str = " and ".join([" - ".join(t) for t in triplets])
            else:
                # 处理单个三元组 case
                # e.g. ["Grasper","Gallbladder","Hold"]
                triplet_str = " - ".join(triplets)

            text_description = f"Surgical action: {triplet_str}. {analysis}"
            
            frame_text_dict[frame_id] = text_description

        # 存入全局字典
        video_text_data[video_id] = frame_text_dict

    return video_text_data  # {video_id: {frame_id: "文本"}}


def generate_full_frame_texts(json_dir: str, frames_root_dir: str) -> Dict[str, Dict[int, str]]:
    """
    读取 JSON 的 keyframes + 实际图片 frames，进行补齐。
    返回 {video_id: {frame_id: text}}，保证每一帧都有描述。
    
    参数:
        json_dir: JSON 路径
        frames_root_dir: 帧图片总路径（每个视频一个子文件夹）
    """
    video_text_data = {}

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json") or not json_file.startswith("triplets_"):
            continue

        video_id = json_file.replace("triplets_", "").replace(".json", "")
        json_path = os.path.join(json_dir, json_file)
        frames_dir = os.path.join(frames_root_dir, f"video{video_id}")  # 注意 video+数字

        if not os.path.isfile(json_path) or not os.path.isdir(frames_dir):
            print(f"⚠️ 跳过 {video_id}, 缺少 JSON 或帧目录")
            continue

        # 1. 读取 JSON 里的 keyframe 描述
        with open(json_path, "r", encoding="utf-8") as f:
            keyframes = json.load(f)
        
        keyframe_texts = {}  # {frame_id: text}
        for entry in keyframes:
            fid = entry["frame_id"]
            triplets = entry.get("triplets", [])
            analysis = entry.get("analysis", "")

            if len(triplets) == 0:
                triplet_str = "No triplets"
            elif isinstance(triplets[0], list):
                triplet_str = " and ".join([" - ".join(t) for t in triplets])
            else:
                triplet_str = " - ".join(triplets)

            text = f"Surgical action: {triplet_str}. {analysis}".strip()
            keyframe_texts[fid] = text

        # 2. 读取所有帧的 frame_id
        all_frames = os.listdir(frames_dir)
        all_frame_ids = []
        for fname in all_frames:
            if fname.endswith(".png") and fname[:-4].isdigit():
                all_frame_ids.append(int(fname[:-4]))
        all_frame_ids.sort()

        if not all_frame_ids:
            print(f"⚠️ No frames found for video {video_id}, skipped")
            continue

        # 3. 对每一帧，根据 keyframe 补齐文本
        sorted_key_ids = sorted(keyframe_texts.keys())  # keyframe id 列表（已排序）
        full_frame_texts = {}

        for fid in all_frame_ids:
            # 找最近的（小于等于 fid）的 keyframe
            candidates = [k for k in sorted_key_ids if k <= fid]
            if candidates:
                nearest_kf = candidates[-1]
            else:
                nearest_kf = sorted_key_ids[0]  # 比最小还小，用第一个 keyframe

            full_frame_texts[fid] = keyframe_texts[nearest_kf]

        video_text_data[video_id] = full_frame_texts

    return video_text_data


def embed_text(text, tokenizer, model, device="cpu"):
    """
    对给定文本用 BERT 模型做 embedding，并返回一个 numpy array。
    这里示例使用CLS向量（或你可改为average pooling).
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)  # last_hidden_state, pooler_output
        # 这里示例: 取[CLS]向量 (batch_size=1, seq_len, hidden=768)
        # last_hidden_state[:,0,:] => (1, hidden)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embedding_np = cls_embedding.squeeze(0).cpu().numpy()
        # print(f"Encoding shape: {embedding_np.shape}")  # Encoding shape: (768,)
    return embedding_np


def main():
    parser = argparse.ArgumentParser(description="对 GPT 生成的手术视频 JSON 做文本embedding并保存Numpy")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="存放 triplets_{videoID}.json 文件的路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存embedding的输出路径")
    parser.add_argument("--video_start", type=int, default=1,
                        help="起始视频编号")
    parser.add_argument("--video_end", type=int, default=80,
                        help="结束视频编号")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="HuggingFace的BERT模型名称，如 'bert-base-uncased'")
    parser.add_argument("--frames_dir", type=str, required=True, help="存放所有帧图像的根目录（如 frames_cutmargin）")
    args = parser.parse_args()

    # 1) 加载 BERT tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model)
    model.eval()

    # 若有GPU可用,可使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2) 调用上面函数, 获取 {video_id -> {frame_id -> text}}
    # all_text_data = generate_surgical_text_embeddings_input(args.json_dir)#这个只有keyframes的embedding
    all_text_data = generate_full_frame_texts(args.json_dir, args.frames_dir)#所有图像帧的的embedding
    # 形如:
    # {
    #   "01": {
    #       0: "Surgical action: Grasper - Gallbladder - Hold. The grasper is holding the gallbladder ...",
    #       10: "Surgical action: Clipper - Cystic Duct - Clipping. The clipper is being used..."
    #   },
    #   "02": {...},
    #   ...
    # }

    # 3) 只处理 [video_start, video_end] 范围
    for vid in range(args.video_start, args.video_end+1):
        video_id_str = f"{vid:02d}"  # "01","02",...
        if video_id_str not in all_text_data:
            print(f"⚠️ video_id={video_id_str} 不在 JSON 里,跳过")
            continue
        # 生成输出文件夹
        video_outdir = os.path.join(args.output_dir, video_id_str)
        os.makedirs(video_outdir, exist_ok=True)

        frame_texts = all_text_data[video_id_str]  # {frame_id -> text}
        for frame_id, text in frame_texts.items():
            # 做embedding
            embedding = embed_text(text, tokenizer, model, device=device)
            # 保存npy => {output_dir}/{videoID}/{frame_id}.npy
            out_path = os.path.join(video_outdir, f"{frame_id}.npy")
            np.save(out_path, embedding)
            print(f"✅ Saved embedding for video={video_id_str}, frame={frame_id} => {out_path}")


if __name__ == "__main__":
    main()
