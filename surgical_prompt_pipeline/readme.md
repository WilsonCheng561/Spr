# GPT 自动化处理流程（提取 Keyframe、挂载url、 生成 Triplets、Embedding）

本流程依赖的 Conda 环境为：`surg_phase_sam2`。脚本路径均为绝对路径，方便直接运行。核心功能包括：

- 从视频中提取关键帧
- 上传图片到 GitHub 并生成 URL
- 构建 GPT 输入的 Triplets
- 对 Triplets 文本进行语义嵌入（embedding）

---

## 1. `extract_keyframes.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/extract_keyframes.py
```

**功能说明**：
- 从 `/mnt/disk0/haoding/cholec80/annotated_data/videoxx/ws_0/prompts.json` 中提取关键帧 ID。
- 如果关键帧超过 64，则补充伪 ID。
- 将对应的帧图片保存到 `extracted_frames`，并记录关键帧 ID 到：
  ```
  /mnt/disk0/haoding/cholec80/extracted_frames/video00/keyframes.txt
  ```

**后续操作**（复制帧图像到本地github仓库，用于上传挂载 GitHub）：
```bash
rsync -av --ignore-existing /mnt/disk0/haoding/cholec80/extracted_frames/ /mnt/disk0/haoding/surgical-images/
```

---

## 2. `generate_url.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/generate_url.py
```

**功能说明**：
- 从 `keyframes.txt` 中读取keyframe索引，查找对应图像（包括伪keyframe），生成 GitHub 图片 URL。
- 将所有 URL 写入：
  ```
  /mnt/disk0/haoding/surgical-images/videoxx/image_urls.txt
  ```

**挂载图片到 GitHub**：
```bash
cd /mnt/disk0/haoding/surgical-images
git pull
git add .
git commit -m "update all videos"
git push
```

---

## 3. `generate_triple_prompt.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/generate_triple_prompt.py
```

**功能说明**：
- 使用上一步的 URL 生成 GPT 需要的 Triplets。
- 输出文件为：
  ```
  /mnt/disk0/haoding/cholec80/extracted_frames/.../triplets_xx.json
  ```

---

## 4. `move_triple.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/move_triple.py
```

**功能说明**：
- 将生成的 triplets 文件从 `src_base_dir` 移动到 `dst_base_dir`：
  ```python
  src_base_dir = "/mnt/disk0/haoding/cholec80/extracted_frames"
  dst_base_dir = "/mnt/disk0/haoding/cholec80_dt/gpt_response"
  ```

---

## 5. `bert_encoder.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/bert_encoder.py \
  --json_dir /mnt/disk0/haoding/cholec80_dt/gpt_response \
  --output_dir /mnt/disk0/haoding/cholec80_dt/bert_text_embeddings \
  --video_start 1 \
  --video_end 80 \
  --bert_model bert-base-uncased
```

**功能说明**：
- 对 GPT 生成的 Triplets 文本进行逐帧语义嵌入，输出 `.npy` 格式。

---

## 6. `sentence_trans_encoder.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/sentence_trans_encoder.py \
  --json_dir /mnt/disk0/haoding/cholec80_dt/gpt_response \
  --output_dir /mnt/disk0/haoding/cholec80_dt/st_text_embeddings \
  --video_start 1 \
  --video_end 80 \
  --st_model sentence-transformers/all-MiniLM-L6-v2
```

**功能说明**：
- 与 `bert_encoder.py` 并列，用 `SentenceTransformer` 模型做文本嵌入。
- 推荐用于精度与效率的平衡替代。

---

如需支持 LLaMA 编码器嵌入（llama_encoder.py)，可后续扩展，但目前因权限受限暂不支持。


## 7. `test.py`

**运行方式**：
```bash
python /home/haoding/DT_SPR_utils/SAM2/automate_GPT/test.py
```

**功能说明**：
- 对 GPT 生成的 Triplets 与ground truth 对比，结果保存到test.txt，triplets的异常保存到test_errors.txt

---