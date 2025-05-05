import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ----------------------------------------------
# （A）基于Cholec80只加载RGB图像的dataset
# ----------------------------------------------
class Cholec80RGBDataset(Dataset):
    """
    仅加载 Cholec80 的 RGB 图像 (3 通道)，
    从 anno_path(train.pickle) 中读出 video_id, frame_id，
    拼出 /path/to/cholec80/frames/<video_id>/000XX.png，并在 __getitem__ 返回 (RGB_tensor, video_id, frame_id)。
    """

    def __init__(self, anno_path, data_root_rgb, transform=None):
        super().__init__()
        self.anno_path = anno_path
        self.data_root_rgb = data_root_rgb
        self.transform = transform

        # 载入标注
        with open(self.anno_path, "rb") as f:
            self.infos = pickle.load(f)

        # 构建列表 (video_id, frame_id, rgb_path)
        self.samples = self._make_dataset(self.infos)

    def _make_dataset(self, infos):
        dataset = []
        for video_id, data_list in infos.items():
            for line_info in data_list:
                frame_id = int(line_info["frame_id"])
                # 构造RGB图像路径
                # data_root_rgb/frames/<video_id>/<000XX>.png
                rgb_path = os.path.join(
                    self.data_root_rgb,
                    video_id,
                    f"{frame_id:05d}.png",
                )
                dataset.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "rgb_path": rgb_path
                })
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        rgb_path = info["rgb_path"]
        # 读PIL
        img = Image.open(rgb_path).convert("RGB")
        # 做transform
        if self.transform:
            img = self.transform(img)
        # 返回 (Tensor, video_id, frame_id)
        return img, info["video_id"], info["frame_id"]


# ----------------------------------------------
# （B） EfficientViT删去 Decoder，仅保留特征提取部分
# ----------------------------------------------
from EfficientViT.classification.model.build import EfficientViT_M5

class GSViTFeatureExtractor(nn.Module):
    """
    根据load gsvit 中的 EfficientViT，只保留特征提取部分
    （原本 forward(x)会 decoder，但只要evit输出）
    """
    def __init__(self):
        super().__init__()
        # 使用 EfficientViT_M5(pretrained='efficientvit_m5') 并去掉最后分类层
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # 移除分类头
        # 1)  self.evit 是一个nn.Module
        # 2)  其children()最后一层可能是 classifier
        # 让 self.evit = nn.Sequential(*list(self.evit.children())[:-1]) => 只保留 backbone
        self.evit = nn.Sequential(*list(self.evit.children())[:-1])

    def forward(self, x):
        """
        x: shape (B, 3, H, W), 归一化后图像
        output: shape (B, hidden_dim), 例如 B, 384 or B,768 ...
        """
        feat = self.evit(x)     # (B, C)
        return feat


# ----------------------------------------------
# （C） 冻结并加载预训练权重
# ----------------------------------------------
class FrozenGSViT(nn.Module):
    """
    加载预训练GSViT(只提取器部分)，并冻结其权重。
    ckpt_path 需要加载你的 .pkl 或 .pth
    """
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.gsvit = GSViTFeatureExtractor()
        # 如果有ckpt_path，需要 load_state_dict
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f"[FrozenGSViT] Loading from {ckpt_path} ...")
            state = torch.load(ckpt_path, map_location="cpu")
            # 注意：如果你的 ckpt 是通过 self.gsvit.state_dict() 直接保存的，可直接 load
            # 如果是别的字段(例如 "model"), 需要对 state 做一次筛选
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.gsvit.load_state_dict(state, strict=False)
        else:
            print(f"[FrozenGSViT] Warning: ckpt_path={ckpt_path} not found. Will use default weights from 'efficientvit_m5'")

        # 冻结
        for p in self.gsvit.parameters():
            p.requires_grad = False
        self.gsvit.eval()

    def forward(self, x):
        with torch.no_grad():
            feats = self.gsvit(x)  # shape (B, feat_dim)
        return feats


# ----------------------------------------------
# （D） 数据预处理 transform
# ----------------------------------------------
import torchvision.transforms as T

def get_rgb_transform():
    """
    GSViT 可能要求 224x224 + 常规 ImageNet mean/std
    """
    return T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        # 例如 flipping color channels, 参见 process_inputs()
        # 如果确实需要 swap R/B，请在 forward() 或 transform 里手动做
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


# ----------------------------------------------
# （E） 主函数：遍历Cholec80RGBDataset -> FrozenGSViT -> feats -> .npy
# ----------------------------------------------
def save_cholec80_gsvit_features(
    anno_path,
    data_root_rgb,
    ckpt_path,
    out_dir,
    batch_size=16,
    num_workers=4,
    device="cuda"
):
    os.makedirs(out_dir, exist_ok=True)

    dataset = Cholec80RGBDataset(
        anno_path=anno_path,
        data_root_rgb=data_root_rgb,
        transform=get_rgb_transform()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    model = FrozenGSViT(ckpt_path).to(device)
    model.eval()

    print("[Info] Start extracting features ... total", len(dataset), "samples.")
    with torch.no_grad():
        for idx, (imgs, video_ids, frame_ids) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)        # (B, feat_dim)
            feats_np = feats.cpu().numpy()  # => (B, feat_dim)

            # 每个 sample 对应: feats_np[i], video_ids[i], frame_ids[i]
            for i in range(len(video_ids)):
                v_id = video_ids[i]
                f_id = frame_ids[i]
                feat_i = feats_np[i]
                # 保存为 v_id_frameXXXXX.npy
                save_name = f"{v_id}_frame{f_id:05d}.npy"
                save_path = os.path.join(out_dir, save_name)
                np.save(save_path, feat_i)
            
            if (idx+1) % 10 == 0:
                print(f"[Info] Processed batch {idx+1}, total {(idx+1)*batch_size}/{len(dataset)} done.")


# ----------------------------------------------
#  (F) if __name__ == "__main__"...
# ----------------------------------------------
if __name__ == "__main__":
    anno_path   = "/mnt/disk0/haoding/cholec80_original/labels/train_20/1fpstrain.pickle"   # label
    data_root   = "/mnt/disk0/haoding/cholec80_original/frames"
    ckpt        = "/home/haoding/Wenzheng/Digital-Twin-based-Surgical-Phase-Recognition/GSViT/GSViT.pkl"     # gsvit
    out_dir     = "/mnt/disk0/haoding/cholec80_original/gsvit/train_20"              # 输出特征保存目录
    batch_size  = 16
    device      = "cuda:0"

    save_cholec80_gsvit_features(
        anno_path=anno_path,
        data_root_rgb=data_root,
        ckpt_path=ckpt,
        out_dir=out_dir,
        batch_size=batch_size,
        num_workers=4,
        device=device
    )
    print("Done extracting GSViT features!")
