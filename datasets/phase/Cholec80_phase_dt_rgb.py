import os
import cv2
import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import video_transforms, volume_transforms
from datasets.transforms.random_erasing import RandomErasing

def convert_masks_depth(mask_path, depth_path, frame_id):
    """
    读取并返回 (H,W,11): 其中 10 通道来自 mask 的 one-hot，1 通道来自 depth。
    假设 frame_id => target_frame_id = frame_id * 25，用于拼成 '0000025.png'。
    """
    frame_id = int(frame_id)
    target_frame_id = frame_id * 25
    depth_file_name = f"{target_frame_id:07d}.png"
    mask_file_name  = f"{target_frame_id:07d}.png"

    depth_file_path = os.path.join(depth_path, depth_file_name)
    if not os.path.exists(depth_file_path):
        raise FileNotFoundError(f"Depth file not found: {depth_file_path}")

    depth = np.array(Image.open(depth_file_path)).astype(np.float32)
    depth = depth / 65535.0  # 归一化到 [0,1]

    mask_file_path = os.path.join(mask_path, mask_file_name)
    if not os.path.exists(mask_file_path):
        raise FileNotFoundError(f"Mask file not found: {mask_file_path}")

    mask = np.array(Image.open(mask_file_path))
    if not np.all((mask >= 0) & (mask <= 10)):
        raise ValueError(f"Mask {mask_file_path} 存在不在 [0,10] 范围的类别")

    H, W = mask.shape
    tmp = np.zeros((H, W, 11), dtype=np.float32)
    tmp[np.arange(H)[:, None], np.arange(W), mask] = 1  # 0通道是背景 => 其后 10 通道
    mask_10 = tmp[:, :, 1:]  # 去掉背景 => 只留 10 通道

    depth_channel = np.expand_dims(depth, axis=-1)
    final_11ch = np.concatenate((mask_10, depth_channel), axis=-1)  # (H,W,11)
    return final_11ch


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    针对 shape=(C,T,H,W) 的视频张量做随机缩放、裁剪、翻转等操作。
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                frames, min_scale, max_scale, inverse_uniform_sampling
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


class PhaseDataset_Cholec80_dt_rgb(Dataset):
    """
    同时读取：
      - 10mask+1depth => (H,W,11)
      - GSViT提取好的特征 => .npy shape=(feat_dim,)
    并将二者在最后一维拼接 => (H,W, 11+feat_dim)
    """

    def __init__(
        self,
        anno_path,           # 标注文件 (pickle)
        data_path,           # 指向 (masks, depths) 的根目录
        gsvit_feat_root,     # 存放 <video_id>_frameXXXXX.npy 的目录
        mode="train",
        data_strategy="online",  # online/offline
        output_mode="key_frame", # key_frame/all_frame
        cut_black=True,
        clip_len=16,
        frame_sample_rate=2,
        keep_aspect_ratio=True,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        args=None,
    ):
        super().__init__()
        self.anno_path         = anno_path
        self.dt_data_path      = data_path       # 10mask+1depth
        self.gsvit_feat_root   = gsvit_feat_root # GSViT特征保存路径
        self.mode              = mode
        self.data_strategy     = data_strategy
        self.output_mode       = output_mode
        self.cut_black         = cut_black
        self.clip_len          = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.keep_aspect_ratio = keep_aspect_ratio
        self.crop_size         = crop_size
        self.short_side_size   = short_side_size
        self.new_height        = new_height
        self.new_width         = new_width
        self.args              = args
        
        self.frame_span = self.clip_len * self.frame_sample_rate

        # 是否训练阶段
        self.aug = False
        self.rand_erase = False
        if self.mode == "train":
            self.aug = True
            if self.args and getattr(self.args, "reprob", 0) > 0:
                self.rand_erase = True

        # 载入标注
        with open(self.anno_path, "rb") as f:
            self.infos = pickle.load(f)

        # 构建列表 => dataset_samples
        self.dataset_samples = self._make_dataset(self.infos)

        # val/test 的 transforms，看是否需要自己写
        if self.mode == "val":
            # 这里不再做 volume_transforms.ClipToTensor(channel_nb=14) 之类，
            # 因为我们是 offline 处理 (H,W, 11+feat_dim) => (C,T,H,W) 在 _aug_frame()
            # or _valtest_normalize() 做
            pass
        elif self.mode == "test":
            pass

    def _make_dataset(self, infos):
        """
        infos: { video_id: [ {frame_id, video_id, frames, phase_gt, ...}, ... ], ... }
        """
        samples = []
        for video_id, data_list in infos.items():
            for line_info in data_list:
                # line_info 包含 frame_id, video_id, ...
                samples.append(line_info)
        return samples

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        info = self.dataset_samples[index]
        video_id = info["video_id"]
        frame_id = int(info["frame_id"])   # e.g. 1, 2, 3 ...
        duration = int(info["frames"])
        # 1) 根据 data_strategy
        if self.data_strategy == "online":
            buffer, phase_labels, sampled_list = self._video_batch_loader(duration, frame_id, video_id, index)
        else:
            buffer, phase_labels, sampled_list = self._video_batch_loader_for_key_frames(duration, frame_id, video_id, index)

        # 如果 train => 做空间采样 + augment
        if self.mode == "train":
            buffer = self._aug_frame(buffer)
        else:
            # val/test => 做简单 transforms (如需)
            buffer = self._valtest_transform(buffer)

        # 是否有重复帧
        flag = (len(sampled_list) != len(np.unique(sampled_list)))

        # 根据 output_mode
        if self.output_mode == "key_frame":
            if self.data_strategy == "offline":
                # 取 clip_len // 2
                label = phase_labels[self.clip_len // 2]
            else:
                label = phase_labels[-1]
            return buffer, label, f"{index}_{video_id}_{frame_id}", flag
        else:
            # all_frame
            return buffer, phase_labels, f"{index}_{video_id}_{frame_id}", flag


    def _video_batch_loader(self, duration, frame_id, video_id, index):
        offset_value = index - frame_id
        sr = self.frame_sample_rate

        frame_ids = []
        cur_fid = frame_id
        for i in range(self.clip_len):
            frame_ids.append(cur_fid)
            if sr == -1:
                sr = random.randint(1, 5)
            elif sr == 0:
                sr = 2**i
            elif sr == -2:
                sr = 1 if 2*i==0 else 2*i
            if cur_fid - sr >= 0:
                cur_fid -= sr

        sampled_list = sorted([x + offset_value for x in frame_ids])
        frames_data = []
        labels_data = []

        for fid in sampled_list:
            fid = int(fid)
            # 读取 10mask+1depth
            mask_dir  = os.path.join(self.dt_data_path, "masks", video_id)
            depth_dir = os.path.join(self.dt_data_path, "depths", video_id)
            # real_frame_id => e.g. 00123.png
            real_frame_id = int(self.dataset_samples[fid]["frame_id"])
            md_data = convert_masks_depth(mask_dir, depth_dir, real_frame_id)  # (H,W,11)

            # 读取 GSViT 特征 => shape=(feat_dim,)
            feat_path = os.path.join(self.gsvit_feat_root, f"{video_id}_frame{real_frame_id:05d}.npy")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"GSViT feat not found: {feat_path}")
            feat_vec = np.load(feat_path)  # shape = (feat_dim,)
            feat_dim = feat_vec.shape[0]

            # 将 feat_vec 广播到 (H,W,feat_dim)
            H, W = md_data.shape[:2]
            feat_map = np.tile(feat_vec, (H, W, 1))  # => (H,W,feat_dim)
            # 合并 => (H, W, 11+feat_dim)
            combined = np.concatenate((md_data, feat_map), axis=-1)

            frames_data.append(combined)
            labels_data.append(self.dataset_samples[fid]["phase_gt"])

        video_data = np.stack(frames_data, axis=0)  # => (T,H,W, 11+feat_dim)
        phase_data = np.array(labels_data)
        return video_data, phase_data, sampled_list

    def _video_batch_loader_for_key_frames(self, duration, frame_id, video_id, index):
        # offline 的类似逻辑
        # 只演示思路
        right_len = self.clip_len // 2
        left_len  = self.clip_len - right_len
        offset_value = index - frame_id

        # 右侧
        sr = self.frame_sample_rate
        cur_f = frame_id
        right_ids = []
        for i in range(right_len):
            right_ids.append(cur_f)
            if sr == -1:
                sr = random.randint(1, 5)
            elif sr == 0:
                sr = 2**i
            elif sr == -2:
                sr = 1 if 2*i==0 else 2*i
            if cur_f + sr <= duration:
                cur_f += sr

        # 左侧
        left_ids = []
        sr = self.frame_sample_rate
        cur_f = frame_id
        for j in range(left_len):
            left_ids = [cur_f] + left_ids
            if sr == -1:
                sr = random.randint(1,5)
            elif sr == 0:
                sr = 2**j
            elif sr == -2:
                sr = 1 if 2*j==0 else 2*j
            if cur_f - sr >=0:
                cur_f -= sr

        frame_id_list = left_ids + right_ids
        sampled_list = [x + offset_value for x in frame_id_list]

        frames_data = []
        labels_data = []

        for fid in sampled_list:
            fid = int(fid)

            mask_dir  = os.path.join(self.dt_data_path, "masks", video_id)
            depth_dir = os.path.join(self.dt_data_path, "depths", video_id)
            real_frame_id = int(self.dataset_samples[fid]["frame_id"])
            md_data = convert_masks_depth(mask_dir, depth_dir, real_frame_id)

            feat_path = os.path.join(self.gsvit_feat_root, f"{video_id}_frame{real_frame_id:05d}.npy")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"GSViT feat not found: {feat_path}")
            feat_vec = np.load(feat_path)
            feat_dim = feat_vec.shape[0]
            H, W = md_data.shape[:2]
            feat_map = np.tile(feat_vec, (H, W, 1))

            combined = np.concatenate((md_data, feat_map), axis=-1)
            frames_data.append(combined)
            labels_data.append(self.dataset_samples[fid]["phase_gt"])

        video_data = np.stack(frames_data, axis=0)  # => (T,H,W, 11+feat_dim)
        phase_data = np.array(labels_data)
        return video_data, phase_data, sampled_list

    def _aug_frame(self, buffer):
        """
        训练集处理:
         buffer: shape=(T,H,W, 11+feat_dim) in np.float32
         => (C,T,H,W) => spatial_sampling => split前N channels(??) => ...
        """
        # => (C,T,H,W)
        buffer = torch.from_numpy(buffer)  # => shape=(T,H,W, 11+dim)
        buffer = buffer.permute(3,0,1,2)   # => (11+dim, T, H, W)

        # 做随机裁剪
        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            aspect_ratio=[0.75,1.3333],
            scale=[0.7,1.0],
            motion_shift=False,
        )
        return buffer

    def _valtest_transform(self, buffer):
        """
        val/test 的简化处理；
        buffer: (T,H,W, 11+dim) => (C,T,H,W)
        """
        buffer = torch.from_numpy(buffer).permute(3,0,1,2)  # =>(11+dim, T,H,W)
        # 你可能还想做 resize / center-crop 之类
        return buffer
