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
    根据 frame_id 读取 mask(10通道) + depth(1通道)，合并为 shape=(H,W,11)。
    假设 frame_id => target_frame_id = frame_id*25 用于拼接 '0000025.png' 等。
    """
    frame_id = int(frame_id)
    target_frame_id = frame_id * 25
    depth_file_name = f"{target_frame_id:07d}.png"
    mask_file_name  = f"{target_frame_id:07d}.png"
    
    # depth_file_name = f"{frame_id:05d}.png"
    # mask_file = f"{frame_id:05d}.png"

    depth_file_path = os.path.join(depth_path, depth_file_name)
    if not os.path.exists(depth_file_path):
        raise FileNotFoundError(f"Depth file not found: {depth_file_path}")

    depth = np.array(Image.open(depth_file_path)).astype(np.float32)
    depth = depth / 65535.0  # 归一化到[0,1]

    mask_path_ = os.path.join(mask_path, mask_file_name)
    if not os.path.exists(mask_path_):
        raise FileNotFoundError(f"Mask file not found: {mask_path_}")

    mask = np.array(Image.open(mask_path_))
    if not np.all((mask >= 0) & (mask <= 10)):
        raise ValueError(f"Mask {mask_path_} 存在不在 [0,10] 范围的类别")

    H, W = mask.shape
    tmp = np.zeros((H, W, 11), dtype=np.float32)
    tmp[np.arange(H)[:, None], np.arange(W), mask] = 1
    # 前 1 通道是背景0类，这里去掉 => 10通道
    mask_10 = tmp[:, :, 1:]

    # depth 作为第 11 通道
    depth_channel = np.expand_dims(depth, axis=-1)
    final_11ch = np.concatenate((mask_10, depth_channel), axis=-1)
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
    对 (C,T,H,W) 的视频张量进行随机缩放、裁剪、翻转等空间增强。
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
    同时读取 3通道RGB (data_path_rgb) 与 10mask+1depth (data_path) => 14通道输入
    """

    def __init__(
        self,
        anno_path,          # 标注文件
        data_path,          # 10mask+1depth 的根目录
        data_path_rgb,      # 3通道 RGB 的根目录
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
        self.rgb_data_path     = data_path_rgb   # 3通道原图
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

        self.dataset_samples = self._make_dataset(self.infos)

        # =============== 修改 val/test 变为 14 通道 ===============
        if self.mode == "val":
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize((self.short_side_size, self.short_side_size), interpolation="bilinear"),
                # 原先: volume_transforms.ClipToTensor() => 3 通道
                volume_transforms.ClipToTensor(channel_nb=14),  # 改成14通道
            ])
        elif self.mode == "test":
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize((self.short_side_size, self.short_side_size), interpolation="bilinear"),
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(channel_nb=14),  # 同理14通道
            ])
        # =============== 训练阶段如果你也是14通道，也可这样改 ===============
        elif self.mode == "train":
            # 演示: 简单的 resize + ClipToTensor(14)，
            # 如果有更多 augment，可自己写在 _aug_frame
            
            # self.data_transform = video_transforms.Compose([
            #     video_transforms.Resize((self.short_side_size, self.short_side_size), interpolation="bilinear"),
            #     volume_transforms.ClipToTensor(channel_nb=14),
            # ])
            pass
            
        # ==========================================================
          
    
    def _make_dataset(self, infos):
        """
        处理标注文件，构建 self.dataset_samples，每项包含 {
            'video_id', 'frame_id', 'frames', ..., 'img_path'
        } 等字段
        """
        frames = []
        for video_id, data_list in infos.items():
            for line_info in data_list:
                if len(line_info) < 8:
                    raise RuntimeError(f"Video input format not correct: {line_info}")
                # 拼出对应的 RGB frames 路径
                rgb_path = os.path.join(
                    self.rgb_data_path,
                    "frames",
                    line_info["video_id"],
                    f"{int(line_info['frame_id']):05d}.png",
                )
                line_info["img_path"] = rgb_path
                frames.append(line_info)
        return frames
    

    def __len__(self):
        return len(self.dataset_samples)

    
    def __getitem__(self, index):
        info = self.dataset_samples[index]
        video_id = info["video_id"]
        frame_id = info["frame_id"]
        frames_count = info["frames"]

        # 根据 data_strategy
        if self.data_strategy == "online":
            buffer, phase_labels, sampled_list = self._video_batch_loader(
                frames_count, frame_id, video_id, index
            )
        else:
            buffer, phase_labels, sampled_list = self._video_batch_loader_for_key_frames(
                frames_count, frame_id, video_id, index
            )

        # 训练 / 验证 / 测试 三种模式下做的处理略有不同
        if self.mode == "train":
            # 在 _aug_frame 里进行 spatial_sampling + 通道分离 & 归一化
            buffer = self._aug_frame(buffer)   # =>(14,T,H,W) tensor
        else:
            # val 或 test
            if self.mode == "val":
                buffer = self.data_transform(buffer)  # => shape=(14,T,H,W)
                # 这里若想对RGB做归一化 => 分开前3 vs 后11
                buffer = self._valtest_rgb_normalize(buffer)
            elif self.mode == "test":
                buffer = self.data_resize(buffer)      # => list of (H,W,14) or np.array
                if isinstance(buffer, list):
                    buffer = np.stack(buffer, axis=0)  # =>(T,H,W,14)
                buffer = self.data_transform(buffer)   # =>(14,T,H,W)
                buffer = self._valtest_rgb_normalize(buffer)

        # 是否有重复帧
        flag = (len(sampled_list) != len(np.unique(sampled_list)))

        if self.output_mode == "key_frame":
            if self.data_strategy == "offline":
                return buffer, phase_labels[self.clip_len // 2], f"{index}_{video_id}_{frame_id}", flag
            else:
                return buffer, phase_labels[-1], f"{index}_{video_id}_{frame_id}", flag
        else:
            return buffer, phase_labels, f"{index}_{video_id}_{frame_id}", flag
        

    def _video_batch_loader(self, duration, cur_frame_id, video_id, index):
        """
        'online' 的时序采样: 从当前索引倒序取 self.clip_len 帧。
        同时在每帧中分别读 3通道RGB + 11通道(mask+depth) => (H,W,14)
        """
        duration    = int(duration)
        cur_frame_id= int(cur_frame_id)
        index       = int(index)

        offset_value = index - cur_frame_id
        sr = self.frame_sample_rate

        frame_ids = []
        # 从 cur_frame_id 往前找 self.clip_len 帧
        for i in range(self.clip_len):
            frame_ids.append(cur_frame_id)
            if sr == -1:
                sr = random.randint(1, 5)
            elif sr == 0:
                sr = 2**i
            elif sr == -2:
                sr = 1 if 2*i == 0 else 2*i
            if cur_frame_id - sr >= 0:
                cur_frame_id -= sr

        sampled_list = sorted([fid + offset_value for fid in frame_ids])
        frames_data  = []
        labels_data  = []

        for fid in sampled_list:
            fid = int(fid)
            try:
                rgb_path = self.dataset_samples[fid]["img_path"]
                rgb_img  = Image.open(rgb_path).convert("RGB")  # =>(H,W,3)
                rgb_img  = np.array(rgb_img, dtype=np.uint8)
                real_frame_id = int(os.path.basename(rgb_path).split(".")[0])
                
                # ========== 这里把 RGB 3通道全赋值为 0 ==========
                # shape (H,W,3)
                rgb_img[...] = 0.0 
                
                
                
                # mask+depth
                mask_dir  = os.path.join(self.dt_data_path, "masks", video_id)
                depth_dir = os.path.join(self.dt_data_path, "depths", video_id)
                md_data   = convert_masks_depth(mask_dir, depth_dir, real_frame_id)

                # 对齐尺寸
                if rgb_img.shape[:2] != md_data.shape[:2]:
                    rgb_img = cv2.resize(rgb_img, (md_data.shape[1], md_data.shape[0]))

                # 合并 => (H,W,14)
                rgb_img = rgb_img.astype(np.float32)
                combined= np.concatenate((rgb_img, md_data), axis=-1)

                frames_data.append(combined)
                labels_data.append(self.dataset_samples[fid]["phase_gt"])
            except:
                raise RuntimeError(
                    f"Error reading frame index {fid} from video={video_id}, path={self.dataset_samples[fid]['img_path']}"
                )

        video_data = np.stack(frames_data, axis=0)  # => (T,H,W,14)
        phase_data = np.stack(labels_data)
        return video_data, phase_data, sampled_list

    def _video_batch_loader_for_key_frames(self, duration, timestamp, video_id, index):
        """
        'offline' 的采样: 向前 / 向后各取 (clip_len/2) 帧，并合并关键帧 => self.clip_len
        """
        duration = int(duration)
        timestamp= int(timestamp)
        index    = int(index)

        right_len = self.clip_len // 2
        left_len  = self.clip_len - right_len
        offset_value = index - timestamp

        # 右侧
        cur_t = timestamp
        right_frames = []
        for i in range(right_len):
            right_frames.append(cur_t)
            sr = self.frame_sample_rate
            if sr == -1:
                sr = random.randint(1, 5)
            elif sr == 0:
                sr = 2**i
            elif sr == -2:
                sr = 1 if 2*i == 0 else 2*i
            if cur_t + sr <= duration:
                cur_t += sr

        # 左侧
        cur_t = timestamp
        left_frames = []
        for j in range(left_len):
            left_frames = [cur_t] + left_frames
            sr = self.frame_sample_rate
            if sr == -1:
                sr = random.randint(1, 5)
            elif sr == 0:
                sr = 2**j
            elif sr == -2:
                sr = 1 if 2*j == 0 else 2*j
            if cur_t - sr >= 0:
                cur_t -= sr

        frame_id_list = left_frames + right_frames
        assert len(frame_id_list) == self.clip_len

        sampled_list = [fid + offset_value for fid in frame_id_list]
        frames_data  = []
        labels_data  = []

        for fid in sampled_list:
            fid = int(fid)
            try:
                rgb_path = self.dataset_samples[fid]["img_path"]
                rgb_img  = Image.open(rgb_path).convert("RGB")
                rgb_img  = np.array(rgb_img, dtype=np.float32)
                real_frame_id = int(os.path.basename(rgb_path).split(".")[0])
                
                # ========== 把RGB赋值0 ==========
                rgb_img[...] = 0.0
                
                

                mask_dir  = os.path.join(self.dt_data_path, "masks", video_id)
                depth_dir = os.path.join(self.dt_data_path, "depths", video_id)
                md_data   = convert_masks_depth(mask_dir, depth_dir, real_frame_id)

                if rgb_img.shape[:2] != md_data.shape[:2]:
                    rgb_img = cv2.resize(rgb_img, (md_data.shape[1], md_data.shape[0]))

                combined = np.concatenate((rgb_img, md_data), axis=-1)
                frames_data.append(combined)
                labels_data.append(self.dataset_samples[fid]["phase_gt"])
            except:
                raise RuntimeError(
                    f"Error reading frame index {fid} from video={video_id}, path={self.dataset_samples[fid]['img_path']}"
                )

        video_data = np.stack(frames_data, axis=0)  # =>(T,H,W,14)
        phase_data = np.stack(labels_data)
        return video_data, phase_data, sampled_list

    def _aug_frame(self, buffer):
        """
        针对训练集(14通道)做数据增强 + 归一化。
        buffer shape = (T,H,W,14) in numpy
        """
        # =>(C,T,H,W)
        buffer = torch.from_numpy(buffer).permute(3,0,1,2)  # (14,T,H,W)

        # 做随机裁剪、缩放、flip等
        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            aspect_ratio=[0.75, 1.3333],
            scale=[0.7, 1.0],
            motion_shift=False,
        )
        # buffer 仍然是 (14, T, H, W)

        # =========== 重点：分离前3通道(RGB) & 后11通道( mask + depth ) ===============
        rgb = buffer[:3]   # shape (3, T, H, W)
        md  = buffer[3:]   # shape (11, T, H, W)

        # 对RGB做: /255.0 => 减 mean => 除 std
        rgb = rgb / 255.0
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb.dtype, device=rgb.device).view(3,1,1,1)
        rgb_std  = torch.tensor([0.229, 0.224, 0.225], dtype=rgb.dtype, device=rgb.device).view(3,1,1,1)
        rgb = (rgb - rgb_mean) / rgb_std

        # 对 mask+depth => 不做任何归一化
        #   - mask 原本0/1的one-hot, depth是[0,1],都无需再处理

        # 拼回
        buffer = torch.cat((rgb, md), dim=0)  # => (14, T, H, W)

        # 如果需要 random erasing 只对RGB 或同样对整14通道
        # if self.rand_erase:
        #     buffer = buffer.permute(1,0,2,3)  # =>(T,14,H,W)
        #     re_op = RandomErasing(
        #         self.args.reprob, mode=self.args.remode,
        #         max_count=self.args.recount, num_splits=self.args.recount,
        #         device="cpu",
        #     )
        #     buffer = re_op(buffer)  # 这会对所有通道做erase, 可能不太适合mask
        #     buffer = buffer.permute(1,0,2,3)

        return buffer


    def _valtest_rgb_normalize(self, buffer):
            """
            对 val/test 的 14通道tensor, 只对前3通道做 /255 => 减均值/除方差
            buffer: shape=(14,T,H,W), torch.float
            """
            rgb = buffer[:3] / 255.0
            md  = buffer[3:]

            device_ = buffer.device
            rgb_mean = torch.tensor([0.485,0.456,0.406], device=device_).view(3,1,1,1)
            rgb_std  = torch.tensor([0.229,0.224,0.225], device=device_).view(3,1,1,1)
            rgb = (rgb - rgb_mean) / rgb_std

            return torch.cat([rgb, md], dim=0)