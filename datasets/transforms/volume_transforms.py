import numpy as np
from PIL import Image, ImageOps
import torch
import numbers
import random
import torchvision.transforms.functional as TF

def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


#这个是10mask+1depth+3RGB的14channel
# class ClipToTensor(object):
#     """Convert a list of m (H x W x C) numpy.ndarrays in range [0,255]
#        => torch.FloatTensor shape (C x m x H x W) in [0,1.0].
#     """
#     def __init__(self, channel_nb=14, div_255=True, numpy=False):
#         self.channel_nb = channel_nb
#         self.div_255 = div_255
#         self.numpy = numpy

#     def __call__(self, clip):
#         if isinstance(clip[0], np.ndarray):
#             h, w, ch = clip[0].shape
#             # 注释掉硬断言
#             if ch != self.channel_nb:
#                 print(f"[WARNING] ClipToTensor expecting {self.channel_nb} channels, got {ch}. Proceed anyway.")
#         elif isinstance(clip[0], Image.Image):
#             w, h = clip[0].size
#         else:
#             raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")

#         np_clip = np.zeros([self.channel_nb, len(clip), h, w], dtype=np.float32)
#         for img_idx, img in enumerate(clip):
#             if isinstance(img, Image.Image):
#                 img = np.array(img, copy=False)
#             elif not isinstance(img, np.ndarray):
#                 raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(img)}")

#             # shape=(H,W,C)
#             c_here = min(img.shape[2], self.channel_nb) if img.ndim == 3 else 1
#             # 先转 C,H,W
#             img = img.transpose(2,0,1) if img.ndim==3 else np.expand_dims(img,0)
#             np_clip[:c_here, img_idx, :, :] = img[:c_here]

#         if self.numpy:
#             if self.div_255:
#                 np_clip /= 255.0
#             return np_clip
#         else:
#             tensor_clip = torch.from_numpy(np_clip)
#             if self.div_255:
#                 tensor_clip /= 255.0
#             return tensor_clip


class ClipToTensor(object):
    """ 只把 list of numpy(H,W,14) => (14, m, H, W) 的 floatTensor，不做归一化 """
    def __init__(self, channel_nb=14, div_255=False, numpy=False):
        # div_255 可以关掉，否则会把 mask/depth 也除255
        self.channel_nb = channel_nb
        self.div_255 = div_255  # 默认False
        self.numpy = numpy

    def __call__(self, clip):
        # clip: list of (H,W,14) numpy
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            if ch != self.channel_nb:
                print(f"[WARNING] ClipToTensor expects {self.channel_nb} channels, got {ch}")
        else:
            raise TypeError("Expected list of numpy.ndarray ...")

        # np_clip shape = (channel_nb, num_frames, H, W)
        np_clip = np.zeros([self.channel_nb, len(clip), h, w], dtype=np.float32)
        for i, img in enumerate(clip):
            # img shape=(H,W,14)
            # 直接复制
            # 如果img通道<channel_nb可再写一些c_here=...
            np_clip[:, i, :, :] = img.transpose(2,0,1)

        if self.numpy:
            return np_clip
        else:
            tensor_clip = torch.from_numpy(np_clip)
            if self.div_255:
                tensor_clip = tensor_clip / 255.0
            return tensor_clip
        

class ClipToTensor_dt(object):
    """
    Convert list of m (H x W x C) np.ndarrays => shape (C x m x H x W).
    Default channel_nb=11, originally we had an assert => now relaxed
    """

    def __init__(self, channel_nb=11):
        self.channel_nb = channel_nb

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            # ======= 改动：注释掉硬断言 =======
            # assert ch == self.channel_nb, f"Expected {self.channel_nb} channels, but got {ch} channels."
            if ch != self.channel_nb:
                print(f"[WARNING] ClipToTensor_dt expecting {self.channel_nb} channels, got {ch}. Will proceed anyway.")
        else:
            raise TypeError(f"Expected list of numpy.ndarray, but got list of {type(clip[0])}")
        
        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                c_here = min(img.shape[2], self.channel_nb)
                np_clip[:c_here, img_idx, :, :] = np.transpose(img, (2, 0, 1))[:c_here]
            else:
                raise TypeError(f"Expected numpy.ndarray, but got {type(img)}")
        
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip


class ClipToTensor_dt_10(object):
    """Same as above but channel_nb=10"""
    def __init__(self, channel_nb=10):
        self.channel_nb = channel_nb

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            # assert ch == self.channel_nb, f"Expected {self.channel_nb} channels, but got {ch} channels."
            if ch != self.channel_nb:
                print(f"[WARNING] ClipToTensor_dt_10 expecting {self.channel_nb} channels, got {ch}. Will proceed anyway.")
        else:
            raise TypeError(f"Expected list of numpy.ndarray, but got list of {type(clip[0])}")
        
        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                c_here = min(img.shape[2], self.channel_nb)
                np_clip[:c_here, img_idx, :, :] = np.transpose(img, (2, 0, 1))[:c_here]
            else:
                raise TypeError(f"Expected numpy.ndarray, but got {type(img)}")
        
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip


class ClipToTensor_K(object):
    """Convert a list of m (H x W x C) numpy.ndarrays => (C x m x H x W), norm to [-1,1]."""
    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            # assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
            if ch != self.channel_nb:
                print(f"[WARNING] ClipToTensor_K expecting {self.channel_nb} channels, got {ch} channels. Proceed anyway.")
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image but got {type(clip[0])}")

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(f"Expected numpy.ndarray or PIL.Image but got {type(img)}")

            img = convert_img(img)
            c_here = min(img.shape[0], self.channel_nb)
            np_clip[:c_here, img_idx, :, :] = img[:c_here]

        if self.numpy:
            if self.div_255:
                # original: (x-127.5)/127.5 => [-1,1]
                np_clip = (np_clip - 127.5) / 127.5
            return np_clip
        else:
            tensor_clip = torch.from_numpy(np_clip).float()
            if self.div_255:
                tensor_clip = (tensor_clip - 127.5) / 127.5
            return tensor_clip


class ToTensor(object):
    """Converts numpy array to tensor"""
    def __call__(self, array):
        tensor = torch.from_numpy(array)
        return tensor


class RandomCrop(object):
    def __init__(self, size, padding=0, sequence_length=16):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.sequence_length = sequence_length
        self.padding = padding
        self.count = 0

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // self.sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self, sequence_length=16):
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    def __init__(self,degrees, sequence_length=16):
        self.degrees = degrees
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, sequence_length=16):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)
        return img_
