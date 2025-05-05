import torch
import torch.nn as nn
import torch.nn.functional as F

class GSVitMLP(nn.Module):
    """
    简单的 MLP，用于对 GSViT 特征做分类：
      - 输入: (B, T, 384,4,4) 或 (B,384,4,4)
      - 先 flatten => (B, 6144) 或 (B, T, 6144) => 对 T 做 average => => MLP => num_classes
    """
    def __init__(self, in_dim=6144, hidden_dim=1024, num_classes=7, drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x.shape 可以是：
          (B, T, 384, 4, 4) 或 (B, 384, 4,4)
        """
        if x.ndim == 5:
            # (B, T, 384,4,4)
            B, T, C, H, W = x.shape
            x = x.view(B, T, C*H*W)  # => (B, T, 6144)
            x = x.mean(dim=1)       # => (B, 6144)，对时间帧平均
        elif x.ndim == 4:
            # (B,384,4,4)
            B, C, H, W = x.shape
            x = x.view(B, C*H*W)    # => (B,6144)
        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def surgformer_HTA_KCA_dt_rgb(
    pretrained=False, 
    pretrain_path=None, 
    num_classes=7, 
    all_frames=8,
    fc_drop_rate=0.5,
    **kwargs
):
    """
    与原先同名函数，但内部直接返回 GSVitMLP，而非真正的 ViT/Transformer。
    """
    print("====> Using GSVitMLP as model instead of the original Transformer <====")
    model = GSVitMLP(
        in_dim=6144,         # 384*4*4
        hidden_dim=1024,     # 你可调参
        num_classes=num_classes,
        drop=fc_drop_rate,
    )

    # 如果指定了 pretrained，这里也不加载任何特征，因为MLP没对应权重
    if pretrained and pretrain_path is not None:
        print(f"[Warning] MLP does not load pretrain from {pretrain_path} (no matching keys).")

    return model
