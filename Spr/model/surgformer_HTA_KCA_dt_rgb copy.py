# 模型的base版本，重构了TimeSFormer模型，基本上与TimeSFormer模型结果一致；
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/yangshu/Surgformer")
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math


def _cfg(url="", **kwargs):
    """
    修改为 14 通道输入，将 mean 和 std 对应到 14 个通道。
    """
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (14, 224, 224),   # 14 通道
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5,) * 14,
        "std": (0.5,) * 14,
        **kwargs,
    }


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Spatial(nn.Module):
    """
    与 3 通道或 11 通道版本相同，仅在 patch_embed 时区分 in_chans=14 即可
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, B):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv,
            "(b t) k (qkv nh c) -> qkv (b t) nh k c",
            b=B,
            t=T,
            qkv=3,
            nh=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(x, "(b t) nh k c -> (b t) k (nh c)", b=B)
        x = self.proj(x)
        return self.proj_drop(x)


class Attention_Temporal(nn.Module):
    """
    与 surgformer_HTA_KCA_dt.py 相同，对时序进行 4/8/16 的 attention。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv_4 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_8 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_16 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj_4 = nn.Linear(dim, dim)
            self.proj_8 = nn.Linear(dim, dim)
            self.proj_16 = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, B):
        BK, T, C = x.shape
        t1 = T // 4
        t2 = T // 2
        x_4 = x[:, T - t1 :, :]
        x_8 = x[:, t2:, :]
        x_16 = x
        K = BK // B

        # 4
        qkv_4 = self.qkv_4(x_4)
        qkv_4 = rearrange(
            qkv_4,
            "(b k) t (qkv nh c) -> qkv (b k) nh t c",
            b=B,
            k=K,
            qkv=3,
            nh=self.num_heads,
        )
        q_4, k_4, v_4 = qkv_4[0], qkv_4[1], qkv_4[2]
        attn_4 = (q_4 @ k_4.transpose(-2, -1)) * self.scale
        attn_4 = attn_4.softmax(dim=-1)
        attn_4 = self.attn_drop(attn_4)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_4 = self.proj_4(x_4)

        # 8
        qkv_8 = self.qkv_8(x_8)
        qkv_8 = rearrange(
            qkv_8,
            "(b k) t (qkv nh c) -> qkv (b k) nh t c",
            b=B,
            k=K,
            qkv=3,
            nh=self.num_heads,
        )
        q_8, k_8, v_8 = qkv_8[0], qkv_8[1], qkv_8[2]
        attn_8 = (q_8 @ k_8.transpose(-2, -1)) * self.scale
        attn_8 = attn_8.softmax(dim=-1)
        attn_8 = self.attn_drop(attn_8)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_8 = self.proj_8(x_8)

        # 16
        qkv_16 = self.qkv_16(x_16)
        qkv_16 = rearrange(
            qkv_16,
            "(b k) t (qkv nh c) -> qkv (b k) nh t c",
            b=B,
            k=K,
            qkv=3,
            nh=self.num_heads,
        )
        q_16, k_16, v_16 = qkv_16[0], qkv_16[1], qkv_16[2]
        attn_16 = (q_16 @ k_16.transpose(-2, -1)) * self.scale
        attn_16 = attn_16.softmax(dim=-1)
        attn_16 = self.attn_drop(attn_16)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_16 = self.proj_16(x_16)

        # fusion
        x_8[:, t1:, :] = 0.5 * x_8[:, t1:, :] + 0.5 * x_4
        x_8 = self.proj_drop(x_8)
        x_16[:, t2:, :] = 0.5 * x_16[:, t2:, :] + 0.5 * x_8
        x_16 = self.proj_drop(x_16)
        return x_16


class Block(nn.Module):
    """
    与 surgformer_HTA_KCA_dt.py 中相同，只要注意输入通道对齐就可以。
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Spatial(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention_Temporal(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_cls = norm_layer(dim)

    def forward(self, x, B, T, K):
        # x.shape = (B, M, C)，其中 M = T*K + 1
        B, M, C = x.shape
        assert T * K + 1 == M
        # Step1: Temporal Attn
        xt = x[:, 1:, :]  # 去掉 cls token
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
        res_temp = self.drop_path(self.temporal_attn(self.temporal_norm1(xt), B))
        res_temp = rearrange(res_temp, "(b k) t c -> b (k t) c", b=B)
        xt = self.temporal_fc(res_temp) + x[:, 1:, :]

        # Step2: Spatial Attn
        init_cls_token = x[:, 0, :].unsqueeze(1)  # (B,1,C)
        cls_token = init_cls_token.repeat(1, T, 1)  # (B,T,C)
        cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = rearrange(xt, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), dim=1)  # (B*T, K+1, C)
        res_spatial = self.drop_path(self.attn(self.norm1(xs), B))

        # Step2.1: 处理 spatial CLS token
        cls_token_spatial = res_spatial[:, 0, :]  # (B*T, C)
        cls_token_spatial = rearrange(cls_token_spatial, "(b t) c -> b t c", b=B, t=T)
        cls_token_spatial = self.norm_cls(cls_token_spatial)
        target_token = cls_token_spatial[:, -1, :].unsqueeze(1)  # 只取最后一帧
        attn = (target_token @ cls_token_spatial.transpose(-1, -2)).softmax(dim=-1)
        cls_token_spatial = attn @ cls_token_spatial

        # Step2.2: spatial leftover
        res_spatial = res_spatial[:, 1:, :]  # (B*T, K, C)
        res_spatial = rearrange(res_spatial, "(b t) k c -> b (k t) c", b=B)
        x_s = res_spatial

        # Step3: Combine
        x_cat = torch.cat((init_cls_token, xt), 1) + torch.cat((cls_token_spatial, x_s), 1)
        x_out = x_cat + self.drop_path(self.mlp(self.norm2(x_cat)))
        return x_out


class PatchEmbed(nn.Module):
    """
    将输入 (B, 14, T, H, W) -> (B, T, K, C)
    注意 in_chans=14。
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=14,
        embed_dim=768,
        num_frames=8,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x.shape = (B, 14, T, H, W)
        B, C, T, H, W = x.shape
        # 先 (B*T, C, H, W)，再 conv2d => (B*T, embed_dim, H//patch, W//patch)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        assert (H == self.img_size[0]) and (W == self.img_size[1]), (
            f"Input size {H}x{W} doesn't match {self.img_size}!"
        )
        x = self.proj(x).flatten(2)  # => (B*T, embed_dim, patch_H*patch_W)
        # 再 (B*T, embed_dim, K) => (B, T, K, embed_dim)
        x = rearrange(x, "(b t) c k -> b t k c", b=B)
        return x


class VisionTransformer(nn.Module):
    """
    主干，与 surgformer_HTA_KCA_dt.py 类似，
    但 _cfg() / patch_embed 中 in_chans=14。
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=14,
        num_classes=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        all_frames=16,
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
        )
        num_patches = self.patch_embed.num_patches

        # Position Embeds
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (num_patches // all_frames) + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Head
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Init
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # 使多层的 temporal_fc 在初始化时后几层为 0
        i = 0
        for m in self.blocks.modules():
            if isinstance(m, Block):
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}
    
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        # # x: (B, 14, T, H, W)
        # B, C, T, H, W = x.shape
        # # patch embedding => (B, T, K, C_emb)
        # x = self.patch_embed(x)
        # B, T, K, C_emb = x.shape

        # # + spatial pos embed
        # x = rearrange(x, "b t k c -> (b t) k c")  
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B*T, 1, C)
        # x = torch.cat((cls_tokens, x), dim=1)  # (B*T, K+1, C)
        # x = x + self.pos_embed  # broadcasting到 K+1 维度
        # x = self.pos_drop(x)

        # # + temporal pos embed
        # # 把 cls token 与普通 token 分离
        # cls_tokens = x[:, 0, :].unsqueeze(1)  # (B*T,1,C)
        # x = x[:, 1:, :]  # (B*T,K,C)
        # x = rearrange(x, "(b t) k c -> (b k) t c", b=B)  # (B*K, T, C)
        # x = x + self.time_embed
        # x = self.time_drop(x)

        # # 再次合并
        # x = rearrange(x, "(b k) t c -> b (k t) c", b=B)
        # print('cls_tokens shape:',cls_tokens.shape)
        # print('x shape:',x.shape)
        # x = torch.cat((cls_tokens, x), dim=1)  # (B, K*T+1, C)

        # # 进入 block
        # for blk in self.blocks:
        #     x = blk(x, B, T, K)

        # x = self.norm(x)
        # return x[:, 0]
        
        # B, C, T, H, W
        B, C, T, H, W = x.shape
        
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        
        # 1x1 convolution to convert 11 channels to 3 channels
        # x = rearrange(x, 'b c t h w -> (b t) c h w')  # Convert for 2D conv
        # x = self.conv1x1(x)
        # x = rearrange(x, '(b t) c h w -> b c t h w', b=B) 
        
        x = self.patch_embed(x)
        # B, T, K, C
        B, T, K, C = x.size()
        W = int(math.sqrt(K))

        # 添加Spatial Position Embedding
        x = rearrange(x, "b t k c -> (b t) k c")
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # BT, 1, C
        x = torch.cat((cls_tokens, x), dim=1)  # BT, HW+1, C  ---> 2*8, 196+1, 768
        x = x + self.pos_embed  # BT, HW, C  ---> 2*8, 196, 768
        x = self.pos_drop(x)

        # 添加Temporal Position Embedding
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]  # 过滤掉cls_tokens
        x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
        x = x + self.time_embed  # BK, T, C  ---> 2*196, 8, 768
        x = self.time_drop(x)

        # 添加Cls token
        x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens

        x = torch.cat((cls_tokens, x), dim=1)  # 时空tokens对应的class token的添加；

        for blk in self.blocks:
            x = blk(x, B, T, K)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def surgformer_HTA_KCA_dt_rgb(pretrained=False, pretrain_path=None, **kwargs):
    """
    与 surgformer_HTA_KCA_dt 相同，但改为 in_chans=14；_cfg() 中 input_size=(14,224,224)。
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=14,        # 这里关键之处：14 通道
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained and pretrain_path is not None:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        # （下面这段与 dt 版本类似，如果你有特定的预训练模型，需要做 rename/strip 等操作可自行修改）
        if "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            # 根据需要，对 qkv_4, qkv_8, qkv_16 或 patch_embed 做一些兼容处理
            add_list = []
            for k in state_dict.keys():
                if ("blocks" in k) and ("qkv_4" in k):
                    k_init = k.replace("qkv_4", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                # ... 同理 qkv_8, qkv_16, proj_4, proj_8, proj_16
            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                # 当输入通道不同，patch_embed.weight 形状不符，就删掉
                if ("patch_embed" in k or "head" in k) and k in checkpoint:
                    if checkpoint[k].shape != state_dict[k].shape:
                        remove_list.append(k)
                        del checkpoint[k]
            print("Removing keys from pretrained checkpoint:", ", ".join(remove_list))

            utils.load_state_dict(model, checkpoint)
        else:
            # 其他情况的 load 逻辑略
            pass

    return model
