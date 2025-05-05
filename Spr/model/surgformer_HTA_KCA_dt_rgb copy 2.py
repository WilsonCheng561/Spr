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
    这里仅保留原先14通道的 mean/std，不过由于现在分成了RGB+MD两支，
    如果你在数据层面已经分别做了归一化，也可以不必写14维的mean/std。
    """
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (14, 224, 224),   # 原先写14通道
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5,) * 14,
        "std": (0.5,) * 14,
        **kwargs,
    }


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop= nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Spatial(nn.Module):
    """
    与原先相同
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
            self.proj= nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop= nn.Dropout(attn_drop)

    def forward(self, x, B):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "(b t) k (three nh c) -> three (b t) nh k c", b=B, t=T, three=3, nh=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(x, "(b t) nh k c -> (b t) k (nh c)", b=B)
        x = self.proj(x)
        return self.proj_drop(x)


class Attention_Temporal(nn.Module):
    """
    与原先相同: 4/8/16 multi-scale
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv_4  = nn.Linear(dim, dim*3, bias=qkv_bias)
            self.qkv_8  = nn.Linear(dim, dim*3, bias=qkv_bias)
            self.qkv_16 = nn.Linear(dim, dim*3, bias=qkv_bias)
            self.proj_4 = nn.Linear(dim, dim)
            self.proj_8 = nn.Linear(dim, dim)
            self.proj_16= nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop= nn.Dropout(attn_drop)

    def forward(self, x, B):
        BK, T, C = x.shape
        t1 = T//4
        t2 = T//2
        x_4  = x[:, T - t1:, :]
        x_8  = x[:, t2:, :]
        x_16 = x
        K = BK//B

        # step4
        qkv_4 = self.qkv_4(x_4)
        qkv_4 = rearrange(qkv_4, "(b k) t (three nh c) -> three (b k) nh t c", b=B, k=K, three=3, nh=self.num_heads)
        q_4, k_4, v_4 = qkv_4[0], qkv_4[1], qkv_4[2]
        attn_4 = (q_4 @ k_4.transpose(-2,-1))*self.scale
        attn_4 = attn_4.softmax(dim=-1)
        attn_4 = self.attn_drop(attn_4)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_4 = self.proj_4(x_4)

        # step8
        qkv_8 = self.qkv_8(x_8)
        qkv_8 = rearrange(qkv_8, "(b k) t (three nh c) -> three (b k) nh t c", b=B, k=K, three=3, nh=self.num_heads)
        q_8, k_8, v_8 = qkv_8[0], qkv_8[1], qkv_8[2]
        attn_8 = (q_8 @ k_8.transpose(-2,-1))*self.scale
        attn_8 = attn_8.softmax(dim=-1)
        attn_8 = self.attn_drop(attn_8)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_8 = self.proj_8(x_8)

        # step16
        qkv_16= self.qkv_16(x_16)
        qkv_16= rearrange(qkv_16, "(b k) t (three nh c) -> three (b k) nh t c", b=B, k=K, three=3, nh=self.num_heads)
        q_16, k_16, v_16 = qkv_16[0], qkv_16[1], qkv_16[2]
        attn_16 = (q_16 @ k_16.transpose(-2,-1))*self.scale
        attn_16 = attn_16.softmax(dim=-1)
        attn_16 = self.attn_drop(attn_16)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) nh t c -> (b k) t (nh c)", b=B)
        x_16 = self.proj_16(x_16)

        # fusion
        x_8[:, t1:]  = 0.5*x_8[:, t1:]  + 0.5*x_4
        x_8 = self.proj_drop(x_8)
        x_16[:, t2:] = 0.5*x_16[:, t2:] + 0.5*x_8
        x_16= self.proj_drop(x_16)
        return x_16


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.2, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = dim**-0.5
        self.norm1 = norm_layer(dim)
        self.attn  = Attention_Spatial(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn  = Attention_Temporal(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                 qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc    = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        hidden_dim     = int(dim*mlp_ratio)
        self.mlp       = Mlp(dim, hidden_dim, dim, act_layer=act_layer, drop=drop)
        self.norm_cls  = norm_layer(dim)

    def forward(self, x, B, T, K):
        # x: (B, M, C), M= 1 + T*K
        # 1) TemporalAttn
        xt = x[:, 1:, :]  # remove cls
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
        res_temp= self.drop_path(self.temporal_attn(self.temporal_norm1(xt), B))
        res_temp= rearrange(res_temp, "(b k) t c -> b (k t) c", b=B)
        xt = self.temporal_fc(res_temp)+ x[:,1:,:]

        # 2) SpatialAttn
        init_cls_token = x[:,0,:].unsqueeze(1) # (B,1,C)
        cls_token      = init_cls_token.repeat(1,T,1)  #(B,T,C)
        cls_token      = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = rearrange(xt, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), dim=1)  # =>(B*T, K+1, C)
        res_spatial= self.drop_path(self.attn(self.norm1(xs), B))

        # 2.1 handle spatial cls
        cls_token_spatial= res_spatial[:, 0, :]
        cls_token_spatial= rearrange(cls_token_spatial, "(b t) c -> b t c", b=B, t=T)
        cls_token_spatial= self.norm_cls(cls_token_spatial)
        target_token     = cls_token_spatial[:, -1, :].unsqueeze(1)
        attn2 = (target_token @ cls_token_spatial.transpose(-1, -2)).softmax(dim=-1)
        cls_token_spatial= attn2 @ cls_token_spatial

        # leftover
        res_spatial= res_spatial[:,1:,:]
        res_spatial= rearrange(res_spatial, "(b t) k c -> b (k t) c", b=B)
        x_s= res_spatial

        # 3) combine
        x_cat= torch.cat((init_cls_token, xt), dim=1) + torch.cat((cls_token_spatial, x_s), dim=1)
        x_out= x_cat + self.drop_path(self.mlp(self.norm2(x_cat)))
        return x_out


# =============== 重点改动：加两个PatchEmbed分别处理RGB/MD，然后合并 ===============
class PatchEmbedRGB(nn.Module):
    """原先 PatchEmbed 改成 in_chans=3"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_frames=8):
        super().__init__()
        self.img_size   = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_frames = num_frames
        num_patches_hw  = (self.img_size[0]//self.patch_size[0])*(self.img_size[1]//self.patch_size[1])
        self.num_patches= num_patches_hw*self.num_frames
        self.proj       = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x).flatten(2)  # => (B*T, embed_dim, #patch_hw)
        x = rearrange(x, '(b t) c p -> b t p c', b=B, t=T)
        return x


class PatchEmbedMD(nn.Module):
    """原先 PatchEmbed 改成 in_chans=11"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_frames=8):
        super().__init__()
        self.img_size   = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_frames = num_frames
        num_patches_hw  = (self.img_size[0]//self.patch_size[0])*(self.img_size[1]//self.patch_size[1])
        self.num_patches= num_patches_hw*self.num_frames
        self.proj       = nn.Conv2d(in_channels=11, out_channels=embed_dim,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x: (B, 11, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x).flatten(2)
        x = rearrange(x, '(b t) c p -> b t p c', b=B, t=T)
        return x


class VisionTransformerTwoPatch(nn.Module):
    """
    在原 surgformer_HTA_KCA_dt_rgb基础上做最小改动：
    1) 用 patch_embed_rgb + patch_embed_md 分别处理RGB(3通道)和MD(11通道);
    2) 合并后再做同样的时空Attention流程.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=14,  # 不再使用; 仅保持参数形式
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
        self.depth       = depth
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.all_frames  = all_frames

        # 改动1: 两个分支 patch embed
        self.patch_embed_rgb = PatchEmbedRGB(img_size, patch_size, embed_dim, num_frames=all_frames)
        self.patch_embed_md  = PatchEmbedMD(img_size, patch_size, embed_dim, num_frames=all_frames)
        # => 两倍 patch
        # 原先 patch_embed.num_patches => K*T
        # 现在 => (K_rgb + K_md)*T = 2*K*T
        # 假设 K_rgb == K_md => 2*K
        self.num_patches = self.patch_embed_rgb.num_patches + self.patch_embed_md.num_patches

        # Position Embeds
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed 大小改为 (num_patches +1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        # Time embed
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop  = nn.Dropout(p=drop_rate)

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
        self.fc_dropout = nn.Dropout(fc_drop_rate) if fc_drop_rate>0 else nn.Identity()
        self.head       = nn.Linear(embed_dim, num_classes) if num_classes>0 else nn.Identity()

        # init
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # init temporal_fc=0 for blocks>0
        i = 0
        for m in self.blocks.modules():
            if isinstance(m, Block):
                if i>0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i+=1

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

    def forward_features(self, x):
        """
        x: (B,14,T,H,W)，其中前3通道是RGB，后11通道是mask+depth
        """
        B, C, T, H, W = x.shape
        # Step1: split
        x_rgb = x[:, :3, ...]   # =>(B,3,T,H,W)
        x_md  = x[:, 3:, ...]   # =>(B,11,T,H,W)

        # Step2: 分别 patch_embed
        x_rgb = self.patch_embed_rgb(x_rgb)  # => (B,T,K_rgb,dim)
        x_md  = self.patch_embed_md(x_md)    # => (B,T,K_md,dim)
        # 合并 patch 维度
        x = torch.cat([x_rgb, x_md], dim=2)  # =>(B,T, K_rgb+K_md, dim) ~ (B,T,2K,dim)

        # Step3: flatten time+patch => (B, T*(2K), dim)
        B, T_, P_, C_ = x.shape
        # T_== self.all_frames, P_== K_rgb+K_md
        x = rearrange(x, "b t p c -> b (t p) c")

        # Step4: 添加CLS token & 位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)  # =>(B,1,C)
        x = torch.cat((cls_tokens, x), dim=1)          # =>(B, 1 + T*(2K), C)
        # x.shape[1] 应当 <= self.pos_embed.shape[1]
        if x.shape[1]> self.pos_embed.shape[1]:
            # 若(实际patch数>pos_embed大小),请自行插值pos_embed
            # 这里仅给个警告
            print(f"[WARN] x.shape[1]={x.shape[1]} > pos_embed.shape[1]={self.pos_embed.shape[1]}, may need interpolation.")
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)

        # Step5: time_embed
        # 先把 cls_token 与普通token分开
        cls_ = x[:, 0, :].unsqueeze(1)  # =>(B,1,C)
        x_   = x[:, 1:, :]             # =>(B, T*(2K), C)
        # resh =>(B*(2K), T, C)
        # 需要知道 2K 与 T, or T_ & P_...
        # x_ = rearrange(x_, 'b (t p) c -> (b p) t c', b=B, t=self.all_frames)
        # 这里2K= P_, so
        x_ = rearrange(x_, 'b (t p) c -> (b p) t c', b=B, t=self.all_frames)
        # + time_embed
        if x_.shape[1] != self.time_embed.shape[1]:
            print(f"[WARN] time dim mismatch for time_embed, shape={x_.shape}")
        x_ = x_ + self.time_embed
        x_ = self.time_drop(x_)

        # reshape回 =>(B, 1 + T*(2K), C)
        x_ = rearrange(x_, '(b p) t c -> b (t p) c', b=B)
        x = torch.cat((cls_, x_), dim=1)

        # Step6: 进入 blocks
        # 现在K=2K(合并后), T= self.all_frames
        # => M= 1 + T*(2K).  sqrt(2K) ~ W'
        K_2 = P_  # K_2 = (K_rgb + K_md)
        T_2 = self.all_frames
        for blk in self.blocks:
            x = blk(x, B, T_2, K_2)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def surgformer_HTA_KCA_dt_rgb(pretrained=False, pretrain_path=None, **kwargs):
    """
    最小改动：双 patch_embed 分支 + concat => 进后续Block
    """
    model = VisionTransformerTwoPatch(**kwargs)
    model.default_cfg = _cfg()

    if pretrained and pretrain_path is not None:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()

        if "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k,v in checkpoint.items():
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            # qkv_4, _8, _16 rename etc. 与原先一样
            add_list=[]
            for k in state_dict.keys():
                if ("blocks" in k) and ("qkv_4" in k):
                    k_init = k.replace("qkv_4","qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                # ...  同理
            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list=[]
            for k in state_dict.keys():
                # 由于本模型是 两倍patch => patch_embed_rgb.weight, patch_embed_md.weight
                # 预训练权重很可能形状对不上
                if "patch_embed" in k or "head" in k:
                    if k in checkpoint and checkpoint[k].shape!=state_dict[k].shape:
                        remove_list.append(k)
                        del checkpoint[k]
            print("Removing keys from pretrained checkpoint:", ", ".join(remove_list))

            utils.load_state_dict(model, checkpoint)
        else:
            # 其他情况
            pass

    return model
