import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import sys

sys.path.append("/home/haoding/Wenzheng/Digital-Twin-based-Surgical-Phase-Recognition")

from datasets.transforms.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from datasets.transforms.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)

# 14 通道数据集构建脚本
from downstream_phase.datasets_phase_dt_rgb import build_dataset_dt_rgb

# 训练 / 验证 / 测试的核心函数
from downstream_phase.engine_for_phase import (
    train_one_epoch,
    validation_one_epoch,
    final_phase_test,
    merge,
)

from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils

from model.surgformer_HTA_KCA_dt_rgb import surgformer_HTA_KCA_dt_rgb


def get_args():
    parser = argparse.ArgumentParser(
        "SurgVideoMAE script for testing GSViT features via MLP classification",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)

    # resume
    parser.add_argument("--resume", default="", help="resume from checkpoint path")

    # Model parameters
    parser.add_argument("--model", default="surgformer_HTA_KCA_dt_rgb", type=str)
    parser.add_argument("--pretrained_path", default="", type=str)
    parser.add_argument("--use_pretrain", action="store_true", default=False)

    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--fc_drop_rate", type=float, default=0.5)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.1)

    parser.add_argument("--disable_eval_during_finetuning", action="store_true", default=False)

    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=(0.9, 0.999), type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=-1)

    # Aug
    parser.add_argument("--color_jitter", type=float, default=0.4)
    parser.add_argument("--aa", type=str, default="rand-m7-n4-mstd0.5-inc1")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--train_interpolation", type=str, default="bicubic")

    # Eval
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)

    # Random Erase
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # Mixup
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Finetuning
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    # Dataset
    parser.add_argument("--data_path", default="/path/to/data", type=str)
    parser.add_argument("--data_path_rgb", default="/path/to/data", type=str)
    parser.add_argument("--eval_data_path", default="/path/to/data", type=str)
    parser.add_argument("--gsvit_feat_root", default="/path/to/data", type=str)  
    parser.add_argument("--gsvit_feat_root_test", default="/path/to/data", type=str)

    parser.add_argument("--nb_classes", default=7, type=int)
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--data_strategy", type=str, default="online")
    parser.add_argument("--output_mode", type=str, default="key_frame")
    parser.add_argument("--cut_black", action="store_true")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--data_set", default="Cholec80", choices=["Cholec80", "AutoLaparo"], type=str)
    parser.add_argument("--data_fps", default="1fps", choices=["", "5fps", "1fps"], type=str)

    parser.add_argument("--output_dir", default="./output_dt_rgb", help="path where to save")
    parser.add_argument("--log_dir", default="./output_dt_rgb/log", help="log path")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    # early stop
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="stop if val acc not improve after N epochs")

    known_args, _ = parser.parse_known_args()
    ds_init = None
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 如果仅想测试，可以直接 skip 训练集
    # 但这里还是把数据集都加载了，只是用不到
    dataset_train, args.nb_classes = build_dataset_dt_rgb(
        is_train=True, test_mode=False, fps=args.data_fps, args=args
    )
    dataset_val, _ = build_dataset_dt_rgb(
        is_train=False, test_mode=False, fps=args.data_fps, args=args
    )
    dataset_test, _= build_dataset_dt_rgb(
        is_train=False, test_mode=True,  fps=args.data_fps, args=args
    )

    print("Train Dataset:", len(dataset_train))
    print("Val Dataset:", len(dataset_val))
    print("Test Dataset:", len(dataset_test))

    # 仅构建 test DataLoader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # =============== 创建 MLP 模型 =============== #
    model = create_model(
        args.model,               # 'surgformer_HTA_KCA_dt_rgb'
        pretrained=args.use_pretrain,
        pretrain_path=args.pretrained_path,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    )
    model.to(device)

    # 如果你有 checkpoint，需要手动加载
    if args.resume and os.path.isfile(args.resume):
        print(f"==> Resume from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        # 这里按需加载 'model' / 'model_state' / 其他
        if "model" in checkpoint:
            utils.load_state_dict(model, checkpoint["model"])
        elif "model_state" in checkpoint:
            utils.load_state_dict(model, checkpoint["model_state"])
        else:
            utils.load_state_dict(model, checkpoint)
        print("Resume success!")
    else:
        print(f"No valid checkpoint found at {args.resume} (ignore)")

    # =============== 只做 test =============== #
    preds_file = os.path.join(args.output_dir, "test_preds.txt")
    test_stats = final_phase_test(data_loader_test, model, device, preds_file)
    print("Save test predictions to:", preds_file)
    print("Test Stats:", test_stats)

    # =============== 合并多卡输出 (若分布式) =============== #
    global_rank= utils.get_rank()
    num_tasks = utils.get_world_size()
    torch.distributed.barrier()  # 同步
    if global_rank==0:
        print("Start merging results for final test...")
        final_top1, final_top5= merge(args.output_dir, num_tasks)
        print(f"Final Test: Top-1={final_top1:.2f}%, Top-5={final_top5:.2f}%")


if __name__ == "__main__":
    opts, ds_init = get_args()
    main(opts, ds_init)
