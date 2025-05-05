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

# === 改动处：直接 import 你的新 “双分支PatchEmbed” 模型 ===
# 保留函数名 surgformer_HTA_KCA_dt_rgb，但内部是 VisionTransformerTwoPatch
from model.surgformer_HTA_KCA_dt_rgb import surgformer_HTA_KCA_dt_rgb


def get_args():
    parser = argparse.ArgumentParser(
        "SurgVideoMAE fine-tuning and evaluation script for video phase recognition (14-channel: dt+rgb)",
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
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    # For naming
    if args.sampling_rate == 0:
        frame_manner = "Exponential_Stride"
    elif args.sampling_rate == -1:
        frame_manner = "Random_Stride"
    elif args.sampling_rate == -2:
        frame_manner = "Incremental_Stride"
    else:
        frame_manner = "Fixed_Stride_" + str(args.sampling_rate)

    # =============== Output folder ===============
    args.output_dir = os.path.join(
        args.output_dir,
        "_".join([
            args.model,
            args.data_set,
            str(args.lr),
            str(args.layer_decay),
            args.data_strategy,
            args.output_mode,
            "frame" + str(args.num_frames),
            frame_manner,
        ]),
    )
    args.log_dir = os.path.join(args.output_dir, "log")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # =============== Save hyperparams ===============
    if not args.eval:
        txt_file = open(os.path.join(args.output_dir, "hyperparams.txt"), "w")
        txt_file.write(str(args))
        txt_file.close()
    else:
        txt_file = open(os.path.join(args.output_dir, "val_hyperparams.txt"), "w")
        txt_file.write(str(args))
        txt_file.close()

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # =============== Build dataset (14 channels) ===============
    dataset_train, args.nb_classes = build_dataset_dt_rgb(is_train=True,  test_mode=False, fps=args.data_fps, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset_dt_rgb(is_train=False, test_mode=False, fps=args.data_fps, args=args)
    dataset_test, _ = build_dataset_dt_rgb(is_train=False, test_mode=True, fps=args.data_fps, args=args)

    print("Train Dataset Length:", len(dataset_train))
    if dataset_val is not None:
        print("Val Dataset Length:", len(dataset_val))
    else:
        print("Val Dataset is None")
    print("Test Dataset Length:", len(dataset_test))

    # =============== Samplers & DataLoader ===============
    num_tasks = utils.get_world_size()
    global_rank= utils.get_rank()
    sampler_train= torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    if args.dist_eval:
        sampler_val  = torch.utils.data.DistributedSampler(dataset_val,   num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(dataset_test,  num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Use distributed sampler for val/test.")
    else:
        sampler_val  = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        print("Log dir:", args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test= torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_test= None

    # =============== Mixup ===============
    mixup_fn = None
    mixup_active= (args.mixup>0) or (args.cutmix>0) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    # =============== Create model (two-patch branch) ===============
    model = create_model(
        args.model,
        pretrained=args.use_pretrain,
        pretrain_path=args.pretrained_path,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        # 其他kwargs...
    )

    # 注意：新的“TwoPatch”模型没有 .patch_embed.patch_size 了，需要单独取
    if hasattr(model, "patch_embed_rgb"):
        patch_size_rgb = model.patch_embed_rgb.patch_size
        patch_size_md  = model.patch_embed_md.patch_size
        print("Patch size (RGB):", patch_size_rgb, " / Patch size (MD):", patch_size_md)
    # else:  如果 model是别的，就不处理

    # 仍然可以做 window_size 之类(若你需要)
    # args.window_size = ...
    # args.patch_size  = patch_size_rgb

    # Finetune if needed
    if args.finetune:
        print("Load finetune checkpoint from:", args.finetune)
        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print(f"Load state_dict by model_key = {model_key}")
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        # do remove head.weight/head.bias
        state_dict = model.state_dict()
        for k in ["head.weight","head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape!= state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # =============== LR 相关 ===============
    total_batch_size = args.batch_size*args.update_freq* utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train)// total_batch_size
    args.lr       = args.lr* total_batch_size/64
    args.min_lr   = args.min_lr* total_batch_size/64
    args.warmup_lr= args.warmup_lr* total_batch_size/64
    print("Adjusted LR = %.8f" % args.lr)
    print("Total batch size =", total_batch_size)
    print("Update freq =", args.update_freq)
    print("Number of train examples =", len(dataset_train))
    print("Train steps per epoch =", num_training_steps_per_epoch)

    # layer-wise decay
    if hasattr(model_without_ddp, "get_num_layers"):
        num_layers = model_without_ddp.get_num_layers()
    else:
        num_layers = 12
    if args.layer_decay<1.0:
        assigner = LayerDecayValueAssigner(
            [args.layer_decay**(num_layers+1 - i) for i in range(num_layers+2)]
        )
        print("Layer-wise LR decay assignment:", assigner.values)
    else:
        assigner=None

    skip_weight_decay_list = model.no_weight_decay() if hasattr(model,"no_weight_decay") else {}
    print("Skip weight decay list:", skip_weight_decay_list)

    # Deepspeed or normal
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner else None,
            assigner.get_scale   if assigner else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        assert model.gradient_accumulation_steps()== args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module
        optimizer = create_optimizer(
            args, model_without_ddp,
            skip_list= skip_weight_decay_list,
            get_num_layer= assigner.get_layer_id if assigner else None,
            get_layer_scale= assigner.get_scale if assigner else None,
        )
        loss_scaler = NativeScaler()

    # LR & WD schedule
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps
    )
    if args.weight_decay_end is None:
        args.weight_decay_end= args.weight_decay
    wd_schedule_values= utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch
    )

    # Criterion
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing>0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("Criterion =", criterion)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu")
            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"]+1
                print(f"setting start_epoch={args.start_epoch} from resume ckpt")
            # load model
            if "model" in checkpoint:
                utils.load_state_dict(model, checkpoint["model"])
            elif "model_state" in checkpoint:
                utils.load_state_dict(model, checkpoint["model_state"])
            else:
                utils.load_state_dict(model, checkpoint)
            # load optimizer
            if "optimizer" in checkpoint and "param_groups" in optimizer.state_dict():
                print("Resuming optimizer state..")
                optimizer.load_state_dict(checkpoint["optimizer"])
            if loss_scaler and "scaler" in checkpoint:
                print("Resuming loss_scaler state..")
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print(f"=> resume success, start epoch {args.start_epoch}")
        else:
            print(f"Warning: no ckpt found at {args.resume}, ignore resume")

    # eval only?
    if args.eval:
        preds_file = os.path.join(args.output_dir, f"{global_rank}.txt")
        test_stats = final_phase_test(data_loader_test, model, device, preds_file)
        print("Save Files:", preds_file)
        if global_rank==0:
            print("Start merging results...")
            final_top1, final_top5 = merge(args.output_dir, num_tasks)
            print(f"Test: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats= {"Final top-1": final_top1, "Final Top-5": final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir,"log.txt"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats)+"\n")
        return

    # ============ Train Loop ============
    print(f"Start training for {args.epochs} epochs")
    start_time  = time.time()
    max_accuracy= 0.0
    max_epoch   = 0
    no_improve_count=0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if log_writer:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device,
            epoch, loss_scaler, args.clip_grad, None, mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )

        # save checkpoint
        if args.output_dir and args.save_ckpt:
            if ((epoch + 1) % args.save_ckpt_freq == 0) or ((epoch + 1) == args.epochs):
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch, model_ema=None,
                )

        # 注释掉验证部分
        # validation
        # if (dataset_val is not None) and (not args.disable_eval_during_finetuning):
        #     test_stats = validation_one_epoch(data_loader_val, model, device)
        #     print(f"Val Acc@1: {test_stats['acc1']:.1f}%")
        #     if max_accuracy < test_stats["acc1"]:
        #         max_accuracy = test_stats["acc1"]
        #         max_epoch = epoch
        #         no_improve_count = 0
        #         # save best
        #         if args.output_dir and args.save_ckpt:
        #             utils.save_model(
        #                 args=args, model=model, model_without_ddp=model_without_ddp,
        #                 optimizer=optimizer, loss_scaler=loss_scaler,
        #                 epoch="best", model_ema=None,
        #             )
        #     else:
        #         no_improve_count += 1
        #         print(f"No improvement at epoch={epoch}, no_improve_count={no_improve_count}")
        #         if no_improve_count >= args.early_stop_patience:
        #             print(f"Early stop triggered (>= {args.early_stop_patience}).")
        #             break

        # 更新日志，不包括验证指标
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            # **{f"val_{k}": v for k, v in test_stats.items()},  # 注释掉验证日志部分
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if log_writer:
            log_writer.update(val_acc1=None, head="perf", step=epoch)
            log_writer.update(val_acc5=None, head="perf", step=epoch)
            log_writer.update(val_loss=None, head="perf", step=epoch)
            # log_writer.update(val_acc1=test_stats["acc1"], head="perf", step=epoch)
            # log_writer.update(val_acc5=test_stats["acc5"], head="perf", step=epoch)
            # log_writer.update(val_loss=test_stats["loss"], head="perf", step=epoch)

        if args.output_dir and utils.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
           

    # ============ Training done, load best => test ============

    # create same model from scratch
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    )
    preds_file = os.path.join(args.output_dir, f"{global_rank}.txt")

    best_pretrained_path = os.path.join(args.output_dir, "checkpoint-best/mp_rank_00_model_states.pt")
    if not os.path.exists(best_pretrained_path):
        print("Warning: no checkpoint-best found. Use last epoch checkpoint instead.")
        last_ckpt = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
        if os.path.exists(last_ckpt):
            best_pretrained_path = last_ckpt
        else:
            print("No suitable ckpt found, skip final test.")
            return

    checkpoint= torch.load(best_pretrained_path, map_location="cpu")
    print("Load best ckpt from", best_pretrained_path)
    checkpoint_model=None
    for model_key in args.model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model= checkpoint[model_key]
            print("Load state_dict by model_key =", model_key)
            break
    if checkpoint_model is None:
        checkpoint_model= checkpoint

    state_dict= final_model.state_dict()
    for k in ["head.weight","head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape!= state_dict[k].shape:
            print(f"Removing key {k} from ckpt")
            del checkpoint_model[k]

    utils.load_state_dict(final_model, checkpoint_model, prefix=args.model_prefix)
    final_model.to(device)

    if data_loader_test is not None:
        test_stats= final_phase_test(data_loader_test, final_model, device, preds_file)
        torch.distributed.barrier()
        if global_rank==0:
            print("Start merging results for final test...")
            final_top1, final_top5= merge(args.output_dir, num_tasks)
            print(f"Final Test: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats= {"Final top-1": final_top1, "Final Top-5": final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir,"log.txt"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats)+"\n")

    total_time= time.time()- start_time
    print("Training time:", str(datetime.timedelta(seconds=int(total_time))))


if __name__=="__main__":
    opts, ds_init= get_args()
    main(opts, ds_init)
