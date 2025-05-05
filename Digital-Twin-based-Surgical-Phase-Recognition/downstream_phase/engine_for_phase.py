import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.transforms.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from datetime import datetime
from scipy.special import softmax


def train_class_batch(model, samples, features, targets, criterion):
    """
    Args:
        samples: mask+depth输入 (B,11,T,H,W)
        features: GSViT特征 (B,T,384,4,4)
        targets: 标签
    """
    # print("samples min/max: ", samples.min().item(), samples.max().item())
    # print("features min/max: ", features.min().item(), features.max().item())
    # print("targets min/max:", targets.min().item(), targets.max().item())
    # if torch.isnan(targets).any():
    #     print("targets contain NaN!")

    outputs = model(samples, features)
    loss = criterion(outputs, targets)
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return (
        optimizer.loss_scale
        if hasattr(optimizer, "loss_scale")
        else optimizer.cur_scale
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, features, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)  # 新增
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets) #这里可能也要改

        if loss_scaler is None:
            samples = samples.half()
            features = features.half()  # 新增
            loss, output = train_class_batch(model, samples, features, targets, criterion)  # 修改
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(model, samples, features, targets, criterion)  # 修改

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # 数据集返回: (videos, features, target, ids, flags)
        samples, features, targets = batch
        samples = samples.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # model 接收两个输入
            output = model(samples, features)
            loss = criterion(output, target)

        # 计算 top1, top5
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 统计
        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def final_phase_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        # 只拿到 3 个值 (samples, features, targets)
        samples, features, targets = batch

        samples = samples.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(samples, features)  
            output = model(features) 
            loss = criterion(output, targets)

        # 逐样本记录结果 - 无法使用 ids[i]，只能用简单的方式写到文件
        for i in range(output.size(0)):
            # 这里仅示例把预测和标签写到文本
            string = "sample_idx={}  pred={}  label={}\n".format(
                i,
                output[i].cpu().numpy().tolist(),
                int(targets[i].cpu().numpy()),
            )
            final_result.append(string)

        # 计算准确率
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        # 注意 batch_size = samples.shape[0] (而不是 videos.shape[0])
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # 写结果到 file
    if not os.path.exists(file):
        open(file, 'a').close()
    with open(file, "w") as f:
        # 把最后一次 batch 的 acc1, acc5 写进去
        f.write("{}, {}\n".format(acc1.item(), acc5.item()))
        for line in final_result:
            f.write(line)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}

    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, f"{x}.txt")
        print("Merge File %d/%d: %s" % (x + 1, num_tasks, file))

        with open(file, "r") as f:
            lines = f.readlines()[1:]  # 跳过第一行（通常是header）

        for line in lines:
            line = line.strip()
            if "[" not in line or "]" not in line:
                print(f"Warning: Skipping invalid line -> {line}")
                continue

            try:
                # 拿 pred 数据
                data_str = line.split("[")[1].split("]")[0].strip()
                data = np.fromstring(data_str, dtype=float, sep=",")

                # 拿 label (改成用 split('label='))
                right_part = line.split("]")[1].strip()  # 原来 "  label=6"
                if 'label=' not in right_part:
                    print(f"Warning: Skipping line without 'label=' -> {line}")
                    continue
                label_str = right_part.split('label=')[-1].strip()  # => "6"
                if label_str == "":
                    print(f"Warning: Skipping line with empty label -> {line}")
                    continue

                # softmax
                data = softmax(data)
                name = line.split()[0].strip()  # => "sample_idx=1"

                if name not in dict_feats:
                    dict_feats[name] = []
                    dict_label[name] = 0

                dict_feats[name].append(data)
                dict_label[name] = label_str

            except Exception as e:
                print(f"Error processing line: {line}\n{e}")
                continue


    print("Computing final results")
    
    input_lst = []
    print(len(dict_feats))

    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])

    # **减少 `Pool` 进程数，防止 OOM**
    from multiprocessing import Pool

    num_workers = min(16, os.cpu_count() // 2)  # 限制进程数
    with Pool(num_workers) as p:
        ans = p.map(compute_video, input_lst)

    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]

    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100



def compute_video(lst):
    _, _, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]