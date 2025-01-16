# -----------------------------
# Surgical Video Frame Black Margin Removal Script
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import cv2
import os
import numpy as np
import _multiprocessing
from tqdm import tqdm  

# ''''''''
# 裁剪外科手术视频帧中的黑色边框，以标准化帧的大小和内容。裁剪后的视频帧会被重新调整为固定大小（250x250像素），并保存到指定的路径。
# 适用于需要对帧进行预处理的外科手术视频数据集，具体包括以下步骤：

# 遍历数据集中的所有视频帧。
# 检测帧中存在的有效内容区域，去除黑色边框。
# 将裁剪后的帧调整为固定大小并保存
# ''''''''



# 创建目标文件夹（如果不存在）
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 去除帧中的黑色边框
def filter_black(image):
    # 转换图像为灰度图
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值处理，将像素值小于15的区域变为黑色
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)

    # 使用中值滤波去除噪声（参数19可根据数据集调整）
    binary_image2 = cv2.medianBlur(binary_image2, 19)

    # 获取图像尺寸
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    # 存储有效区域的边界坐标
    edges_x = []
    edges_y = []

    # 遍历图像，寻找非黑色区域的边界
    for i in range(x):
        for j in range(10, y - 10):  # 避免考虑边缘的噪声像素
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    # 如果未找到有效区域，直接返回原图
    if not edges_x:
        return image

    # 计算有效区域的边界
    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 下边界
    top = max(edges_y)  # 上边界
    height = top - bottom  # 高度

    # 裁剪图像到有效区域
    pre1_picture = image[left : left + width, bottom : bottom + height]

    return pre1_picture


# 处理单张图像
def process_image(image_source, image_save):
    # 读取图像
    frame = cv2.imread(image_source)

    # 调整图像大小以保证统一的纵横比
    dim = (int(frame.shape[1] / frame.shape[0] * 300), 300)
    frame = cv2.resize(frame, dim)

    # 去除黑色边框
    frame = filter_black(frame)

    # 调整图像到固定大小 250x250
    img_result = cv2.resize(frame, (250, 250))

    # 保存处理后的图像
    cv2.imwrite(image_save, img_result)


# 处理单个视频文件夹中的所有帧
def process_video(video_id, video_source, video_save):
    # 确保保存路径存在
    create_directory_if_not_exists(video_save)

    # 遍历视频帧文件
    for image_id in sorted(os.listdir(video_source)):
        if image_id == ".DS_Store":  # 跳过无效的系统文件
            continue
        image_source = os.path.join(video_source, image_id)  # 源图像路径
        image_save = os.path.join(video_save, image_id)  # 保存图像路径

        # 处理并保存图像
        process_image(image_source, image_save)


# 主程序
if __name__ == "__main__":
    # 设置源帧路径和保存路径
    source_path = "/home/yangshu/Surgformer/data/Cholec80/frames"  # 原始帧路径
    save_path = "/home/yangshu/Surgformer/data/Cholec80/frames_cutmargin"  # 处理后帧的保存路径

    # 确保保存路径存在
    create_directory_if_not_exists(save_path)

    # 初始化进程列表
    processes = []

    # 遍历所有视频文件夹
    for video_id in tqdm(os.listdir(source_path)):
        if video_id == ".DS_Store":  # 跳过无效的系统文件
            continue
        video_source = os.path.join(source_path, video_id)  # 源视频帧路径
        video_save = os.path.join(save_path, video_id)  # 保存路径

        # 创建并启动多进程处理视频帧
        process = multiprocessing.Process(
            target=process_video, args=(video_id, video_source, video_save)
        )
        process.start()
        processes.append(process)

    # 等待所有进程结束
    for process in processes:
        process.join()

    print("Cut Done")  # 打印处理完成消息
