import numpy as np
import os
import cv2
from tqdm import tqdm 

# 从 Cholec80 数据集中提取帧的标签并将其保存为 .pickle 格式的文件，用于模型训练或验证。
# 主要完成以下任务：

# 读取视频的帧数据及其对应的工具（Tool）和阶段（Phase）标签。
# 每隔一秒（基于帧率 25fps）提取一个帧，提取的信息包括帧的工具和阶段标签。
# 将提取的帧信息按照训练集和测试集分别保存为 .pickle 文件。


# 设置数据集根目录，包含视频文件的路径
ROOT_DIR = "/home/yangshu/Surgformer/data/Cholec80"

# 获取所有视频文件的文件名，并过滤出仅包含".mp4"格式的视频文件
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])  # 按字典序排序视频文件名

# 初始化统计帧总数的变量
FRAME_NUMBERS = 0

# 遍历所有视频文件
for video_name in VIDEO_NAMES:
    print(video_name)  # 打印当前正在处理的视频文件名
    
    # 使用OpenCV加载视频文件
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
    
    # 获取视频的帧率（FPS：每秒帧数）
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)  # 打印当前视频的帧率
    
    # 检查帧率是否为25 FPS，如果不是，则提示用户
    if fps != 25:
        print(video_name, 'not at 25fps', fps)
    
    # 初始化变量，用于控制帧读取和保存
    success = True  # 表示是否成功读取帧
    count = 0  # 当前帧的计数器

    # 设置保存当前视频帧的目录
    save_dir = './frames/' + video_name.replace('.mp4', '') + '/'  # 将.mp4替换为文件夹名
    save_dir = os.path.join(ROOT_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建目录
    
    # 循环读取视频的每一帧
    while success is True:
        # 读取视频帧
        success, image = vidcap.read()
        
        # 如果成功读取帧，则进行处理
        if success:
            # 每隔FPS帧（即每秒一帧）保存为图像文件
            if count % fps == 0:
                # 保存帧为PNG格式的图像，文件名为5位数字，零填充
                cv2.imwrite(save_dir + str(int(count // fps)).zfill(5) + '.png', image)
            count += 1  # 更新帧计数器
    
    # 释放视频资源，关闭所有相关窗口
    vidcap.release()
    cv2.destroyAllWindows()
    
    # 打印当前视频的总帧数
    print(count)
    
    # 累加当前视频的帧数到总帧数变量中
    FRAME_NUMBERS += count

# 最后打印所有视频的总帧数
print('Total Frames', FRAME_NUMBERS)
