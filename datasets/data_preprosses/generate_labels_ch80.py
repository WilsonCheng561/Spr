import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm  

# 从 Cholec80 数据集中提取帧的标签并将其保存为 .pickle 格式的文件，用于模型训练或验证。
# 主要完成以下任务：

# 读取视频的帧数据及其对应的工具（Tool）和阶段（Phase）标签。
# 每隔一秒（基于帧率 25fps）提取一个帧，提取的信息包括帧的工具和阶段标签。
# 将提取的帧信息按照训练集和测试集分别保存为 .pickle 文件。


def main():
    # 数据集根目录
    ROOT_DIR = "/mnt/disk0/haoding/cholec80"

    # 获取所有视频文件名并按视频编号排序
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'videos'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

    # 定义训练集和测试集视频编号
    TEST_NUMBERS = np.arange(41, 51).tolist()  # 测试集编号
    TRAIN_NUMBERS = np.array([])  # 训练集编号（此处为空）

    # 初始化帧数计数器
    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    # 初始化保存的字典
    train_pkl = dict()
    test_pkl = dict()

    # 全局帧唯一标识符
    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0

    # 定义阶段名称到ID的映射
    phase2id = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6
    }

    # 遍历每个视频
    for video_name in VIDEO_NAMES[:-1]:  # 忽略最后一个视频
        video_id = video_name.replace('.mp4', '')  # 视频ID
        vid_id = int(video_id.replace("video", ""))  # 提取视频编号

        # 根据视频编号确定是否为训练集或测试集
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 打开视频文件
        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, './videos/' + video_name))
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        if fps != 25:  # 检查帧率是否为25fps
            print(video_name, 'not at 25fps', fps)
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数

        # 打开工具标签文件
        tool_path = os.path.join(ROOT_DIR, 'tool_annotations', video_name.replace('.mp4', '-tool.txt'))
        tool_file = open(tool_path, 'r')
        tool = tool_file.readline().strip().split()
        tool_name = tool[1:]  # 工具名称
        tool_dict = dict()  # 工具标签字典
        while tool:
            tool = tool_file.readline().strip().split()
            if len(tool) > 0:
                tool = list(map(int, tool))  # 将工具标签转换为整数
                tool_dict[str(tool[0])] = tool[1:]  # 保存帧号对应的工具标签

        # 打开阶段标签文件
        phase_path = os.path.join(ROOT_DIR, 'phase_annotations', video_name.replace('.mp4', '-phase.txt'))
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]  # 跳过标题行

        # 初始化帧信息列表
        frame_infos = list()
        frame_id_ = 0

        # 遍历视频的每一帧
        for frame_id in tqdm(range(0, int(frames))):
            if frame_id % fps == 0:  # 每隔1秒处理一帧
                info = dict()  # 初始化帧信息字典
                info['unique_id'] = unique_id
                info['frame_id'] = frame_id // fps  # 当前帧编号
                assert frame_id // fps == frame_id_
                info['video_id'] = video_id

                # 工具标签
                if str(frame_id) in tool_dict:
                    info['tool_gt'] = tool_dict[str(frame_id)]
                else:
                    info['tool_gt'] = None

                # 阶段标签
                phase = phase_results[frame_id].strip().split()
                assert int(phase[0]) == frame_id  # 确保帧号一致
                phase_id = phase2id[phase[1]]
                info['phase_gt'] = phase_id
                info['phase_name'] = phase[1]

                # 添加其他帧信息
                info['fps'] = 1
                info['original_frames'] = int(frames)
                info['frames'] = int(frames) // fps

                frame_infos.append(info)
                unique_id += 1
                frame_id_ += 1

        # 保存到训练集或测试集字典中
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += frames
            unique_id_train = unique_id
            print("train", vid_id, video_id)
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += frames
            unique_id_test = unique_id
            print("test", vid_id, video_id)

    # 保存测试集标签
    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test_41-50')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpsval_test.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    # 打印帧数统计信息
    print('TRAIN Frames', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frames', TEST_FRAME_NUMBERS, unique_id_test)

if __name__ == '__main__':
    main()
