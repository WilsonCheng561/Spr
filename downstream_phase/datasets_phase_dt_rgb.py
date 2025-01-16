import os
from datasets.transforms import *
from datasets.transforms.surg_transforms import *

from datasets.phase.Cholec80_phase_dt_rgb import PhaseDataset_Cholec80_dt_rgb


def build_dataset_dt_rgb(is_train, test_mode, fps, args):
    """
    3 通道 RGB + 10 通道 mask + 1 通道 depth = 14 通道输入的数据集构建。
    """
    if args.data_set == "Cholec80":
        mode = None
        anno_path = None

        # 根据 is_train, test_mode 来决定 mode 以及对应的标注文件位置
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", "train_20", fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", "test_41-50", fps + "val_test.pickle"
            )
        else:
            mode = "test"  
            anno_path = os.path.join(args.data_path, "labels", "test_41-50", fps + "val_test.pickle")

        # 实例化我们新的 14 通道数据集
        dataset = PhaseDataset_Cholec80_dt_rgb(
            anno_path=anno_path,
            data_path=args.data_path,
            data_path_rgb= args.data_path_rgb, # 3通道原图路径
            mode=mode,
            data_strategy=args.data_strategy,      # online / offline
            output_mode=args.output_mode,          # key_frame / all_frame
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=False,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7
    else:
        print("Error")
        return None, None
    

    # 与原 build_dataset_dt 中的日志输出相同
    assert nb_classes == args.nb_classes
    print("%s - %s : Number of the class = %d" % (mode, fps, args.nb_classes))
    print("Data Strategy: %s" % args.data_strategy)
    print("Output Mode: %s" % args.output_mode)
    print("Cut Black: %s" % args.cut_black)
    if args.sampling_rate == 0:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Exponential Stride")
        )
    elif args.sampling_rate == -1:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Random Stride (1-5)")
        )
    elif args.sampling_rate == -2:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Incremental Stride")
        )
    else:
        print(
            "%s Frames with Temporal sample Rate %s"
            % (str(args.num_frames), str(args.sampling_rate))
        )

    return dataset, nb_classes
