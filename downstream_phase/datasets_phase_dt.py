import os
from datasets.transforms import *
from datasets.transforms.surg_transforms import *

from datasets.phase.Cholec80_phase_dt import PhaseDataset_Cholec80_dt

def build_dataset_dt(is_train, test_mode, fps, args):
    """Load video phase recognition dataset."""

    if args.data_set == "Cholec80":
        mode = None
        anno_path = None
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
            mode = "test"  # for validation
            anno_path = os.path.join(args.data_path, "labels", "test_41-50", fps + "val_test.pickle")

        dataset = PhaseDataset_Cholec80_dt(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
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