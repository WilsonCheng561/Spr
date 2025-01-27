CUDA_VISIBLE_DEVICES=0 python downstream_phase/run_dt_rgb.py \
--batch_size 8 \
--epochs 20 \
--save_ckpt_freq 5 \
--model surgformer_HTA_KCA_dt_rgb \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /mnt/disk0/haoding/cholec80_original \
--data_path_rgb /mnt/disk0/haoding/cholec80_original \
--eval_data_path /mnt/disk0/haoding/cholec80_original \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--finetune /mnt/disk0/haoding/wz_results/10mask_1depth_rgb/results2/surgformer_HTA_KCA_dt_rgb_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-9/mp_rank_00_model_states.pt \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /mnt/disk0/haoding/wz_results/10mask_1depth_rgb/results1/evaluate/cholec80_41-50_14ch_original \
--log_dir /mnt/disk0/haoding/wz_results/10mask_1depth_rgb/results1/evaluate/cholec80_41-50_14ch_original \
--num_workers 15 \
--no_auto_resume \
#--dist_eval \
#--enable_deepspeed \