CUDA_VISIBLE_DEVICES=1 python downstream_phase/run_phase_training.py \
--batch_size 8 \
--epochs 20 \
--save_ckpt_freq 5 \
--model surgformer_HTA_KCA \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /home/hao/dt_sam2/cholec80_41-50contrast3 \
--eval_data_path /home/hao/dt_sam2/cholec80_41-50contrast3 \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--finetune /home/hao/dt_sam2/checkpoints/3depth/mp_rank_00_model_states.pt \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /home/hao/dt_sam2/results/3depth/41-50_contrast3 \
--log_dir /home/hao/dt_sam2/results/3depth/41-50_contrast3 \
--num_workers 15 \
--channel_type pure_depth \
# --enable_deepspeed \
# --no_auto_resume