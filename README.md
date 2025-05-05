# [MICCAI 2025] DT based Surgical Phase Recognition

Digital-Twin-based-Surgical-Phase-Recognition是在本地修改代码跑的
Spr是本地结果还可以，然后用于git push的本地仓库

export PYTHONPATH=$(pwd):$PYTHONPATH


### use tmux（）
1.启动对话 tmux new -s <session_name> \
2.列出所有 tmux ls
3.附加终端到已存在的对话 tmux a -t <session_name>
4.从会话中分离 Ctrl + b, d\
5.杀死会话 tmux kill-session -t <session_name>\
6.切换会话 tmux switch -t <session-name>\
7.重命名会话 tmux rename-session -t 0 <new-name>

### Training
run the following code for training
conda activate Surgformer1
cd Wenzheng/Digital-Twin-based-Surgical-Phase-Recognition
bash scripts/train/train_14ch.sh
bash scripts/test/gsvit/test_41-50_original.sh

cd root/Spr
conda activate Surgformer1
export PYTHONPATH=$(pwd):$PYTHONPATH
bash scripts/train/train_gsvit.sh

```shell
bash scripts/train/train_10ch.sh
bash scripts/train/train_baseline.sh
bash scripts/train/train_11ch.sh
```
> You need to modify **data_path**, **eval_data_path**, **output_dir** and **log_dir** according to your own setting.

> Optional settings \
> **Model**: surgformer_base surgformer_HTA surgformer_HTA_KCA \
> **Dataset**: Cholec80 AutoLaparo

### Testing

1. run the following code for testing, and get **0.txt** and **1.txt**;

cd Wenzheng/Surgformer
```shell
bash scripts/test/10ch/test_41-50_combine.sh
bash scripts/test/10ch/test_41-50_hue.sh
bash scripts/test/10ch/test_41-50_original.sh

bash scripts/test/baseline/test_41-50_baseline_combine.sh
bash scripts/test/baseline/test_41-50_baseline_hue.sh
bash scripts/test/baseline/test_41-50_baseline_original.sh

bash scripts/test/11ch/test_41-50_combine.sh
bash scripts/test/11ch/test_41-50_hue.sh
bash scripts/test/11ch/test_41-50_original.sh
```

2. Merge the files and generate separate txt file for each video;
```python
python datasets/convert_results/convert_cholec80.py
python datasets/convert_results/convert_autolaparo.py
```

3. Use [Matlab Evaluation Code](https://github.com/isyangshu/Surgformer/tree/master/evaluation_matlab) to compute metrics;



