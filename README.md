# Seg_Task
分割任务代码

当前默认超参
参数	默认值
--epochs  50
--lr  0.0002
--batch_size  16
--img_size  224
--num_classes  3
--mask_values  0,3,4
--val_freq  1（每 epoch 验证）
--save_freq  10（每 10 epoch 存一次）

训练指令：
python train_test/train.py \
  --encoder {encoder} \
  --csv_preset {preset} \
  --output_dir ./output/{encoder}_{preset}

例如:
python train_test/train.py \
  --encoder echocare \
  --csv_preset train100 \
  --output_dir ./output/echocare_train100


验证指令：
python train_test/test.py \
  --encoder {encoder} \
  --checkpoint ./output/{encoder}_{preset}/checkpoint_{encoder}_dpt_seg_best.pth \
  --csv_preset {preset} \
  --split test \
  --output_dir ./output/{encoder}_{preset}

例如：
  python train_test/test.py \
  --encoder echocare \
  --checkpoint ./output/echocare_train100/checkpoint_echocare_dpt_seg_best.pth \
  --csv_preset train100 \
  --split test \
  --output_dir ./output/echocare_train100


后续修改数据路径：
方式一：命令行指定（推荐，无需改代码）

训练和测试都支持 --csv_file，直接传入你的 CSV 绝对路径：

# 训练
python train_test/train.py \
  --encoder echocare \
  --csv_file /你的路径/dataset.csv \
  --output_dir ./output/echocare_custom

# 测试
python train_test/test.py \
  --encoder echocare \
  --checkpoint ./output/echocare_custom/checkpoint_echocare_dpt_seg_best.pth \
  --csv_file /你的路径/dataset.csv \
  --split val \
  --output_dir ./output/echocare_custom

注意：不要同时传 --csv_preset，否则 preset 会覆盖 --csv_file（见 [train_test/train.py](Seg_Task/train_test/train.py) 第 234-241 行）。



方式二：修改预设（适合固定数据集、反复使用）

预设定义在 [Seg_Task/encoder/factory.py](Seg_Task/encoder/factory.py) 的 CSV_PRESETS：

CSV_PRESETS = {
    "train10": "/sdb1/liran/downsteam_code/3VT/dataset_train10.csv",
    "train20": "/sdb1/liran/downsteam_code/3VT/dataset_train20.csv",
    "train50": "/sdb1/liran/downsteam_code/3VT/dataset_train50.csv",
    "train100": "/sdb1/liran/downsteam_code/3VT/dataset_train100.csv",
}
