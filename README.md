# Seg_Task

## 当前默认超参
```bash
参数	默认值
--epochs  50
--lr  0.0002
--batch_size  16
--img_size  224
--num_classes  3
--mask_values  0,3,4
--val_freq  1（每 epoch 验证）
--save_freq  10（每 10 epoch 存一次）
```


## 训练指令：
```bash
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
```

## 训练数据路径修改说明

Seg_Task 不直接配置"数据文件夹"，而是通过 **CSV 索引文件** 加载数据。CSV 需包含三列：`image_path`、`mask_path`、`split`（值为 `train` / `val` / `test`）。

数据加载逻辑在 `dataset/csv_seg_dataset.py`：训练时读取 `split=train`，验证时读取 `split=val`，测试时由 `--split` 指定。

### 方式一：命令行指定（推荐，无需改代码）

训练和测试都支持 `--csv_file`，直接传入 CSV 绝对路径：

```bash
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

# 例如：
cd /sdb1/liran/Seg_Task

python train_test/train.py \
  --encoder echocare \
  --csv_file /sdb1/liran/Downstream_task/4CH/dataset_4ch_train100.csv \
  --output_dir ./output/echocare_4ch_train100

python train_test/test.py \
  --encoder echocare \
  --checkpoint ./output/echocare_4ch_train100/checkpoint_echocare_dpt_seg_best.pth \
  --csv_file /sdb1/liran/Downstream_task/4CH/dataset_4ch_train100.csv \
  --split test \
  --output_dir ./output/echocare_4ch_train100
```
注意：**不要同时传 `--csv_preset`**，否则 preset 会覆盖 `--csv_file`（见 `train_test/train.py` 中 `resolve_csv_file`）。


### 方式二：修改预设（适合固定数据集、反复使用）

预设定义在 `encoder/factory.py` 的 `CSV_PRESETS`：

```python
CSV_PRESETS = {
    "train10": "/sdb1/liran/downsteam_code/3VT/dataset_train10.csv",
    "train20": "/sdb1/liran/downsteam_code/3VT/dataset_train20.csv",
    "train50": "/sdb1/liran/downsteam_code/3VT/dataset_train50.csv",
    "train100": "/sdb1/liran/downsteam_code/3VT/dataset_train100.csv",
}
```

你可以：
- 改已有 preset 的路径（如把 `train100` 指向新 CSV）
- 新增 preset（如 `"train200": "/path/to/dataset_train200.csv"`），同时在 `train_test/train.py` 和 `train_test/test.py` 的 `--csv_preset` choices 里加上新名称

之后继续用上面的训练/验证指令，将 `--csv_preset` 换成对应名称即可。
