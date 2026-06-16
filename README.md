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
  --encoder echocare 
  --csv_preset train100 
  --output_dir ./output/echocare_train100


验证指令：
python train_test/test.py \
  --encoder {encoder} \
  --checkpoint ./output/{encoder}_{preset}/checkpoint_{encoder}_dpt_seg_best.pth \
  --csv_preset {preset} \
  --split val \
  --output_dir ./output/{encoder}_{preset}
例如：
  python train_test/test.py \
  --encoder echocare \
  --checkpoint ./output/echocare_train100/checkpoint_echocare_dpt_seg_best.pth \
  --csv_preset train100 \
  --split val \
  --output_dir ./output/echocare_train100

