@echo off
REM DeepSpeed训练脚本 - Windows版本
REM 使用多GPU训练GlossBERT模型

REM 设置参数
set DATA_DIR=data
set OUTPUT_DIR=output
set MODEL_NAME=bert-base-chinese
set NUM_GPUS=2
set BATCH_SIZE=8
set EPOCHS=5
set MAX_LENGTH=256
set LEARNING_RATE=2e-5

REM 使用deepspeed启动训练
deepspeed --num_gpus=%NUM_GPUS% train.py ^
    --model_name_or_path %MODEL_NAME% ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --num_train_epochs %EPOCHS% ^
    --per_device_train_batch_size %BATCH_SIZE% ^
    --per_device_eval_batch_size 16 ^
    --learning_rate %LEARNING_RATE% ^
    --max_length %MAX_LENGTH% ^
    --warmup_ratio 0.1 ^
    --weight_decay 0.01 ^
    --logging_steps 10 ^
    --deepspeed ds_config.json

echo.
echo 训练完成！
pause
