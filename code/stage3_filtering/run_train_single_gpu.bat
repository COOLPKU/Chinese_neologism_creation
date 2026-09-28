@echo off
REM 单GPU训练脚本 - Windows版本
REM 不使用DeepSpeed的简单训练

REM 设置参数
set DATA_DIR=data
set OUTPUT_DIR=output
set MODEL_NAME=bert-base-chinese
set BATCH_SIZE=16
set EPOCHS=5
set MAX_LENGTH=256
set LEARNING_RATE=2e-5

REM 直接运行训练
python train.py ^
    --model_name_or_path %MODEL_NAME% ^
    --data_dir %DATA_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --num_train_epochs %EPOCHS% ^
    --per_device_train_batch_size %BATCH_SIZE% ^
    --per_device_eval_batch_size 32 ^
    --learning_rate %LEARNING_RATE% ^
    --max_length %MAX_LENGTH% ^
    --warmup_ratio 0.1 ^
    --weight_decay 0.01 ^
    --logging_steps 10

echo.
echo 训练完成！
pause
