@echo off
REM DeepSpeed训练启动脚本 (Windows)
REM 使用方法: run_deepspeed.bat

REM 设置参数
set DATA_PATH=.\data\sample_data.json
set OUTPUT_DIR=.\checkpoints\deepspeed_training
set MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
set DEEPSPEED_CONFIG=deepspeed_config.json

REM 创建输出目录
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo 开始DeepSpeed训练...
echo 数据路径: %DATA_PATH%
echo 输出目录: %OUTPUT_DIR%
echo 模型: %MODEL_NAME%
echo DeepSpeed配置: %DEEPSPEED_CONFIG%

REM 单GPU训练
deepspeed --num_gpus=1 train.py ^
    --data_path %DATA_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --model_name %MODEL_NAME% ^
    --use_deepspeed ^
    --deepspeed_config %DEEPSPEED_CONFIG% ^
    --batch_size 4 ^
    --learning_rate 2e-5 ^
    --epochs 3 ^
    --validate ^
    --use_swanlab ^
    --project_name "qwen-deepspeed" ^
    --experiment_name "qwen_deepspeed_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%"

echo 训练完成!
pause
