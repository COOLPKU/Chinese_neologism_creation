@echo off
REM Windows单机多卡推理脚本

REM 配置参数
set NUM_GPUS=4
set INPUT_DIR=.\corpus_filter_output
set MODEL_PATH=..\checkpoints\best_model
set OUTPUT_DIR=.\filtered_results
set BATCH_SIZE=64
set MAX_LENGTH=512
set THRESHOLD=0.5
set NUM_WORKERS=4

REM 运行推理
torchrun --nproc_per_node=%NUM_GPUS% ^
    inference_distributed.py ^
    --input_dir %INPUT_DIR% ^
    --model_path %MODEL_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --max_length %MAX_LENGTH% ^
    --threshold %THRESHOLD% ^
    --num_workers %NUM_WORKERS%

pause
