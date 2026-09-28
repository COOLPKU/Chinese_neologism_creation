#!/bin/bash

# DeepSpeed训练启动脚本
# 使用方法: bash run_deepspeed.sh

# 设置参数
DATA_PATH="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/data/train_data.txt"
VAL_DATA_PATH="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/data/test_data.txt"
OUTPUT_DIR="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/output"
MODEL_NAME="/mnt/model-gui-agent/Qwen3-Embedding-4B"
DEEPSPEED_CONFIG="deepspeed_config.json"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始DeepSpeed训练..."
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模型: $MODEL_NAME"
echo "DeepSpeed配置: $DEEPSPEED_CONFIG"

# 单GPU训练
deepspeed train.py \
    --train_data_path $DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --use_deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --validate \
    --use_swanlab \
    --project_name "qwen-discriminate" \
    --experiment_name "qwen_deepspeed_$(date +%m%d_%H%M)"

echo "训练完成!"
