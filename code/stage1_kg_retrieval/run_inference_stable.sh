#!/bin/bash

# 设置环境变量以避免NCCL超时
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT_S=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# GPU内存优化
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 模型和数据路径
MODEL_PATH="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/output/final_model"
INPUT_FILE="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/data/predict_data.txt"
OUTPUT_DIR="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/predict"
OUTPUT_FILE="/mnt/train-decision-agent/liangqiliang/qwen_discriminate/predict/predictions.json"
BATCH_SIZE=8

# 使用优化的DeepSpeed配置
DEEPSPEED_CONFIG="./deepspeed_inference_config_optimized.json"

echo "开始DeepSpeed推理..."
echo "模型路径: $MODEL_PATH"
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "批次大小: $BATCH_SIZE"
echo "DeepSpeed配置: $DEEPSPEED_CONFIG"

# 运行推理
deepspeed --num_gpus=8 \
    predict.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
    --use_deepspeed \
    --deepspeed_config "$DEEPSPEED_CONFIG"

echo "推理完成！"
