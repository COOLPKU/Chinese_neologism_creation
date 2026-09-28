MODEL_PATH="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/output/final_model"
VAL_DATA_PATH="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/data/test_data.txt"
OUTPUT_DIR="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/evaluate/final_model"
MODEL_NAME="/mnt/largeml-model-gui-agent/Qwen3-Embedding-4B"

mkdir -p $OUTPUT_DIR



deepspeed evaluation.py \
    --model_path $MODEL_PATH \
    --test_data_path $VAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --max_length 512 \
    --use_deepspeed \
    --deepspeed_config "deepspeed_inference_config.json" \
