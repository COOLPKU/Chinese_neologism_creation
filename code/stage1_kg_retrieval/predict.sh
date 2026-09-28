MODEL_PATH="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/output/final_model"
INPUT_FILE="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/data/predict_data.txt"
OUTPUT_DIR="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/predict"
OUTPUT_FILE="/mnt/largeml-train-decision-agent/liangqiliang/qwen_discriminate/predict/predictions.json"

mkdir -p $OUTPUT_DIR

deepspeed predict.py \
    --model_path $MODEL_PATH \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --use_deepspeed \
    --deepspeed_config deepspeed_inference_config.json \