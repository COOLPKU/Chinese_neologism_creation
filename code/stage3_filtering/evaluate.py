import torch
import numpy as np
from transformers import BertTokenizer, Trainer, TrainingArguments
import os
import argparse
from model import GlossBERT
from data_utils import WSDDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    logits = pred.predictions
    
    # 处理logits形状
    if len(logits.shape) > 1:
        logits = np.squeeze(logits)
    
    # 使用sigmoid将logits转换为概率
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    
    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the test.jsonl file."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed evaluation."
    )
    args = parser.parse_args()

    # --- Path Setup ---
    test_file = os.path.join(args.data_dir, 'test.jsonl')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found at {test_file}")

    # --- Model Loading ---
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    print(f"Loading model from {args.model_path}")
    # 首先获取训练时使用的base model路径
    # 如果config.json中没有，则默认使用bert-base-chinese
    import json
    config_path = os.path.join(args.model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        base_model = config.get('_name_or_path', 'bert-base-chinese')
    else:
        base_model = 'bert-base-chinese'
    
    print(f"Using base model: {base_model}")
    model = GlossBERT(base_model)
    
    # 加载训练好的权重
    model_file = os.path.join(args.model_path, 'model.safetensors')
    if os.path.exists(model_file):
        print(f"Loading weights from {model_file}")
        from safetensors.torch import load_file
        state_dict = load_file(model_file, device='cpu')
        model.load_state_dict(state_dict)
    else:
        # 尝试加载pytorch_model.bin
        model_file = os.path.join(args.model_path, 'pytorch_model.bin')
        if os.path.exists(model_file):
            print(f"Loading weights from {model_file}")
            state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model weights found in {args.model_path}")
    
    print("Model loaded successfully")

    # --- Dataset Loading ---
    print(f"Loading test data from {test_file}")
    test_dataset = WSDDataset(test_file, tokenizer, args.max_length)
    print(f"Test samples: {len(test_dataset)}")

    # --- Trainer Setup ---
    eval_output_dir = os.path.join(args.model_path, "eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=torch.cuda.is_available(),
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # --- Evaluate ---
    print("\n" + "="*50)
    print("Starting Evaluation...")
    print("="*50 + "\n")
    
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"{metric_name}: {value:.4f}")
    
    # 保存结果到文件
    results_file = os.path.join(eval_output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
