import torch
import numpy as np
from transformers import BertTokenizer, TrainingArguments, Trainer
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
        "--model_name_or_path",
        type=str,
        default='bert-base-chinese',
        help="Path to pre-trained model or shortcut name."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the .jsonl files for the task."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup ratio."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to apply."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default='./ds_config.json',
        help="Path to deepspeed config file."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training passed by deepspeed."
    )
    args = parser.parse_args()

    # --- Path Setup ---
    train_file = os.path.join(args.data_dir, 'train.jsonl')
    dev_file = os.path.join(args.data_dir, 'dev.jsonl')
    
    # 验证文件存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(dev_file):
        raise FileNotFoundError(f"Development file not found: {dev_file}")
    
    # --- Initialization ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    print(f"Loading model from {args.model_name_or_path}")
    model = GlossBERT(args.model_name_or_path)
    
    print(f"Loading training data from {train_file}")
    train_dataset = WSDDataset(train_file, tokenizer, args.max_length)
    print(f"Training samples: {len(train_dataset)}")
    
    print(f"Loading development data from {dev_file}")
    dev_dataset = WSDDataset(dev_file, tokenizer, args.max_length)
    print(f"Development samples: {len(dev_dataset)}")

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        # 评估和保存策略：每个epoch结束后评估和保存
        eval_strategy='epoch',
        save_strategy='epoch',
        # 保存最佳模型（基于accuracy）
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        # 只保留最佳模型，删除其他checkpoint
        save_total_limit=3,
        # 混合精度训练
        fp16=torch.cuda.is_available(),
        # 禁用外部日志工具
        report_to='none',
        # DeepSpeed配置
        deepspeed=args.deepspeed if args.deepspeed else None,
    )

    # --- Trainer Setup ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()

    # --- Save Best Model ---
    print("\n" + "="*50)
    print("Training completed! Saving best model...")
    print("="*50 + "\n")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # --- Final Evaluation ---
    print("\n" + "="*50)
    print("Running final evaluation on dev set...")
    print("="*50 + "\n")
    
    final_metrics = trainer.evaluate()
    
    print("\nFinal Evaluation Results:")
    print("-" * 40)
    for key, value in final_metrics.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"{metric_name}: {value:.4f}")
    
    print(f"\nBest model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
