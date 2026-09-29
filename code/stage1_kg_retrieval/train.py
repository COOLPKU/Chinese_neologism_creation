#!/usr/bin/env python3
"""
基于Qwen模型的对比学习分类器训练脚本
支持全量微调和Flash Attention优化
"""

import os
import json
import argparse
import logging
from typing import List, Tuple
import torch
from model.discriminate_model import DiscriminateTrainer, ModelConfig

# DeepSpeed支持
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed未安装，将使用标准训练。安装方法: pip install deepspeed")

# SwanLab监控
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLab未安装，将跳过实验监控功能。安装方法: pip install swanlab")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(output_path: str, num_samples: int = 1000):
    """创建示例训练数据"""
    import random
    
    # 正样本示例（标签1）
    positive_samples = [
        "这个产品质量很好，我很满意。",
        "服务态度非常棒，值得推荐。", 
        "功能强大，使用体验excellent。",
        "性价比很高，物超所值。",
        "设计精美，做工精细。",
        "客服响应及时，解决问题迅速。",
        "产品符合描述，没有任何问题。",
        "包装完好，物流速度快。",
        "使用简单，功能齐全。",
        "质量稳定，值得信赖。"
    ]
    
    # 负样本示例（标签0）
    negative_samples = [
        "产品质量有问题，不推荐购买。",
        "服务态度很差，体验不好。",
        "功能有缺陷，使用不便。",
        "价格太高，性价比不好。",
        "设计有问题，做工粗糙。",
        "客服不响应，问题无法解决。",
        "产品与描述不符，有欺骗性。",
        "包装破损，物流有问题。",
        "使用复杂，功能不完善。",
        "质量不稳定，不可靠。"
    ]
    
    data = []
    
    for i in range(num_samples // 2):
        # 正样本
        text = random.choice(positive_samples)
        data.append({"text": text, "label": 1})
        
        # 负样本
        text = random.choice(negative_samples)
        data.append({"text": text, "label": 0})
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已创建 {len(data)} 条示例数据保存到 {output_path}")

def prepare_data_from_txt(txt_path: str, json_path: str):
    """从txt文件转换为json格式"""
    data = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                parts = line.split('\t')
                if len(parts) >= 2:
                    text = parts[0].strip()
                    label = int(parts[1].strip())
                    if label in [0, 1]:
                        data.append({"text": text, "label": label})
                else:
                    logger.warning(f"第 {line_num} 行格式不正确: {line}")
            except ValueError as e:
                logger.warning(f"第 {line_num} 行标签转换失败: {line}, 错误: {e}")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已转换 {len(data)} 条数据从 {txt_path} 到 {json_path}")

def split_train_val(data_path: str, train_ratio: float = 0.8):
    """分割训练集和验证集"""
    import random
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机打乱
    random.shuffle(data)
    
    # 分割
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 保存训练集
    train_path = data_path.replace('.json', '_train.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    val_path = data_path.replace('.json', '_val.json')
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练集: {len(train_data)} 条 -> {train_path}")
    logger.info(f"验证集: {len(val_data)} 条 -> {val_path}")
    
    return train_path, val_path

def main():
    parser = argparse.ArgumentParser(description="训练基于Qwen的对比学习分类器")
    parser.add_argument("--train_data_path", type=str, help="训练数据路径 (.json或.txt格式)")
    parser.add_argument("--val_data_path", type=str, help="验证数据路径 (.json或.txt格式)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="基础模型名称")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--temperature", type=float, default=0.07, help="对比学习温度参数")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    parser.add_argument("--validate", action="store_true", help="是否进行验证")
    
    # DeepSpeed参数
    parser.add_argument("--use_deepspeed", action="store_true", help="启用DeepSpeed加速训练")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed本地rank (由DeepSpeed自动设置)")
    
    # SwanLab监控参数
    parser.add_argument("--use_swanlab", action="store_true", help="启用SwanLab实验监控")
    parser.add_argument("--experiment_name", type=str, default=None, help="SwanLab实验名称")
    parser.add_argument("--project_name", type=str, default="qwen-discriminate", help="SwanLab项目名称")
    parser.add_argument("--tags", type=str, nargs="+", default=None, help="实验标签")
    
    args = parser.parse_args()
    
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    
    # 检查数据文件
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.train_data_path}")
    if not os.path.exists(args.val_data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.val_data_path}")
    # 转换txt到json格式
    if args.train_data_path.endswith('.txt'):
        json_path = args.train_data_path.replace('.txt', '.json')
        prepare_data_from_txt(args.train_data_path, json_path)
        args.train_data_path = json_path
    if args.val_data_path.endswith('.txt'):
        json_path = args.val_data_path.replace('.txt', '.json')
        prepare_data_from_txt(args.val_data_path, json_path)
        args.val_data_path = json_path
    
    
    
    if args.validate:
        train_path, val_path = args.train_data_path, args.val_data_path
    
    # 配置模型
    config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature
    )
    
    # 打印配置信息
    logger.info("训练配置:")
    logger.info(f"  基础模型: {config.model_name}")
    logger.info(f"  最大长度: {config.max_length}")
    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  温度参数: {config.temperature}")
    logger.info(f"  DeepSpeed: {args.use_deepspeed}")
    if args.use_deepspeed:
        logger.info(f"  DeepSpeed配置: {args.deepspeed_config}")
    logger.info(f"  训练数据: {train_path}")
    logger.info(f"  验证数据: {val_path}")
    logger.info(f"  输出目录: {args.output_dir}")
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 检查是否为主进程（用于分布式训练）
    def is_main_process():
        """检查是否为主进程"""
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return os.environ.get('LOCAL_RANK', '0') == '0'
    
    # 初始化SwanLab监控
    swanlab_run = None
    if args.use_swanlab and SWANLAB_AVAILABLE and is_main_process():
        try:
            # 准备实验配置
            experiment_config = {
                "model_name": config.model_name,
                "max_length": config.max_length,
                "learning_rate": config.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "temperature": config.temperature,
                "use_deepspeed": args.use_deepspeed,
                "train_samples": len(json.load(open(train_path, 'r', encoding='utf-8'))),
                "val_samples": len(json.load(open(val_path, 'r', encoding='utf-8'))) if val_path else 0,
            }
            
            # 生成实验名称
            if not args.experiment_name:
                import datetime
                model_short = config.model_name.split('/')[-1]
                timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                args.experiment_name = f"{model_short}_{args.epochs}ep_{timestamp}"
            
            # 初始化SwanLab
            # 初始化SwanLab（API key 请通过环境变量 SWANLAB_API_KEY 提供）
            swanlab.login(api_key=os.environ.get('SWANLAB_API_KEY'))
            swanlab_run = swanlab.init(
                project=args.project_name,
                experiment_name=args.experiment_name,
                description=f"Qwen对比学习分类器训练 - {config.model_name}",
                config=experiment_config,
                tags=args.tags or ["qwen", "contrastive-learning", "classification"]
            )
            
            logger.info(f"SwanLab监控已启用 - 项目: {args.project_name}, 实验: {args.experiment_name}")
            
            # 记录数据集信息
            if val_path:
                swanlab.log({
                    "dataset/train_samples": experiment_config["train_samples"],
                    "dataset/val_samples": experiment_config["val_samples"],
                    "dataset/total_samples": experiment_config["train_samples"] + experiment_config["val_samples"]
                })
                
        except Exception as e:
            logger.warning(f"SwanLab初始化失败: {e}")
            swanlab_run = None
    elif args.use_swanlab and not SWANLAB_AVAILABLE:
        logger.warning("SwanLab未安装，无法启用监控功能")
    elif args.use_swanlab and not is_main_process():
        logger.info("非主进程，跳过SwanLab初始化")
    
    # 创建训练器
    trainer = DiscriminateTrainer(config)
    
    try:
        # 开始训练
        logger.info("开始训练模型...")
        
        # 准备训练参数
        training_kwargs = {
            'num_train_epochs': args.epochs,
            'per_device_train_batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
        }
        
        trainer.train(
            train_path, 
            val_path, 
            args.output_dir,
            use_swanlab=args.use_swanlab and SWANLAB_AVAILABLE,
            use_deepspeed=args.use_deepspeed,
            deepspeed_config=args.deepspeed_config if args.use_deepspeed else None,
            **training_kwargs
        )
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        
        
        logger.info("训练完成!")
        logger.info(f"模型已保存到: {final_model_path}")
        
        # 显示SwanLab链接
        if args.use_swanlab and SWANLAB_AVAILABLE and swanlab_run and is_main_process():
            try:
                logger.info(f"查看训练结果: {swanlab.get_run().url}")
            except:
                logger.info("可在SwanLab控制台查看训练结果")
        
    except Exception as e:
        # 记录训练失败到SwanLab
        if args.use_swanlab and SWANLAB_AVAILABLE and swanlab_run and is_main_process():
            try:
                swanlab.log({
                    'training/error_code': len(str(e))  # 使用错误消息长度作为数值指标
                })
            except:
                pass
        
        logger.error(f"训练过程中发生错误: {e}")
        raise
    
    finally:
        # 结束SwanLab运行
        if args.use_swanlab and SWANLAB_AVAILABLE and swanlab_run and is_main_process():
            try:
                swanlab.finish()
            except Exception as e:
                logger.warning(f"SwanLab结束运行失败: {e}")

def evaluate_model(model_path: str, test_data_path: str):
    """评估训练好的模型"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    # 加载模型
    config = ModelConfig()
    trainer = DiscriminateTrainer(config)
    trainer.load_model(model_path)
    
    # 加载测试数据
    test_texts, test_labels = trainer.load_data(test_data_path)
    
    # 预测
    predictions = trainer.model.predict(test_texts)
    
    # 计算指标
    print("分类报告:")
    print(classification_report(test_labels, predictions, target_names=['负例', '正例']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, predictions))
    
    return predictions

if __name__ == "__main__":
    main()
