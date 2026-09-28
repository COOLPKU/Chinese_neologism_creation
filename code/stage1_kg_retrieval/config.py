"""
模型配置文件
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_length: int = 512
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # 对比学习参数
    temperature: float = 0.07
    margin: float = 0.5
    contrastive_weight: float = 0.5  # 对比学习损失权重
    
    # 训练参数
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    deepspeed: Optional[str] = None  # DeepSpeed配置文件路径
    
    # 批次配置
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # 保存和日志
    output_dir: str = "./checkpoints"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # 评估配置
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    
    # 数据配置
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    def __post_init__(self):
        """后处理配置"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 自动检测是否使用混合精度
        if self.bf16 and self.fp16:
            self.fp16 = False  # bf16优先级更高
        
        # 根据GPU内存调整批次大小
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory < 12:  # 小于12GB显存
                    self.per_device_train_batch_size = max(2, self.per_device_train_batch_size // 2)
                    self.gradient_accumulation_steps *= 2
                    print(f"检测到GPU内存较小({gpu_memory:.1f}GB)，调整批次大小为{self.per_device_train_batch_size}")
        except ImportError:
            pass

# 预设配置
class PresetConfigs:
    """预设配置类"""
    
    @staticmethod
    def get_small_config():
        """小模型配置（适合GPU内存较小的情况）"""
        return TrainingConfig(
            model_name="Qwen/Qwen2-1.5B-Instruct",
            max_length=256,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            hidden_size=512,
        )
    
    @staticmethod
    def get_medium_config():
        """中等模型配置"""
        return TrainingConfig(
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_length=512,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            hidden_size=768,
        )
    
    @staticmethod
    def get_large_config():
        """大模型配置（需要大GPU内存）"""
        return TrainingConfig(
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            max_length=1024,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            hidden_size=1024,
        )
    
    @staticmethod
    def get_fast_config():
        """快速训练配置（用于测试）"""
        return TrainingConfig(
            model_name="Qwen/Qwen2-1.5B-Instruct",
            max_length=128,
            per_device_train_batch_size=16,
            num_train_epochs=1,
            eval_steps=100,
            save_steps=200,
        )

# 环境配置
def setup_environment():
    """设置训练环境"""
    import os
    import logging
    
    # 设置环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 便于调试CUDA错误
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

def get_model_info(model_name: str):
    """获取模型信息"""
    model_info = {
        "Qwen/Qwen2-1.5B-Instruct": {
            "parameters": "1.5B",
            "hidden_size": 1536,
            "vocab_size": 151936,
            "memory_requirement": "6GB+"
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
            "parameters": "7B", 
            "hidden_size": 3584,
            "vocab_size": 151936,
            "memory_requirement": "16GB+"
        },
        "Qwen/Qwen2.5-Coder-14B-Instruct": {
            "parameters": "14B",
            "hidden_size": 5120,
            "vocab_size": 151936,
            "memory_requirement": "32GB+"
        }
    }
    
    return model_info.get(model_name, {
        "parameters": "Unknown",
        "hidden_size": 768,
        "vocab_size": 151936,
        "memory_requirement": "Unknown"
    })
