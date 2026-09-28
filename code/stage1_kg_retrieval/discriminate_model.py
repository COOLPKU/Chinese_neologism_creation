import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# SwanLab监控
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 使用Qwen2.5作为基础模型
    max_length: int = 512
    hidden_size: int = 768
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    temperature: float = 0.07  # 对比学习温度参数
    margin: float = 0.5  # 对比学习边界
    
    def to_dict(self):
        """转换为字典格式，兼容transformers"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            'margin': self.margin
        }

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        计算对比学习损失
        Args:
            embeddings: [batch_size, hidden_size] 文本嵌入
            labels: [batch_size] 标签 (0 or 1)
        """
        # 标准化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 数值稳定性：限制相似度范围
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)
        
        # 创建标签掩码
        labels = labels.unsqueeze(0)
        positive_mask = torch.eq(labels, labels.T).float()
        
        # 对角线置0（自己与自己的相似度不考虑）
        positive_mask.fill_diagonal_(0)
        
        # 检查是否有正样本对，如果没有则返回0损失
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # 使用数值稳定的log-sum-exp技巧
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix - max_sim)
        
        # 计算正样本和总样本的指数和
        positive_sum = (exp_sim * positive_mask).sum(dim=1, keepdim=True)
        all_sum = exp_sim.sum(dim=1, keepdim=True)
        
        # 避免除零和log(0)
        positive_sum = torch.clamp(positive_sum, min=1e-8)
        all_sum = torch.clamp(all_sum, min=1e-8)
        
        # 对比损失（只对有正样本的行计算）
        valid_rows = positive_mask.sum(dim=1) > 0
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        contrastive_loss = -torch.log(positive_sum[valid_rows] / all_sum[valid_rows])
        contrastive_loss = contrastive_loss.mean()
        
        return contrastive_loss

class QwenDiscriminateModel(nn.Module):
    """基于Qwen的判别模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            pad_token='<|endoftext|>'
        )
        
        # 配置flash attention
        model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

        # 确定数据类型
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        
        # 获取隐藏层大小
        self.hidden_size = self.backbone.config.hidden_size
        
        # 投影层 - 确保使用相同的数据类型
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size)
        ).to(self.dtype)
        
        # 分类头 - 确保使用相同的数据类型
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 2)  # 二分类
        ).to(self.dtype)
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(config.temperature, config.margin)
        self.classification_loss = nn.CrossEntropyLoss()
        
    def last_token_pool(self, last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """获取最后一个有效token的隐藏状态"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def encode_text(self, input_ids, attention_mask):
        """编码文本获取嵌入"""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 对于CausalLM，需要从hidden_states中获取最后一层
        # outputs.hidden_states是一个tuple，包含每一层的隐藏状态
        # 最后一层是 hidden_states[-1]
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = self.last_token_pool(last_hidden_state, attention_mask)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask, labels=None):
        """前向传播"""
        # 获取文本嵌入
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # 确保数据类型一致
        text_embeddings = text_embeddings.to(self.dtype)
        
        # 投影到对比学习空间
        projected_embeddings = self.projection(text_embeddings)
        
        # 分类预测
        logits = self.classifier(projected_embeddings)
        
        outputs = {
            'logits': logits,
            'embeddings': projected_embeddings,
            'text_embeddings': text_embeddings
        }
        
        if labels is not None:
            # 计算分类损失
            cls_loss = self.classification_loss(logits, labels)
            
            # 计算对比学习损失
            contrastive_loss = self.contrastive_loss(projected_embeddings, labels)
            
            # 总损失
            total_loss = cls_loss + 0.5 * contrastive_loss
            
            outputs['loss'] = total_loss
            outputs['classification_loss'] = cls_loss
            outputs['contrastive_loss'] = contrastive_loss
        
        return outputs
    
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本标签"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # 分词
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 预测
                outputs = self.forward(**inputs)
                logits = outputs['logits']
                prediction = torch.argmax(logits, dim=-1).cpu().item()
                predictions.append(prediction)
        
        return predictions

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    # 调试信息
    print(f"Debug - predictions type: {type(predictions)}")
    print(f"Debug - predictions shape: {getattr(predictions, 'shape', 'No shape attr')}")
    if isinstance(predictions, (list, tuple)):
        print(f"Debug - predictions length: {len(predictions)}")
        if len(predictions) > 0:
            print(f"Debug - first prediction type: {type(predictions[0])}")
            print(f"Debug - first prediction shape: {getattr(predictions[0], 'shape', 'No shape attr')}")
    
    # 处理predictions
    if isinstance(predictions, dict):
        predictions = predictions['logits']
    elif isinstance(predictions, (list, tuple)) and len(predictions) > 0:
        # 如果predictions是列表或元组，取第一个元素（通常是logits）
        predictions = predictions[0] if isinstance(predictions[0], (np.ndarray, torch.Tensor)) else predictions
    
    # 确保predictions是numpy数组
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, (list, tuple)):
        predictions = np.array(predictions)
    
    print(f"Debug - processed predictions shape: {predictions.shape}")
    
    # 检查predictions的形状
    if len(predictions.shape) == 1:
        # 如果是1D数组，假设已经是预测的类别
        predicted_labels = predictions
    else:
        # 如果是2D数组，取argmax
        predicted_labels = np.argmax(predictions, axis=1)
    
    # 处理labels
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    elif isinstance(labels, (list, tuple)):
        labels = np.array(labels)
    
    accuracy = accuracy_score(labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted_labels, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # 检查是否为主进程
    def is_main_process():
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return os.environ.get('LOCAL_RANK', '0') == '0'
    
    # 记录到SwanLab - 只在主进程中记录
    if SWANLAB_AVAILABLE and hasattr(swanlab, 'log') and is_main_process():
        try:
            swanlab.log({
                'eval/accuracy': accuracy,
                'eval/precision': precision,
                'eval/recall': recall,
                'eval/f1': f1
            })
        except Exception as e:
            logger.warning(f"SwanLab记录评估指标失败: {e}")
    
    return metrics

class SwanLabCallback:
    """SwanLab监控回调类"""
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step_count = 0
        
    def is_main_process(self):
        """检查是否为主进程"""
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return os.environ.get('LOCAL_RANK', '0') == '0'
        
    def on_log(self, logs: Dict):
        """训练日志回调"""
        if not SWANLAB_AVAILABLE or not self.is_main_process():
            return
            
        try:
            # 过滤和重命名指标
            swanlab_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if key.startswith('train_'):
                        swanlab_logs[f'train/{key[6:]}'] = value
                    elif key.startswith('eval_'):
                        swanlab_logs[f'eval/{key[5:]}'] = value
                    elif key in ['learning_rate', 'epoch', 'step']:
                        swanlab_logs[key] = value
                    elif 'loss' in key:
                        swanlab_logs[f'train/{key}'] = value
            
            if swanlab_logs:
                swanlab.log(swanlab_logs)
                
        except Exception as e:
            logger.warning(f"SwanLab记录训练指标失败: {e}")
    
    def on_train_begin(self, model_info: Dict):
        """训练开始回调"""
        if not SWANLAB_AVAILABLE or not self.is_main_process():
            return
            

    def on_epoch_end(self, epoch: int, metrics: Dict):
        """轮次结束回调"""
        if not SWANLAB_AVAILABLE or not self.is_main_process():
            return
            
        try:
            epoch_logs = {'epoch': epoch}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    epoch_logs[f'epoch_end/{key}'] = value
            
            if epoch_logs:
                swanlab.log(epoch_logs)
                
        except Exception as e:
            logger.warning(f"SwanLab记录轮次指标失败: {e}")

class DiscriminateTrainer:
    """训练器类"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """初始化模型"""
        logger.info("正在初始化模型...")
        self.model = QwenDiscriminateModel(self.config)
        self.tokenizer = self.model.tokenizer
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info(f"模型已移至GPU: {torch.cuda.get_device_name()}")
        
    def load_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """加载训练数据"""
        texts = []
        labels = []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    texts.append(item['text'])
                    labels.append(item['label'])
        elif data_path.endswith('.txt'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        texts.append(parts[0])
                        labels.append(int(parts[1]))
        
        logger.info(f"加载了 {len(texts)} 条训练数据")
        return texts, labels
    
    def is_main_process(self):
        """检查是否为主进程"""
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return os.environ.get('LOCAL_RANK', '0') == '0'

    def train(self, train_data_path: str, val_data_path: str = None, output_dir: str = "./checkpoints", 
              use_swanlab: bool = False, use_deepspeed: bool = False, deepspeed_config: str = None, **training_kwargs):
        """训练模型"""
        # 设置模型
        self.setup_model()
        
        # 加载训练数据
        train_texts, train_labels = self.load_data(train_data_path)
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        
        # 加载验证数据
        eval_dataset = None
        if val_data_path:
            val_texts, val_labels = self.load_data(val_data_path)
            eval_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        
        # SwanLab回调 - 只在主进程中启用
        swanlab_callback = SwanLabCallback() if (use_swanlab and self.is_main_process()) else None
        
        # 记录模型信息到SwanLab
        if swanlab_callback:

            
            # 记录数据集信息
            if SWANLAB_AVAILABLE and self.is_main_process():
                try:
                    swanlab.log({
                        'dataset/train_samples': len(train_texts),
                        'dataset/val_samples': len(val_texts) if val_data_path else 0,
                    })
                except Exception as e:
                    logger.warning(f"SwanLab记录模型信息失败: {e}")
        
        # 训练参数 (支持自定义参数)
        default_training_args = {
            'output_dir': output_dir,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,  # 减小批次大小以提高稳定性
            'per_device_eval_batch_size': 8,   # 减小评估批次大小
            'gradient_accumulation_steps': 4,  # 增加梯度累积以补偿小批次
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'learning_rate': min(self.config.learning_rate, 1e-5),  # 限制最大学习率
            'logging_dir': f"{output_dir}/logs",
            'logging_steps': 100,
            'save_total_limit': 3,
            'fp16': torch.cuda.is_available(),
            'max_grad_norm': 1.0,  # 添加梯度裁剪
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'report_to': ["none"],  # 禁用默认的日志记录
        }
        
        # 根据是否有验证集来设置评估和保存策略
        if eval_dataset:
            default_training_args.update({
                'eval_strategy': 'steps',
                'eval_steps': 500,
                'save_strategy': 'steps',
                'save_steps': 500,  # 与eval_steps保持一致
                'load_best_model_at_end': True,
                'metric_for_best_model': "f1",
            })
        else:
            default_training_args.update({
                'eval_strategy': 'no',
                'save_strategy': 'steps',
                'save_steps': 1000,
                'load_best_model_at_end': False,
            })
        
        # DeepSpeed配置
        if use_deepspeed:
            if deepspeed_config and os.path.exists(deepspeed_config):
                default_training_args['deepspeed'] = deepspeed_config
                logger.info(f"使用DeepSpeed配置文件: {deepspeed_config}")
            else:
                # 使用默认DeepSpeed配置
                default_training_args['deepspeed'] = {
                    "train_batch_size": "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                    "gradient_accumulation_steps": "auto",
                    "optimizer": {
                        "type": "AdamW",
                        "params": {
                            "lr": "auto",
                            "betas": "auto",
                            "eps": "auto",
                            "weight_decay": "auto"
                        }
                    },
                    "scheduler": {
                        "type": "WarmupLR",
                        "params": {
                            "warmup_min_lr": "auto",
                            "warmup_max_lr": "auto",
                            "warmup_num_steps": "auto"
                        }
                    },
                    "zero_optimization": {
                        "stage": 2,
                        "allgather_partitions": True,
                        "allgather_bucket_size": 2e8,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 2e8,
                        "contiguous_gradients": True
                    },
                    "fp16": {
                        "enabled": True,
                        "loss_scale": 0,  # 动态损失缩放
                        "initial_scale_power": 16,  # 初始缩放因子 2^16
                        "loss_scale_window": 1000,  # 缩放窗口
                        "hysteresis": 2,  # 滞后参数
                        "min_loss_scale": 1  # 最小损失缩放，避免降到0
                    },
                    "gradient_clipping": 1.0,
                    "steps_per_print": 100
                }
                logger.info("使用默认DeepSpeed配置")
        
        # 更新自定义参数
        default_training_args.update(training_kwargs)
        
        training_args = TrainingArguments(**default_training_args)
        
        # 数据整理器
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建自定义Trainer类以支持SwanLab
        class CustomTrainer(Trainer):
            def __init__(self, swanlab_callback=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.swanlab_callback = swanlab_callback
                
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                """重写prediction_step以确保输出格式正确"""
                inputs = self._prepare_inputs(inputs)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # 提取logits和loss
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits')
                        loss = outputs.get('loss')
                    else:
                        logits = outputs
                        loss = None
                    
                    # 确保logits是tensor
                    if logits is not None:
                        logits = logits.detach()
                    
                    # 获取labels
                    labels = inputs.get('labels')
                    if labels is not None:
                        labels = labels.detach()
                
                return (loss, logits, labels)
                
            def log(self, logs: Dict[str, float], start_time=None, **kwargs) -> None:
                """重写log方法以支持SwanLab"""
                # 调用父类的log方法，传递所有参数
                if start_time is not None:
                    super().log(logs, start_time, **kwargs)
                else:
                    super().log(logs, **kwargs)
                    
                # 记录到SwanLab
                if self.swanlab_callback:
                    self.swanlab_callback.on_log(logs)
            
            def on_epoch_end(self):
                """轮次结束时的回调"""
                super().on_epoch_end()
                if self.swanlab_callback and hasattr(self, 'state'):
                    metrics = getattr(self.state, 'log_history', [])
                    if metrics:
                        latest_metrics = metrics[-1]
                        self.swanlab_callback.on_epoch_end(
                            epoch=int(latest_metrics.get('epoch', 0)),
                            metrics=latest_metrics
                        )
        
        # 创建训练器
        trainer = CustomTrainer(
            swanlab_callback=swanlab_callback,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info("开始训练...")
        
        # 记录训练开始
        if swanlab_callback and SWANLAB_AVAILABLE and self.is_main_process():
            try:
                swanlab.log({
                    'training/epochs': training_args.num_train_epochs,
                    'training/batch_size': training_args.per_device_train_batch_size,
                    'training/learning_rate': training_args.learning_rate,
                })
            except Exception as e:
                logger.warning(f"SwanLab记录训练开始失败: {e}")
        
        trainer.train()
        
        # 记录训练完成
        if swanlab_callback and SWANLAB_AVAILABLE and self.is_main_process():
            try:
                swanlab.log({
                    'training/final_step': trainer.state.global_step,
                })
            except Exception as e:
                logger.warning(f"SwanLab记录训练完成失败: {e}")
        
        # 保存模型
        trainer.save_model()
        logger.info(f"模型已保存到 {output_dir}")
        
        return trainer
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, os.path.join(save_path, 'model.pt'))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"模型已保存到 {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        import torch.serialization
        
        # 检查模型文件是否存在
        model_file = os.path.join(load_path, 'model.pt')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        # 加载检查点
        # 设置weights_only=False以兼容PyTorch 2.6+版本
        # 这是安全的，因为我们信任自己训练的模型文件
        try:
            # 首先尝试使用weights_only=False（推荐方式）
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        except TypeError:
            # 兼容旧版本PyTorch（没有weights_only参数）
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        except Exception as e:
            # 如果weights_only=False仍然失败，尝试添加安全全局变量
            try:
                # 添加ModelConfig为安全的全局变量
                with torch.serialization.safe_globals([ModelConfig]):
                    checkpoint = torch.load(model_file, map_location='cpu')
            except Exception as e2:
                logger.error(f"模型加载失败，尝试了多种方法: {e}, {e2}")
                raise e2
        
        # 初始化模型
        self.config = checkpoint['config']
        self.setup_model()
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已从 {load_path} 加载")

if __name__ == "__main__":
    # 示例使用
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_length=512,
        learning_rate=2e-5
    )
    
    trainer = DiscriminateTrainer(config)
    
    # 训练示例（需要准备训练数据）
    # trainer.train("train_data.json", "val_data.json", "./checkpoints")
