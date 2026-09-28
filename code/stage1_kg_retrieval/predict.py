#!/usr/bin/env python3
"""
模型推理脚本
用于加载训练好的模型并进行预测
支持DeepSpeed多卡推理加速
"""

import os
import json
import argparse
import torch
import logging
from typing import List, Dict, Any
from model.discriminate_model import DiscriminateTrainer, ModelConfig

# DeepSpeed支持
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed未安装，将使用标准推理。安装方法: pip install deepspeed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """模型预测器 - 支持DeepSpeed多卡推理"""
    
    def __init__(self, model_path: str, use_deepspeed: bool = False, deepspeed_config: str = None):
        self.model_path = model_path
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.trainer = None
        
        # 检查DeepSpeed可用性
        if use_deepspeed and not DEEPSPEED_AVAILABLE:
            logger.warning("DeepSpeed未安装，将使用标准推理")
            self.use_deepspeed = False
        
        # 初始化分布式环境（如果使用DeepSpeed）
        if self.use_deepspeed:
            self._init_distributed()
        
        self.load_model()
    
    def _init_distributed(self):
        """初始化分布式环境"""
        try:
            # 设置更长的NCCL超时时间和优化配置
            os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '3600')  # 1小时
            os.environ.setdefault('TORCH_NCCL_ENABLE_MONITORING', '0')  # 禁用监控
            os.environ.setdefault('NCCL_TIMEOUT_S', '3600')  # 1小时超时
            os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # 阻塞等待
            os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # 异步错误处理
            os.environ.setdefault('NCCL_DEBUG', 'INFO')  # 调试信息
            os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用InfiniBand
            os.environ.setdefault('NCCL_P2P_DISABLE', '1')  # 禁用P2P
            
            # 初始化进程组时设置更长的超时
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl",
                    timeout=torch.distributed.default_pg_timeout * 6  # 6倍默认超时
                )
            
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            
            # 设置CUDA内存策略
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制内存使用
            
            logger.info(f"初始化分布式环境，local_rank: {local_rank}")
        except Exception as e:
            logger.warning(f"分布式环境初始化失败: {e}")
            self.use_deepspeed = False
    
    def is_main_process(self) -> bool:
        """检查是否为主进程"""
        if self.use_deepspeed:
            return int(os.environ.get('LOCAL_RANK', 0)) == 0
        return True
    
    def load_model(self):
        """加载训练好的模型"""
        if self.is_main_process():
            logger.info(f"正在加载模型: {self.model_path}")
        
        # 检查模型文件
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 加载模型
        config = ModelConfig()
        self.trainer = DiscriminateTrainer(config)
        
        if self.use_deepspeed:
            self._load_model_with_deepspeed()
        else:
            self.trainer.load_model(self.model_path)
        
        if self.is_main_process():
            logger.info("模型加载完成")
    
    def _load_model_with_deepspeed(self):
        """使用DeepSpeed加载模型"""
        try:
            # 首先正常加载模型
            self.trainer.load_model(self.model_path)
            
            # 然后使用DeepSpeed包装模型
            if self.deepspeed_config and os.path.exists(self.deepspeed_config):
                with open(self.deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
            else:
                # 使用优化的推理配置 - 降低通信开销
                ds_config = {
                    "fp16": {
                        "enabled": True
                    },
                    "zero_optimization": {
                        "stage": 3,
                        "offload_param": {
                            "device": "cpu",
                            "pin_memory": True
                        },
                        "overlap_comm": False,  # 禁用通信重叠以避免超时
                        "contiguous_gradients": False,
                        "reduce_bucket_size": 5e7,  # 减小bucket size
                        "stage3_prefetch_bucket_size": 5e6,  # 减小prefetch bucket
                        "stage3_param_persistence_threshold": 1e4,
                        "sub_group_size": 5e7,  # 减小subgroup size
                        "stage3_max_live_parameters": 5e7,
                        "stage3_max_reuse_distance": 5e7,
                        "stage3_gather_16bit_weights_on_model_save": False
                    },
                    "train_micro_batch_size_per_gpu": 4,  # 减小batch size
                    "gradient_accumulation_steps": 1,
                    "communication_data_type": "fp16",
                    "bucket_cap_mb": 10,  # 减小bucket容量
                    "allgather_bucket_size": 5e7,  # 添加allgather设置
                    "reduce_scatter_bucket_size": 5e7  # 添加reduce_scatter设置
                }
            
            # 使用DeepSpeed初始化模型
            self.trainer.model, _, _, _ = deepspeed.initialize(
                model=self.trainer.model,
                config=ds_config
            )
            
            if self.is_main_process():
                logger.info("成功使用DeepSpeed加载模型")
            
        except Exception as e:
            logger.warning(f"DeepSpeed模型加载失败，回退到标准模式: {e}")
            self.use_deepspeed = False
            # 如果DeepSpeed加载失败，重新尝试标准加载
            try:
                self.trainer.load_model(self.model_path)
            except Exception as e2:
                logger.error(f"标准模式模型加载也失败: {e2}")
                raise e2
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """预测单个文本"""
        if self.trainer is None:
            raise ValueError("模型未加载")
        
        # 预测
        predictions = self.trainer.model.predict([text])
        prediction = predictions[0]
        
        # 获取概率分布
        with torch.no_grad():
            inputs = self.trainer.tokenizer(
                text,
                max_length=self.trainer.config.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.trainer.model.forward(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        result = {
            'text': text,
            'prediction': int(prediction),
            'label': '正例' if prediction == 1 else '负例',
            'confidence': float(max(probabilities)),
            'probabilities': {
                '负例': float(probabilities[0]),
                '正例': float(probabilities[1])
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """真正的批量预测 - 支持分批处理以避免内存问题"""
        if self.trainer is None:
            raise ValueError("模型未加载")
        
        results = []
        total_texts = len(texts)
        
        if self.is_main_process():
            logger.info(f"开始批量预测 {total_texts} 条文本，批次大小: {batch_size}")
        
        # 分批处理
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 在多GPU环境下添加同步点 - 使用更安全的同步方式
                if self.use_deepspeed and torch.distributed.is_initialized():
                    try:
                        # 使用更短的超时进行barrier
                        torch.distributed.barrier(timeout=torch.timedelta(seconds=300))
                    except Exception as barrier_e:
                        logger.warning(f"Barrier超时，跳过同步: {barrier_e}")
                
                # 批量tokenize
                with torch.no_grad():
                    inputs = self.trainer.tokenizer(
                        batch_texts,
                        max_length=self.trainer.config.max_length,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # 批量预测
                    outputs = self.trainer.model.forward(**inputs)
                    logits = outputs['logits']
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    
                    # 组织结果
                    for j, (text, pred, probs) in enumerate(zip(batch_texts, predictions, probabilities)):
                        result = {
                            'text': text,
                            'prediction': int(pred),
                            'label': '正例' if pred == 1 else '负例',
                            'confidence': float(max(probs)),
                            'probabilities': {
                                '负例': float(probs[0]),
                                '正例': float(probs[1])
                            }
                        }
                        results.append(result)
                
                # 定期清理GPU缓存，但减少同步频率
                if torch.cuda.is_available() and i % (batch_size * 50) == 0:  # 减少清理频率
                    torch.cuda.empty_cache()
                    # 只在必要时同步，避免频繁barrier导致超时
                    if self.use_deepspeed and torch.distributed.is_initialized() and i % (batch_size * 100) == 0:
                        try:
                            torch.distributed.barrier(timeout=torch.timedelta(seconds=60))
                        except Exception as sync_e:
                            logger.warning(f"定期同步失败，继续处理: {sync_e}")
                
                # 进度报告
                if self.is_main_process() and (i + batch_size) % (batch_size * 50) == 0:  # 更频繁的进度报告
                    processed = min(i + batch_size, total_texts)
                    logger.info(f"已处理 {processed}/{total_texts} ({processed/total_texts*100:.1f}%)")
                    
            except Exception as e:
                if self.is_main_process():
                    logger.error(f"批次 {i//batch_size + 1} 处理失败: {e}")
                # 如果批量失败，尝试逐个处理这个批次
                for text in batch_texts:
                    try:
                        result = self.predict_single(text)
                        results.append(result)
                    except Exception as e2:
                        if self.is_main_process():
                            logger.warning(f"跳过失败的文本: {text[:50]}... 错误: {e2}")
                        # 添加失败占位符
                        results.append({
                            'text': text,
                            'prediction': 0,
                            'label': '预测失败',
                            'confidence': 0.0,
                            'probabilities': {'负例': 0.5, '正例': 0.5}
                        })
        
        # 最终同步 - 使用可选同步
        if self.use_deepspeed and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier(timeout=torch.timedelta(seconds=120))
            except Exception as final_sync_e:
                logger.warning(f"最终同步失败，但预测已完成: {final_sync_e}")
        
        if self.is_main_process():
            logger.info(f"批量预测完成，共处理 {len(results)} 条文本")
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str = None, batch_size: int = 32):
        """从文件读取文本进行预测"""
        # 读取输入文件
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'text' in data[0]:
                    texts = [item['text'] for item in data]
                else:
                    texts = [str(item) for item in data]
            else:
                texts = [str(data)]
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("不支持的文件格式，仅支持 .json 和 .txt")
        
        if self.is_main_process():
            logger.info(f"从 {input_file} 读取了 {len(texts)} 条文本")
        
        # 批量预测
        results = self.predict_batch(texts, batch_size=batch_size)
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"预测结果已保存到 {output_file}")
        
        return results

def interactive_mode(predictor: ModelPredictor):
    """交互式预测模式"""
    print("=== 交互式预测模式 ===")
    print("输入文本进行预测，输入 'quit' 退出")
    print()
    
    while True:
        try:
            text = input("请输入文本: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = predictor.predict_single(text)
            
            print(f"预测结果: {result['label']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"概率分布: 负例 {result['probabilities']['负例']:.4f}, 正例 {result['probabilities']['正例']:.4f}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"预测失败: {e}")
    
    print("退出交互模式")

def main():
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument("--model_path", required=True, help="训练好的模型路径")
    parser.add_argument("--input_file", help="输入文件路径 (.json 或 .txt)")
    parser.add_argument("--output_file", help="输出文件路径 (.json)")
    parser.add_argument("--text", help="直接输入要预测的文本")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    
    # DeepSpeed参数
    parser.add_argument("--use_deepspeed", action="store_true", help="启用DeepSpeed多卡推理")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed本地rank (由DeepSpeed自动设置)")
    
    args = parser.parse_args()
    
    # 创建预测器
    try:
        predictor = ModelPredictor(
            model_path=args.model_path,
            use_deepspeed=args.use_deepspeed,
            deepspeed_config=args.deepspeed_config
        )
    except Exception as e:
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            logger.error(f"模型加载失败: {e}")
        return
    
    # 只有主进程执行预测逻辑
    if predictor.is_main_process():
        # 根据参数选择运行模式
        if args.interactive:
            interactive_mode(predictor)
        
        elif args.text:
            # 单文本预测
            result = predictor.predict_single(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.input_file:
            # 文件预测
            try:
                results = predictor.predict_from_file(
                    args.input_file, 
                    args.output_file, 
                    batch_size=args.batch_size
                )
                
                # 打印统计信息
                total = len(results)
                positive = sum(1 for r in results if r['prediction'] == 1)
                negative = total - positive
                avg_confidence = sum(r['confidence'] for r in results) / total
                
                print(f"预测完成:")
                print(f"  总数: {total}")
                print(f"  正例: {positive} ({positive/total*100:.1f}%)")
                print(f"  负例: {negative} ({negative/total*100:.1f}%)")
                print(f"  平均置信度: {avg_confidence:.4f}")
                
            except Exception as e:
                logger.error(f"文件预测失败: {e}")
        
        else:
            print("请指定预测模式：--interactive, --text, 或 --input_file")
            print("使用 --help 查看详细帮助")

if __name__ == "__main__":
    main()
