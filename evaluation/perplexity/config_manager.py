"""
配置管理模块：加载和管理困惑度计算的配置参数
"""

import json
import os
from typing import Dict, Any
import logging

class ConfigManager:
    """
    配置管理器：负责加载和管理配置参数
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        default_config = {
            "model_config": {
                "model_name": "answerdotai/ModernBERT-large",
                "device": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True
            },
            "processing_config": {
                "batch_size": 10,
                "max_sequence_length": 512,
                "use_fast_tokenizer": True,
                "padding": True,
                "truncation": True
            },
            "output_config": {
                "save_detailed_results": True,
                "save_summary_report": True,
                "output_format": "json",
                "decimal_places": 4
            },
            "logging_config": {
                "level": "INFO",
                "log_to_file": True,
                "log_file": "perplexity.log",
                "log_format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "checkpoint_config": {
                "enable_checkpoints": True,
                "checkpoint_interval": 10,
                "auto_resume": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置（用户配置覆盖默认配置）
                config = self._merge_configs(default_config, user_config)
                print(f"已加载配置文件: {self.config_file}")
            except Exception as e:
                print(f"加载配置文件失败，使用默认配置: {e}")
                config = default_config
        else:
            print(f"配置文件不存在，使用默认配置: {self.config_file}")
            config = default_config
            # 创建默认配置文件
            self._save_config(config)
        
        return config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """
        合并配置字典
        
        Args:
            default: 默认配置
            user: 用户配置
            
        Returns:
            合并后的配置
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _save_config(self, config: Dict):
        """
        保存配置到文件
        
        Args:
            config: 配置字典
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"默认配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """
        获取配置值
        
        Args:
            section: 配置节名
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """
        设置配置值
        
        Args:
            section: 配置节名
            key: 配置键名
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save(self):
        """
        保存当前配置到文件
        """
        self._save_config(self.config)
    
    def setup_logging(self):
        """
        根据配置设置日志
        """
        log_config = self.get("logging_config")
        
        # 设置日志级别
        level = getattr(logging, log_config.get("level", "INFO").upper())
        
        # 配置处理器
        handlers = [logging.StreamHandler()]
        
        if log_config.get("log_to_file", True):
            log_file = log_config.get("log_file", "perplexity.log")
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        
        # 设置日志格式
        log_format = log_config.get("log_format", "%(asctime)s - %(levelname)s - %(message)s")
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers,
            force=True  # 重新配置已存在的logger
        )
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.get("model_config", default={})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        获取处理配置
        
        Returns:
            处理配置字典
        """
        return self.get("processing_config", default={})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        获取输出配置
        
        Returns:
            输出配置字典
        """
        return self.get("output_config", default={})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """
        获取检查点配置
        
        Returns:
            检查点配置字典
        """
        return self.get("checkpoint_config", default={})

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Returns:
        配置管理器实例
    """
    return config_manager