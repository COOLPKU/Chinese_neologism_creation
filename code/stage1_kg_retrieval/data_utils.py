"""
数据处理工具模块
"""

import json
import os
import random
import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.data = []
        self.stats = {}
    
    def load_from_txt(self, file_path: str, delimiter: str = '\t') -> List[Dict]:
        """
        从txt文件加载数据
        格式: text\tlabel
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split(delimiter, 1)
                    if len(parts) >= 2:
                        text = parts[0].strip()
                        label = int(parts[1].strip())
                        if label in [0, 1]:
                            data.append({"text": text, "label": label})
                        else:
                            logger.warning(f"行 {line_num}: 标签不是0或1: {label}")
                    else:
                        logger.warning(f"行 {line_num}: 格式不正确: {line}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"行 {line_num}: 解析失败: {e}")
        
        self.data = data
        self._compute_stats()
        logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
        return data
    
    def load_from_json(self, file_path: str) -> List[Dict]:
        """从JSON文件加载数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        valid_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'text' in item and 'label' in item:
                if item['label'] in [0, 1]:
                    valid_data.append(item)
                else:
                    logger.warning(f"项 {i}: 标签不是0或1: {item['label']}")
            else:
                logger.warning(f"项 {i}: 格式不正确: {item}")
        
        self.data = valid_data
        self._compute_stats()
        logger.info(f"从 {file_path} 加载了 {len(valid_data)} 条有效数据")
        return valid_data
    
    def load_from_csv(self, file_path: str, text_column: str = 'text', label_column: str = 'label') -> List[Dict]:
        """从CSV文件加载数据"""
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV文件必须包含 '{text_column}' 和 '{label_column}' 列")
        
        data = []
        for _, row in df.iterrows():
            text = str(row[text_column]).strip()
            try:
                label = int(row[label_column])
                if label in [0, 1] and text:
                    data.append({"text": text, "label": label})
            except (ValueError, TypeError):
                continue
        
        self.data = data
        self._compute_stats()
        logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
        return data
    
    def save_to_json(self, file_path: str, data: List[Dict] = None):
        """保存数据到JSON文件"""
        if data is None:
            data = self.data
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(data)} 条数据到 {file_path}")
    
    def save_to_txt(self, file_path: str, data: List[Dict] = None, delimiter: str = '\t'):
        """保存数据到txt文件"""
        if data is None:
            data = self.data
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"{item['text']}{delimiter}{item['label']}\n")
        
        logger.info(f"已保存 {len(data)} 条数据到 {file_path}")
    
    def _compute_stats(self):
        """计算数据统计信息"""
        if not self.data:
            return
        
        labels = [item['label'] for item in self.data]
        texts = [item['text'] for item in self.data]
        
        self.stats = {
            'total_samples': len(self.data),
            'label_distribution': dict(Counter(labels)),
            'avg_text_length': sum(len(text) for text in texts) / len(texts),
            'min_text_length': min(len(text) for text in texts),
            'max_text_length': max(len(text) for text in texts),
        }
        
        # 计算类别平衡度
        label_counts = list(self.stats['label_distribution'].values())
        if len(label_counts) == 2:
            self.stats['class_balance_ratio'] = min(label_counts) / max(label_counts)
        
    def print_stats(self):
        """打印数据统计信息"""
        if not self.stats:
            logger.warning("没有可用的统计信息")
            return
        
        print("=== 数据统计信息 ===")
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"标签分布: {self.stats['label_distribution']}")
        print(f"平均文本长度: {self.stats['avg_text_length']:.1f} 字符")
        print(f"文本长度范围: {self.stats['min_text_length']} - {self.stats['max_text_length']} 字符")
        
        if 'class_balance_ratio' in self.stats:
            print(f"类别平衡度: {self.stats['class_balance_ratio']:.3f}")
            if self.stats['class_balance_ratio'] < 0.5:
                print("⚠️ 警告: 数据集类别不平衡，建议进行平衡处理")
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分割数据集"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")
        
        if not self.data:
            raise ValueError("没有可分割的数据")
        
        # 提取标签用于分层分割
        texts = [item['text'] for item in self.data]
        labels = [item['label'] for item in self.data]
        
        # 第一次分割: train vs (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )
        
        # 第二次分割: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state,
                stratify=temp_labels
            )
        else:
            if val_ratio > 0:
                val_texts, val_labels = temp_texts, temp_labels
                test_texts, test_labels = [], []
            else:
                test_texts, test_labels = temp_texts, temp_labels
                val_texts, val_labels = [], []
        
        # 构建数据集
        train_data = [{"text": text, "label": label} for text, label in zip(train_texts, train_labels)]
        val_data = [{"text": text, "label": label} for text, label in zip(val_texts, val_labels)]
        test_data = [{"text": text, "label": label} for text, label in zip(test_texts, test_labels)]
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(train_data)} 条")
        logger.info(f"  验证集: {len(val_data)} 条")
        logger.info(f"  测试集: {len(test_data)} 条")
        
        return train_data, val_data, test_data
    
    def balance_data(self, method: str = 'undersample') -> List[Dict]:
        """平衡数据集"""
        if not self.data:
            raise ValueError("没有可平衡的数据")
        
        # 按标签分组
        label_groups = {}
        for item in self.data:
            label = item['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        
        if method == 'undersample':
            # 下采样到最少的类别
            min_count = min(len(group) for group in label_groups.values())
            balanced_data = []
            for label, group in label_groups.items():
                sampled = random.sample(group, min_count)
                balanced_data.extend(sampled)
        
        elif method == 'oversample':
            # 上采样到最多的类别
            max_count = max(len(group) for group in label_groups.values())
            balanced_data = []
            for label, group in label_groups.items():
                if len(group) < max_count:
                    # 重复采样
                    needed = max_count - len(group)
                    additional = random.choices(group, k=needed)
                    balanced_data.extend(group + additional)
                else:
                    balanced_data.extend(group)
        
        else:
            raise ValueError("method 必须是 'undersample' 或 'oversample'")
        
        # 打乱数据
        random.shuffle(balanced_data)
        
        logger.info(f"数据平衡完成 ({method}): {len(self.data)} -> {len(balanced_data)} 条")
        return balanced_data
    
    def filter_by_length(self, min_length: int = 5, max_length: int = 512) -> List[Dict]:
        """根据文本长度过滤数据"""
        if not self.data:
            raise ValueError("没有可过滤的数据")
        
        filtered_data = [
            item for item in self.data 
            if min_length <= len(item['text']) <= max_length
        ]
        
        logger.info(f"长度过滤完成: {len(self.data)} -> {len(filtered_data)} 条")
        return filtered_data
    
    def remove_duplicates(self) -> List[Dict]:
        """去除重复数据"""
        if not self.data:
            raise ValueError("没有可去重的数据")
        
        seen_texts = set()
        unique_data = []
        
        for item in self.data:
            text = item['text']
            if text not in seen_texts:
                seen_texts.add(text)
                unique_data.append(item)
        
        logger.info(f"去重完成: {len(self.data)} -> {len(unique_data)} 条")
        return unique_data

def create_sample_data(output_path: str, num_samples: int = 1000):
    """创建示例数据"""
    # 正样本示例
    positive_templates = [
        "这个产品质量很好，{adjective}。",
        "服务态度{adjective}，值得推荐。",
        "功能{adjective}，使用体验很棒。",
        "{adjective}的产品，性价比很高。",
        "设计{adjective}，做工精细。",
    ]
    
    positive_adjectives = ["优秀", "出色", "完美", "卓越", "精良", "一流", "顶级", "超赞", "满意", "惊艳"]
    
    # 负样本示例
    negative_templates = [
        "这个产品质量{adjective}，不推荐。",
        "服务态度{adjective}，体验不好。", 
        "功能{adjective}，使用不便。",
        "{adjective}的产品，不值得购买。",
        "设计{adjective}，做工粗糙。",
    ]
    
    negative_adjectives = ["很差", "糟糕", "劣质", "失望", "不行", "垃圾", "恶劣", "低劣", "次品", "缺陷"]
    
    data = []
    
    for i in range(num_samples // 2):
        # 正样本
        template = random.choice(positive_templates)
        adjective = random.choice(positive_adjectives)
        text = template.format(adjective=adjective)
        data.append({"text": text, "label": 1})
        
        # 负样本
        template = random.choice(negative_templates)
        adjective = random.choice(negative_adjectives)
        text = template.format(adjective=adjective)
        data.append({"text": text, "label": 0})
    
    # 保存数据
    processor = DataProcessor()
    processor.data = data
    processor._compute_stats()
    processor.save_to_json(output_path)
    processor.print_stats()
    
    return data

if __name__ == "__main__":
    # 示例使用
    processor = DataProcessor()
    
    # 创建示例数据
    sample_data = create_sample_data("sample_data.json", 1000)
    
    # 分割数据
    train_data, val_data, test_data = processor.split_data()
    
    # 保存分割后的数据
    processor.save_to_json("train_data.json", train_data)
    processor.save_to_json("val_data.json", val_data)
    processor.save_to_json("test_data.json", test_data)
