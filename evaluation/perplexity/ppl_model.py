import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from torch.nn import functional as F
import re
from typing import List, Dict, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerplexityCalculator:
    """
    基于ModernBERT-large模型或自回归模型计算词汇在例句中的困惑度
    支持两种模式：
    1. masked: 双向掩码模型（如BERT）
    2. autoregressive: 自回归模型（如GPT、Qwen等）
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        use_data_parallel: bool = False,
        fp16: bool = False,
        model_type: str = "masked",
    ):
        """
        初始化困惑度计算器
        
        Args:
            model_name: 模型名称，默认为ModernBERT-large
            use_data_parallel: 是否启用单机多卡并行（DataParallel）。仅在单机多卡推理时建议打开。
            fp16: 是否以半精度加载/推理（需要GPU支持）。
            model_type: 模型类型，"masked"表示双向掩码模型，"autoregressive"表示自回归模型
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self.model_type = model_type
        logger.info(f"模型类型: {self.model_type}")
        
        # 加载tokenizer和模型
        logger.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 根据模型类型加载不同的模型类
        torch_dtype = torch.float16 if fp16 and torch.cuda.is_available() else None
        
        if model_type == "masked":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        elif model_type == "autoregressive":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            # 对于自回归模型，确保tokenizer有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，请使用'masked'或'autoregressive'")

        # 半精度保护（某些模型需要手动 half）。
        if fp16 and torch_dtype is None and torch.cuda.is_available():
            try:
                self.model.half()
            except Exception:
                logger.warning("模型不支持 half()，将以原精度运行。")

        self.model.to(self.device)

        # 可选：单机多卡并行（DataParallel），仅在有 2 块及以上 GPU 时生效。
        self.use_data_parallel = use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
        if self.use_data_parallel:
            gpu_count = torch.cuda.device_count()
            logger.info(f"启用 DataParallel，GPU 数量: {gpu_count}")
            self.model = torch.nn.DataParallel(self.model)
        else:
            if torch.cuda.is_available():
                logger.info("未启用 DataParallel（use_data_parallel=False 或 仅单卡可用）")

        self.model.eval()
        
        logger.info("模型加载完成")
    
    def extract_target_word(self, head_id: str, tail_id: str) -> str:
        """
        从head_id和tail_id中提取目标词
        
        Args:
            head_id: 头词ID，例如"心02795_1_05_02"
            tail_id: 尾词ID，例如"爱05269_1_05_03"
            
        Returns:
            目标词，例如"心爱"
        """
        # 提取第一个字符
        head_char = head_id[0] if head_id else ""
        tail_char = tail_id[0] if tail_id else ""
        target_word = head_char + tail_char
        
        return target_word
    
    def find_word_positions(self, sentence: str, target_word: str) -> List[Tuple[int, int]]:
        """
        在句子中找到目标词的位置
        
        Args:
            sentence: 输入句子
            target_word: 目标词
            
        Returns:
            目标词在句子中的位置列表 (start, end)
        """
        positions = []
        # 处理波浪号占位符
        sentence_clean = sentence.replace('～', target_word)
        
        # 查找目标词的所有出现位置
        start = 0
        while True:
            pos = sentence_clean.find(target_word, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(target_word)))
            start = pos + 1
            
        return positions
    
    def calculate_word_perplexity(self, sentence: str, target_word: str) -> float:
        """
        计算目标词在给定句子中的困惑度
        
        Args:
            sentence: 输入句子
            target_word: 目标词
            
        Returns:
            困惑度值
        """
        if self.model_type == "masked":
            return self._calculate_word_perplexity_masked(sentence, target_word)
        elif self.model_type == "autoregressive":
            return self._calculate_word_perplexity_autoregressive(sentence, target_word)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _calculate_word_perplexity_masked(self, sentence: str, target_word: str) -> float:
        """
        使用掩码模型计算目标词在给定句子中的困惑度
        
        Args:
            sentence: 输入句子
            target_word: 目标词
            
        Returns:
            困惑度值
        """
        # 处理句子，将波浪号替换为目标词
        sentence_processed = sentence.replace('～', target_word)
        
        # 找到目标词在句子中的位置
        word_positions = self.find_word_positions(sentence_processed, target_word)
        
        if not word_positions:
            logger.warning(f"在句子中未找到目标词 '{target_word}': {sentence_processed}")
            return float('inf')
        
        perplexities = []
        
        for start_pos, end_pos in word_positions:
            # 分别计算目标词的每个字符的困惑度
            word_perplexity = self._calculate_token_perplexity_masked(
                sentence_processed, start_pos, end_pos
            )
            perplexities.append(word_perplexity)
        
        # 返回几何平均困惑度
        if perplexities:
            # 几何平均 = exp(mean(log(x)))
            log_perplexities = [np.log(p) for p in perplexities if p > 0 and not np.isinf(p)]
            return float(np.exp(np.mean(log_perplexities))) if log_perplexities else float('inf')
        return float('inf')
    
    def _calculate_word_perplexity_autoregressive(self, sentence: str, target_word: str) -> float:
        """
        使用自回归模型计算目标词在给定句子中的困惑度
        使用3个不同的prompt模板，结果进行几何平均以增强鲁棒性
        
        Args:
            sentence: 输入句子（包含～占位符）
            target_word: 目标词
            
        Returns:
            困惑度值
        """
        # 构建3个不同的prompt模板
        sentence_with_mask = sentence  # 保留～
        prompt_templates = [
            f"{sentence_with_mask}～是：",
            f"{sentence_with_mask}这里的～指的是：",
            f"{sentence_with_mask}句中的～表示："
        ]
        
        # 预先tokenize目标词的所有字符
        target_tokens = []
        for char in target_word:
            char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
            if char_tokens:
                target_tokens.append(char_tokens[0])
            else:
                logger.warning(f"无法tokenize字符: {char}")
                return float('inf')
        
        # 对每个prompt分别计算困惑度
        perplexities = []
        
        for prompt_idx, prompt in enumerate(prompt_templates):
            # 使用prefill一次性计算所有位置
            # 构建完整输入：prompt + 完整目标词
            full_input = prompt + target_word
            inputs = self.tokenizer(full_input, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # 一次前向传播获取所有位置的logits
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [1, L, V]
            
            # 找到prompt结束的位置
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            prompt_length = prompt_inputs["input_ids"].size(1)
            
            # 计算每个目标token的负对数概率
            neg_log_probs = []
            for i, target_token_id in enumerate(target_tokens):
                # 位置：prompt_length - 1 + i（因为预测的是下一个token）
                pos = prompt_length - 1 + i
                
                if pos >= logits.size(1):
                    logger.warning(f"Prompt {prompt_idx}: 位置 {pos} 超出logits范围")
                    continue
                
                # 获取该位置的logits
                position_logits = logits[0, pos, :]  # [V]
                log_probs = F.log_softmax(position_logits, dim=-1)
                
                # 获取目标token的负对数概率
                neg_log_prob = -log_probs[target_token_id].item()
                neg_log_probs.append(neg_log_prob)
                
                logger.debug(f"Prompt {prompt_idx}, 位置 {i}, token_id={target_token_id}, 负对数概率: {neg_log_prob:.4f}")
            
            # 计算该prompt的困惑度
            if neg_log_probs:
                avg_neg_log_prob = np.mean(neg_log_probs)
                ppl = float(np.exp(avg_neg_log_prob))
                perplexities.append(ppl)
                logger.debug(f"Prompt {prompt_idx} 困惑度: {ppl:.4f}")
            else:
                logger.warning(f"Prompt {prompt_idx} 无有效token")
        
        # 对3个prompt的困惑度进行几何平均
        if perplexities:
            log_perplexities = [np.log(p) for p in perplexities if p > 0 and not np.isinf(p)]
            if log_perplexities:
                geometric_mean = float(np.exp(np.mean(log_perplexities)))
                logger.debug(f"最终几何平均困惑度: {geometric_mean:.4f}")
                return geometric_mean
        
        return float('inf')
    
    def _calculate_token_perplexity_masked(self, sentence: str, start_pos: int, end_pos: int) -> float:
        """
        使用掩码模型计算指定位置token的困惑度
        
        Args:
            sentence: 完整句子
            start_pos: 目标词开始位置
            end_pos: 目标词结束位置
            
        Returns:
            困惑度值
        """
        # 对句子进行tokenization
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 找到目标词对应的token位置
        target_word = sentence[start_pos:end_pos]
        
        # 将目标字符映射到token位置
        token_positions = self._get_token_positions_for_chars(
            sentence, start_pos, end_pos, inputs
        )
        
        if not token_positions:
            return float('inf')
        
        # 构造批量 masked 输入，利用 batch 维度并行（可被 DataParallel 自动分割到多卡）。
        masked_batches = []
        masked_positions = []
        original_tokens = []

        for token_pos in token_positions:
            if token_pos >= input_ids.size(1):
                continue
            mi = input_ids.clone()
            original_tokens.append(mi[0, token_pos].item())
            mi[0, token_pos] = self.tokenizer.mask_token_id
            masked_batches.append(mi)
            masked_positions.append(token_pos)

        if not masked_batches:
            return float('inf')

        masked_input_ids = torch.cat(masked_batches, dim=0).to(self.device)  # [N, L]
        batch_size = masked_input_ids.size(0)
        # 复制 attention mask
        attn = attention_mask.expand(batch_size, -1).contiguous()

        with torch.no_grad():
            outputs = self.model(input_ids=masked_input_ids, attention_mask=attn)
            logits = outputs.logits  # [N, L, V]
            # 取出各样本其对应被 mask 的位置的 logits
            idx = torch.arange(batch_size, device=logits.device)
            pos_tensor = torch.tensor(masked_positions, device=logits.device, dtype=torch.long)
            logits_at_pos = logits[idx, pos_tensor, :]  # [N, V]
            log_probs = F.log_softmax(logits_at_pos, dim=-1)  # [N, V]
            orig_tokens_tensor = torch.tensor(original_tokens, device=log_probs.device, dtype=torch.long)
            selected_log_probs = log_probs[idx, orig_tokens_tensor]  # [N]

        # 负对数概率的平均 -> 困惑度
        # 确保先转为numpy再取负号
        selected_log_probs_np = selected_log_probs.detach().cpu().numpy()
        neg_log_probs = (-selected_log_probs_np).tolist()
        if len(neg_log_probs) == 0:
            return float('inf')
        avg_neg_log_prob = float(np.mean(neg_log_probs))
        return float(np.exp(avg_neg_log_prob))
    
    def _get_token_positions_for_chars(self, sentence: str, start_pos: int, 
                                     end_pos: int, inputs) -> List[int]:
        """
        获取字符位置对应的token位置
        
        Args:
            sentence: 完整句子
            start_pos: 字符开始位置
            end_pos: 字符结束位置
            inputs: tokenizer输出
            
        Returns:
            token位置列表
        """
        # 使用tokenizer的offset_mapping来映射字符位置到token位置
        tokens_with_offsets = self.tokenizer(
            sentence, 
            return_offsets_mapping=True, 
            add_special_tokens=True
        )
        
        offset_mapping = tokens_with_offsets["offset_mapping"]
        token_positions = []
        
        for i, (token_start, token_end) in enumerate(offset_mapping):
            # 检查token是否与目标字符区域重叠
            if token_start < end_pos and token_end > start_pos:
                token_positions.append(i)
        
        return token_positions
    
    def calculate_sentence_perplexities(self, data_item: Dict) -> Dict:
        """
        计算单个数据项中所有句子的困惑度
        
        Args:
            data_item: 包含head_id, tail_id, sentences等字段的字典
            
        Returns:
            包含困惑度结果的字典，保持原有数据结构并添加困惑度字段
        """
        target_word = self.extract_target_word(data_item["head_id"], data_item["tail_id"])
        sentences = data_item["sentences"]
        
        logger.info(f"计算目标词 '{target_word}' 的困惑度")
        
        perplexities = []
        sentence_perplexities = []
        
        for i, sentence in enumerate(sentences):
            try:
                ppl = self.calculate_word_perplexity(sentence, target_word)
                perplexities.append(ppl)
                sentence_perplexities.append(ppl)
                
                logger.info(f"句子 {i+1}: 困惑度 = {ppl:.4f}")
                
            except Exception as e:
                logger.error(f"计算句子 {i+1} 的困惑度时出错: {e}")
                perplexities.append(float('inf'))
                sentence_perplexities.append(float('inf'))
        
        # 计算几何平均困惑度（排除无穷大值）
        valid_perplexities = [p for p in perplexities if p > 0 and not np.isinf(p)]
        if valid_perplexities:
            log_perplexities = [np.log(p) for p in valid_perplexities]
            mean_perplexity = float(np.exp(np.mean(log_perplexities)))
        else:
            mean_perplexity = float('inf')
        
        # 保持原有数据结构，添加困惑度相关字段
        result = data_item.copy()  # 复制原有数据
        result["target_word"] = target_word
        result["mean_perplexity"] = mean_perplexity
        result["sentence_perplexities"] = sentence_perplexities
        
        logger.info(f"目标词 '{target_word}' 的几何平均困惑度: {mean_perplexity:.4f}")
        
        return result
    
    def calculate_sentence_perplexities_batch(self, data_item: Dict, sentence_batch_size: int = 8) -> Dict:
        """
        使用批量处理计算单个数据项中所有句子的困惑度（优化版本，提高GPU利用率）
        
        Args:
            data_item: 包含head_id, tail_id, sentences等字段的字典
            sentence_batch_size: 句子批处理大小
            
        Returns:
            包含困惑度结果的字典，保持原有数据结构并添加困惑度字段
        """
        target_word = self.extract_target_word(data_item["head_id"], data_item["tail_id"])
        sentences = data_item["sentences"]
        
        logger.info(f"计算目标词 '{target_word}' 的困惑度（批量处理模式，batch_size={sentence_batch_size}）")
        
        sentence_perplexities = []
        
        # 批量处理句子
        for i in range(0, len(sentences), sentence_batch_size):
            batch_sentences = sentences[i:i+sentence_batch_size]
            
            try:
                batch_ppls = self._calculate_batch_perplexities(batch_sentences, target_word)
                sentence_perplexities.extend(batch_ppls)
                
                for j, ppl in enumerate(batch_ppls):
                    logger.info(f"句子 {i+j+1}: 困惑度 = {ppl:.4f}")
                    
            except Exception as e:
                logger.error(f"计算句子批次 {i+1}-{i+len(batch_sentences)} 时出错: {e}")
                # 出错时添加无穷大值
                sentence_perplexities.extend([float('inf')] * len(batch_sentences))
        
        # 计算几何平均困惑度（排除无穷大值）
        valid_perplexities = [p for p in sentence_perplexities if p > 0 and not np.isinf(p)]
        if valid_perplexities:
            log_perplexities = [np.log(p) for p in valid_perplexities]
            mean_perplexity = float(np.exp(np.mean(log_perplexities)))
        else:
            mean_perplexity = float('inf')
        
        # 保持原有数据结构，添加困惑度相关字段
        result = data_item.copy()
        result["target_word"] = target_word
        result["mean_perplexity"] = mean_perplexity
        result["sentence_perplexities"] = sentence_perplexities
        
        logger.info(f"目标词 '{target_word}' 的几何平均困惑度: {mean_perplexity:.4f}")
        
        return result
    
    def _calculate_batch_perplexities(self, sentences: List[str], target_word: str) -> List[float]:
        """
        批量计算多个句子的困惑度（真正的批处理，提高GPU利用率）
        
        Args:
            sentences: 句子列表
            target_word: 目标词
            
        Returns:
            困惑度列表
        """
        if self.model_type == "masked":
            return self._calculate_batch_perplexities_masked(sentences, target_word)
        elif self.model_type == "autoregressive":
            return self._calculate_batch_perplexities_autoregressive(sentences, target_word)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _calculate_batch_perplexities_masked(self, sentences: List[str], target_word: str) -> List[float]:
        """
        使用掩码模型批量计算多个句子的困惑度（真正的批处理，提高GPU利用率）
        
        Args:
            sentences: 句子列表
            target_word: 目标词
            
        Returns:
            困惑度列表
        """
        # 预处理所有句子
        processed_sentences = [s.replace('～', target_word) for s in sentences]
        
        # 收集所有需要计算的masked样本
        all_masked_inputs = []
        all_attention_masks = []
        all_original_tokens = []
        sentence_indices = []  # 记录每个masked input对应的句子索引
        
        for sent_idx, sentence in enumerate(processed_sentences):
            # 找到目标词位置
            word_positions = self.find_word_positions(sentence, target_word)
            
            if not word_positions:
                logger.warning(f"在句子中未找到目标词 '{target_word}': {sentence}")
                continue
            
            # 对每个目标词位置
            for start_pos, end_pos in word_positions:
                # tokenization
                inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # 获取token位置
                token_positions = self._get_token_positions_for_chars(
                    sentence, start_pos, end_pos, inputs
                )
                
                if not token_positions:
                    continue
                
                # 为每个token位置创建masked input
                for token_pos in token_positions:
                    if token_pos >= input_ids.size(1):
                        continue
                    
                    mi = input_ids.clone()
                    original_token = mi[0, token_pos].item()
                    mi[0, token_pos] = self.tokenizer.mask_token_id
                    
                    all_masked_inputs.append(mi)
                    all_attention_masks.append(attention_mask)
                    all_original_tokens.append((original_token, token_pos))
                    sentence_indices.append(sent_idx)
        
        if not all_masked_inputs:
            return [float('inf')] * len(sentences)
        
        # 找到最大长度并进行padding
        max_length = max(mi.size(1) for mi in all_masked_inputs)
        
        # Padding所有tensor到相同长度
        padded_masked_inputs = []
        padded_attention_masks = []
        
        for mi, am in zip(all_masked_inputs, all_attention_masks):
            current_length = mi.size(1)
            if current_length < max_length:
                # 使用pad_token_id进行padding
                pad_length = max_length - current_length
                # padding input_ids
                mi_padded = F.pad(mi, (0, pad_length), value=self.tokenizer.pad_token_id)
                # padding attention_mask (用0表示padding位置)
                am_padded = F.pad(am, (0, pad_length), value=0)
                padded_masked_inputs.append(mi_padded)
                padded_attention_masks.append(am_padded)
            else:
                padded_masked_inputs.append(mi)
                padded_attention_masks.append(am)
        
        # 批量推理
        masked_input_ids = torch.cat(padded_masked_inputs, dim=0).to(self.device)  # [N, L]
        attention_masks = torch.cat(padded_attention_masks, dim=0).to(self.device)  # [N, L]
        
        with torch.no_grad():
            outputs = self.model(input_ids=masked_input_ids, attention_mask=attention_masks)
            logits = outputs.logits  # [N, L, V]
            
            # 计算每个样本的负对数概率
            neg_log_probs = []
            for i in range(len(all_masked_inputs)):
                original_token, token_pos = all_original_tokens[i]
                logits_at_pos = logits[i, token_pos, :]  # [V]
                log_probs = F.log_softmax(logits_at_pos, dim=-1)
                neg_log_prob = -log_probs[original_token].item()
                neg_log_probs.append(neg_log_prob)
        
        # 按句子聚合结果
        sentence_neg_log_probs = [[] for _ in range(len(sentences))]
        for i, sent_idx in enumerate(sentence_indices):
            sentence_neg_log_probs[sent_idx].append(neg_log_probs[i])
        
        # 计算每个句子的困惑度
        perplexities = []
        for sent_idx, neg_log_prob_list in enumerate(sentence_neg_log_probs):
            if neg_log_prob_list:
                avg_neg_log_prob = np.mean(neg_log_prob_list)
                ppl = float(np.exp(avg_neg_log_prob))
            else:
                ppl = float('inf')
            perplexities.append(ppl)
        
        return perplexities
    
    def _calculate_batch_perplexities_autoregressive(self, sentences: List[str], target_word: str) -> List[float]:
        """
        使用自回归模型批量计算多个句子的困惑度
        
        Args:
            sentences: 句子列表
            target_word: 目标词
            
        Returns:
            困惑度列表
        """
        perplexities = []
        
        # 对每个句子单独计算（因为自回归模型需要逐字符生成）
        for sentence in sentences:
            try:
                ppl = self._calculate_word_perplexity_autoregressive(sentence, target_word)
                perplexities.append(ppl)
            except Exception as e:
                logger.error(f"计算句子困惑度时出错: {e}")
                perplexities.append(float('inf'))
        
        return perplexities
    
    def process_batch(self, data_items: List[Dict]) -> List[Dict]:
        """
        批量处理多个数据项（旧版本，逐个处理）
        
        Args:
            data_items: 数据项列表
            
        Returns:
            困惑度结果列表
        """
        results = []
        
        for i, item in enumerate(data_items):
            logger.info(f"处理第 {i+1}/{len(data_items)} 个数据项")
            result = self.calculate_sentence_perplexities(item)
            results.append(result)
        
        return results
    
    def process_batch_optimized(self, data_items: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        优化的批量处理：跨词批处理，将不同词的句子放在同一个batch中并行处理
        
        这个方法比 process_batch 更高效，因为它：
        1. 将所有词的所有句子收集到一起
        2. 按照batch_size进行真正的批处理
        3. 将结果正确地分配回每个词
        
        Args:
            data_items: 数据项列表
            batch_size: 批处理大小，推荐16-64（根据GPU显存调整）
            
        Returns:
            困惑度结果列表
        """
        if not data_items:
            return []
        
        logger.info(f"开始优化批处理，共 {len(data_items)} 个数据项，批大小: {batch_size}")
        
        # 步骤1: 收集所有句子和对应的元数据
        all_sentences = []
        all_target_words = []
        sentence_to_item_mapping = []  # 记录每个句子属于哪个数据项的第几个句子
        
        for item_idx, item in enumerate(data_items):
            target_word = self.extract_target_word(item["head_id"], item["tail_id"])
            sentences = item["sentences"]
            
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                all_target_words.append(target_word)
                sentence_to_item_mapping.append((item_idx, sent_idx))
        
        total_sentences = len(all_sentences)
        logger.info(f"总共收集了 {total_sentences} 个句子")
        
        # 步骤2: 批量计算所有句子的困惑度
        all_perplexities = []
        
        if self.model_type == "masked":
            # 掩码模型：可以真正并行处理不同词的句子
            all_perplexities = self._calculate_cross_word_batch_masked(
                all_sentences, all_target_words, batch_size
            )
        elif self.model_type == "autoregressive":
            # 自回归模型：使用prefill批处理
            logger.info("自回归模型模式：使用prefill批处理")
            all_perplexities = self._calculate_cross_word_batch_autoregressive(
                all_sentences, all_target_words, batch_size
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 步骤3: 将困惑度结果分配回各个数据项
        # 优化：预先按item_idx分组困惑度，避免嵌套循环
        item_perplexity_map = {}
        for ppl, (mapped_item_idx, sent_idx) in zip(all_perplexities, sentence_to_item_mapping):
            if mapped_item_idx not in item_perplexity_map:
                item_perplexity_map[mapped_item_idx] = []
            item_perplexity_map[mapped_item_idx].append(ppl)

        results = []
        for item_idx, item in enumerate(data_items):
            target_word = self.extract_target_word(item["head_id"], item["tail_id"])
            
            # 直接获取该数据项的所有困惑度
            item_perplexities = item_perplexity_map.get(item_idx, [])
            
            # 计算几何平均困惑度
            valid_perplexities = [p for p in item_perplexities if p > 0 and not np.isinf(p)]
            if valid_perplexities:
                log_perplexities = [np.log(p) for p in valid_perplexities]
                mean_perplexity = float(np.exp(np.mean(log_perplexities)))
            else:
                mean_perplexity = float('inf')
            
            # 构建结果
            result = item.copy()
            result["target_word"] = target_word
            result["mean_perplexity"] = mean_perplexity
            result["sentence_perplexities"] = item_perplexities
            
            results.append(result)
            
            if (item_idx + 1) % 100 == 0 or (item_idx + 1) == len(data_items):
                logger.info(f"数据项 {item_idx+1}/{len(data_items)}: '{target_word}' 几何平均困惑度 = {mean_perplexity:.4f}")
        
        logger.info("优化批处理完成")
        return results
    
    def _calculate_cross_word_batch_masked(self, sentences: List[str], 
                                          target_words: List[str], 
                                          batch_size: int) -> List[float]:
        """
        跨词批量计算困惑度（仅适用于掩码模型）
        
        这个方法能真正并行处理不同词的句子，大幅提升效率
        
        Args:
            sentences: 句子列表
            target_words: 对应的目标词列表
            batch_size: 批处理大小
            
        Returns:
            困惑度列表
        """
        total = len(sentences)
        all_perplexities = [None] * total  # 预分配结果列表
        
        logger.info(f"开始跨词批处理计算，总句子数: {total}, 批大小: {batch_size}")
        
        # 收集所有需要处理的masked样本
        all_samples = []  # 存储所有待处理的样本
        sample_to_sentence_mapping = []  # 记录每个样本对应的句子索引
        
        for sent_idx, (sentence, target_word) in enumerate(zip(sentences, target_words)):
            # 处理句子，替换波浪号
            sentence_processed = sentence.replace('～', target_word)
            
            # 找到目标词位置
            word_positions = self.find_word_positions(sentence_processed, target_word)
            
            if not word_positions:
                logger.warning(f"句子 {sent_idx}: 未找到目标词 '{target_word}' in '{sentence_processed}'")
                all_perplexities[sent_idx] = float('inf')
                continue
            
            # 为每个目标词位置创建样本
            samples_for_this_sentence = []
            
            for start_pos, end_pos in word_positions:
                # tokenization
                inputs = self.tokenizer(sentence_processed, return_tensors="pt", add_special_tokens=True)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # 获取token位置
                token_positions = self._get_token_positions_for_chars(
                    sentence_processed, start_pos, end_pos, inputs
                )
                
                if not token_positions:
                    continue
                
                # 为每个token位置创建masked input
                for token_pos in token_positions:
                    if token_pos >= input_ids.size(1):
                        continue
                    
                    mi = input_ids.clone()
                    original_token = mi[0, token_pos].item()
                    mi[0, token_pos] = self.tokenizer.mask_token_id
                    
                    all_samples.append({
                        'input_ids': mi,
                        'attention_mask': attention_mask,
                        'original_token': original_token,
                        'token_pos': token_pos
                    })
                    sample_to_sentence_mapping.append(sent_idx)
                    samples_for_this_sentence.append(len(all_samples) - 1)
        
        if not all_samples:
            logger.warning("没有有效的样本需要处理")
            return [float('inf')] * total
        
        logger.info(f"共生成 {len(all_samples)} 个masked样本")
        
        # 批量处理所有样本
        all_neg_log_probs = []
        
        for batch_start in range(0, len(all_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(all_samples))
            batch_samples = all_samples[batch_start:batch_end]
            
            if batch_start % (batch_size * 10) == 0:
                logger.info(f"批处理进度: {batch_start}/{len(all_samples)}")
            
            # 找到这个batch中的最大长度
            max_length = max(sample['input_ids'].size(1) for sample in batch_samples)
            
            # Padding
            padded_input_ids = []
            padded_attention_masks = []
            original_tokens = []
            token_positions = []
            
            for sample in batch_samples:
                mi = sample['input_ids']
                am = sample['attention_mask']
                current_length = mi.size(1)
                
                if current_length < max_length:
                    pad_length = max_length - current_length
                    mi = F.pad(mi, (0, pad_length), value=self.tokenizer.pad_token_id)
                    am = F.pad(am, (0, pad_length), value=0)
                
                padded_input_ids.append(mi)
                padded_attention_masks.append(am)
                original_tokens.append(sample['original_token'])
                token_positions.append(sample['token_pos'])
            
            # 合并为batch tensor
            batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)
            batch_attention_masks = torch.cat(padded_attention_masks, dim=0).to(self.device)
            
            # 批量推理
            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                logits = outputs.logits  # [batch_size, max_length, vocab_size]
                
                # 计算每个样本的负对数概率
                for i in range(len(batch_samples)):
                    token_pos = token_positions[i]
                    original_token = original_tokens[i]
                    
                    logits_at_pos = logits[i, token_pos, :]
                    log_probs = F.log_softmax(logits_at_pos, dim=-1)
                    neg_log_prob = -log_probs[original_token].item()
                    
                    all_neg_log_probs.append(neg_log_prob)
        
        # 按句子聚合结果
        sentence_neg_log_probs = [[] for _ in range(total)]
        for neg_log_prob, sent_idx in zip(all_neg_log_probs, sample_to_sentence_mapping):
            sentence_neg_log_probs[sent_idx].append(neg_log_prob)
        
        # 计算每个句子的困惑度
        for sent_idx in range(total):
            if all_perplexities[sent_idx] is not None:
                # 已经处理过（比如没找到目标词）
                continue
            
            neg_log_prob_list = sentence_neg_log_probs[sent_idx]
            if neg_log_prob_list:
                avg_neg_log_prob = np.mean(neg_log_prob_list)
                ppl = float(np.exp(avg_neg_log_prob))
                all_perplexities[sent_idx] = ppl
            else:
                all_perplexities[sent_idx] = float('inf')
        
        logger.info("跨词批处理计算完成")
        return all_perplexities
    
    def _calculate_cross_word_batch_autoregressive(self, sentences: List[str], 
                                                   target_words: List[str], 
                                                   batch_size: int) -> List[float]:
        """
        自回归模型的跨词批量计算（使用prefill优化）
        
        核心思想：
        1. 将所有句子+目标词构建为完整输入
        2. 批量进行prefill（一次前向传播处理多个句子）
        3. 从输出中提取每个句子的目标词概率
        
        Args:
            sentences: 句子列表
            target_words: 对应的目标词列表
            batch_size: 批处理大小
            
        Returns:
            困惑度列表
        """
        total = len(sentences)
        all_perplexities = []
        
        logger.info(f"自回归模型跨词批处理，总句子数: {total}, 批大小: {batch_size}")
        
        # 准备所有输入 - 使用3个prompt模板
        prompt_templates = [
            "～是：",
            "这里的～指的是：",
            "句中的～表示："
        ]
        
        all_inputs = []  # [(sentence_idx, prompt_idx, full_input), ...]
        all_target_token_ids = []
        all_target_lengths = []
        all_prompts = []  # 对应的prompt文本
        
        for sentence, target_word in zip(sentences, target_words):
            sentence_with_mask = sentence  # 保留～
            
            # 为每个句子构建3个prompt
            for prompt_idx, prompt_template in enumerate(prompt_templates):
                prompt = f"{sentence_with_mask}{prompt_template}"
                full_input = prompt + target_word
                all_inputs.append(full_input)
                all_prompts.append(prompt)
            
            # 预先tokenize目标词（对3个prompt共用）
            target_tokens = []
            for char in target_word:
                char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
                if char_tokens:
                    target_tokens.append(char_tokens[0])
                else:
                    logger.warning(f"无法tokenize字符: {char}")
                    target_tokens = None
                    break
            
            # 每个句子有3个prompt，都使用相同的target_tokens
            for _ in range(3):
                all_target_token_ids.append(target_tokens)
                all_target_lengths.append(len(target_tokens) if target_tokens else 0)
        
        # 批量处理（现在总数是 total * 3）
        total_inputs = len(all_inputs)
        for batch_start in range(0, total_inputs, batch_size):
            batch_end = min(batch_start + batch_size, total_inputs)
            batch_inputs = all_inputs[batch_start:batch_end]
            batch_target_tokens = all_target_token_ids[batch_start:batch_end]
            batch_target_lengths = all_target_lengths[batch_start:batch_end]
            batch_prompts = all_prompts[batch_start:batch_end]
            
            if batch_start % (batch_size * 5) == 0:
                logger.info(f"批处理进度: {batch_start}/{total_inputs}")
            
            # Tokenize整个batch
            tokenized = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            
            # 批量前向传播（prefill）
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # 为每个样本计算困惑度
            for i, (target_tokens, target_length, prompt) in enumerate(zip(batch_target_tokens, batch_target_lengths, batch_prompts)):
                if target_tokens is None or target_length == 0:
                    all_perplexities.append(float('inf'))
                    continue
                
                # 计算prompt长度（考虑padding）
                prompt_tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True
                )
                prompt_length = prompt_tokens["input_ids"].size(1)
                
                # 提取目标词位置的logits并计算负对数概率
                neg_log_probs = []
                for j, target_token_id in enumerate(target_tokens):
                    # 位置：prompt_length - 1 + j
                    pos = prompt_length - 1 + j
                    
                    # 检查位置是否有效（考虑padding）
                    if pos >= logits.size(1) or attention_mask[i, pos].item() == 0:
                        logger.warning(f"样本 {i}: 位置 {pos} 无效")
                        continue
                    
                    # 获取该位置的logits
                    position_logits = logits[i, pos, :]
                    log_probs = F.log_softmax(position_logits, dim=-1)
                    
                    # 计算负对数概率
                    neg_log_prob = -log_probs[target_token_id].item()
                    neg_log_probs.append(neg_log_prob)
                
                # 计算该prompt的困惑度
                if neg_log_probs:
                    avg_neg_log_prob = np.mean(neg_log_probs)
                    ppl = float(np.exp(avg_neg_log_prob))
                else:
                    ppl = float('inf')
                
                all_perplexities.append(ppl)
        
        # 对每个句子的3个prompt进行几何平均
        final_perplexities = []
        for sent_idx in range(total):
            # 获取该句子的3个prompt的困惑度
            ppls = all_perplexities[sent_idx*3:(sent_idx+1)*3]
            
            # 几何平均
            valid_ppls = [p for p in ppls if p > 0 and not np.isinf(p)]
            if valid_ppls:
                log_ppls = [np.log(p) for p in valid_ppls]
                geometric_mean = float(np.exp(np.mean(log_ppls)))
                final_perplexities.append(geometric_mean)
            else:
                final_perplexities.append(float('inf'))
        
        logger.info("自回归模型跨词批处理完成")
        return final_perplexities

def main():
    """
    主函数示例
    """
    # 示例数据
    sample_data = {
        "head_id": "地02795_1_05_02",
        "head_pos": "名词",
        "head_sense": "指地面、土地。",
        "tail_id": "雷05269_1_05_03",
        "tail_pos": "名词",
        "tail_sense": "一种爆炸性武器。",
        "relation": "偏正",
        "confidence": 0.986328125,
        "definition": "埋在地下或布设在地面的爆炸性武器。",
        "sentences": [
            "民众应远离遗弃的～，以免发生爆炸。",
            "工兵正在小心翼翼地排除～。",
            "战争遗留的～给当地居民带来了巨大威胁。"
        ]
    }
    
    import sys
    
    # 根据命令行参数选择模型类型
    if len(sys.argv) > 1 and sys.argv[1] == "autoregressive":
        print("=== 使用自回归模型模式 ===")
        # 使用自回归模型（例如Qwen）
        # 注意：需要根据实际情况修改模型名称
        calculator = PerplexityCalculator(
            model_name="Qwen/Qwen2.5-7B",  # 示例模型，请根据实际情况修改
            model_type="autoregressive",
            fp16=True
        )
        print("\n自回归模型的计算方式：")
        print("1. 输入: 民众应远离遗弃的～，以免发生爆炸。～是:")
        print("   输出: 地")
        print("2. 输入: 民众应远离遗弃的～，以免发生爆炸。～是:地")
        print("   输出: 雷")
        print()
    else:
        print("=== 使用掩码模型模式（默认）===")
        # 使用默认的掩码模型
        calculator = PerplexityCalculator()
        print("\n掩码模型的计算方式：使用MASK替换目标词并计算概率")
        print()
    
    # 计算困惑度
    result = calculator.calculate_sentence_perplexities_batch(sample_data,64)
    
    # 输出结果
    print("=== 计算结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()