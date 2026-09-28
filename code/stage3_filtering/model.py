
import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class GlossBERT(nn.Module):
    """
    GlossBERT 模型用于词义消歧任务的二元分类。
    将上下文和词义拼接后输入BERT，使用[CLS]的输出进行分类。
    
    Args:
        model_name: 预训练BERT模型的名称或路径
        dropout_prob: Dropout概率
    """
    def __init__(self, model_name='/mnt/model-gui-agent/bert-base-chinese', dropout_prob=0.1):
        super(GlossBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        # 二元分类器，输出单个logit值
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids，格式为 [CLS] context [SEP] sense [SEP]
            attention_mask: 注意力掩码
            token_type_ids: token类型ids
            labels: 标签（可选），用于计算loss
            
        Returns:
            如果labels存在，返回(loss, logits)元组
            否则返回logits
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS] token的池化输出
        pooled_output = outputs.pooler_output
        
        # Dropout正则化
        pooled_output = self.dropout(pooled_output)
        
        # 分类得到logits
        logits = self.classifier(pooled_output)
        
        # 调整形状为(batch_size,)
        logits = logits.squeeze(-1)
        
        # 计算loss（如果提供了labels）
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        # 返回结果
        if loss is not None:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )
        else:
            return SequenceClassifierOutput(
                logits=logits,
            )
