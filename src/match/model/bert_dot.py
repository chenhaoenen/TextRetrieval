# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/30 14:25 
# Description:  
# --------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel

class BertDot(nn.Module):

    def __init__(self, pre_trained_path):
        super().__init__()
        self.query_bert = BaseModel(pre_trained_path)
        self.passage_bert = BaseModel(pre_trained_path)

    def forward(self, querys, passages, labels=None):
        query_embed = self.query_bert(querys.input_ids, querys.token_type_ids, querys.attention_mask)
        passage_embed = self.passage_bert(passages.input_ids, passages.token_type_ids, passages.attention_mask)

        if labels is not None:
            out = torch.cosine_similarity(query_embed, passage_embed, dim=1)
            loss = F.binary_cross_entropy_with_logits(out, labels)
            return loss

        return query_embed, passage_embed

class BaseModel(nn.Module):
    def __init__(self, pre_trained_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(pre_trained_path)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert(input_ids, token_type_ids, attention_mask, return_dict=True)
        cls_out = bert_out.last_hidden_state[:,0,:]
        linear_out = self.linear(cls_out)
        return linear_out
