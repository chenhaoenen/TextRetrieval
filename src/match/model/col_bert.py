# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/30 14:25 
# Description:  
# --------------------------------------------
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel

class ColBert(nn.Module):

    def __init__(self, pre_trained_path):
        super().__init__()
        self.query_bert = BaseModel(pre_trained_path)
        self.passage_bert = BaseModel(pre_trained_path)

    def forward(self, query, passage, labels=None):
        query_embed = self.query_bert(query.input_ids, query.token_type_ids, query.attention_mask)
        passage_embed = self.passage_bert(passage.input_ids, passage.token_type_ids, passage.attention_mask)

        if labels is not None:
            loss = F.cosine_similarity(query_embed, passage_embed, dim=1)
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

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, passage):



