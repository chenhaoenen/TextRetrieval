# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/27 15:33 
# Description:  
# --------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


class BertCat(BertPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):

        bert_out = self.bert(input_ids, token_type_ids, attention_mask, return_dict=True)

        cls_out = bert_out.last_hidden_state[:,0,:]
        linear_out = self.linear(cls_out).squeeze(1)
        out = torch.sigmoid(linear_out)  #[B]

        if labels is not None:
            loss = F.binary_cross_entropy(out, labels)
            return loss

        return out
