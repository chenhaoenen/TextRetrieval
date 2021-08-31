# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/31 17:05 
# Description:  
# --------------------------------------------
from torch import nn
from transformers import T5ForConditionalGeneration


class MonoT5():
    def __init__(self, pre_trained_path):
        super(MonoT5, self).__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(pre_trained_path)

    def inference(self, pattern, labels):

        t5_out = self.t5.gen(input_ids=pattern.input_ids, attention_mask=pattern.attention_mask, return_dict=True)
        true_flase_label = t5_out.logits[:, [6136, 1176]]

    def train(self, patterns, labels):
        out = self.t5(input_ids=patterns.input_ids, attention_mask=patterns.attention_mask, labels=labels.input_ids, return_dict=True)
        return out.loss
