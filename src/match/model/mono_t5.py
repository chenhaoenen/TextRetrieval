# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/31 17:05 
# Description:  
# --------------------------------------------
import torch
from transformers import T5ForConditionalGeneration

class MonoT5():
    def __init__(self, pre_trained_path):
        super(MonoT5, self).__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(pre_trained_path)

    def inference(self, patterns):
        '''Reference pygaggle: https://github.com/castorini/pygaggle'''
        with torch.no_grad():
            encoder_outputs = self.t5.get_encoder()(patterns.input_ids, attention_mask=patterns.attention_mask)
            decode_ids = torch.full((patterns.input_ids.size(0), 1),
                                    self.t5.config.decoder_start_token_id,
                                    dtype=torch.long).to(patterns.input_ids.device)
            model_inputs = self.t5.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                past=None,
                attention_mask=patterns.attention_mask,
                use_cache=True)

            outputs = self.t5(**model_inputs)

            next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)

            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = next_token_logits[:, [6136, 1176]]
            batch_scores = torch.log_softmax(batch_scores, dim=1)
            batch_log_probs = batch_scores[:, 1].tolist()

            return batch_log_probs

    def train(self, patterns, labels):
        out = self.t5(input_ids=patterns.input_ids, attention_mask=patterns.attention_mask, labels=labels.input_ids, return_dict=True)
        return out.loss
