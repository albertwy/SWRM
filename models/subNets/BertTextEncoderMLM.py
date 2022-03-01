import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertForMaskedLM

__all__ = ['BertTextEncoderMLM']

class BertTextEncoderMLM(nn.Module):
    def __init__(self, language='en'):
        """
        language: en / cn
        """
        super(BertTextEncoderMLM, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertForMaskedLM
        # directory is fine
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('/home/pretrained_model/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('/home/pretrained_model/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('/home/pretrained_model/bert_cn')
            self.model = model_class.from_pretrained('/home/pretrained_model/bert_cn')
            
    def get_tokenizer(self):
        return self.tokenizer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, input_mask, segment_ids, input_ids, topk_num, masked_idx):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """

        input_ids = input_ids.long()
        segment_ids = segment_ids.long()
        input_mask = input_mask.float()
        with torch.no_grad():
            predictions = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
            
            indexs = []
            tokens = []
            for i in range(len(masked_idx)):
                _, predicted_index = torch.topk(predictions[i, masked_idx[i]], topk_num)
                predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_index.cpu().numpy().tolist())
                indexs.append(predicted_index)
                tokens.append(predicted_token)

        return indexs, tokens
    
