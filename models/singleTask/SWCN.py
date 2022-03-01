# -*- coding: utf-8 -*-
# @Author  : Yang Wu (HIT)
# @Email   : ywu@ir.hit.edu.cn
# based on https://github.com/thuiar/MMSA

import os
import sys
import collections

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.BertSelfTokenEncoder import BertSelfTokenEncoder


__all__ = ['SWCN']

class SWCN(nn.Module):
    def __init__(self, args):
        super(SWCN, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned

        self.gate_network = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)        
        self.text_model = BertSelfTokenEncoder(language=args.language, use_finetune=args.use_finetune)


        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)


        self.gate_rep_audio = AuViGateNet(audio_in, args.a_lstm_hidden_size, \
                            num_layers=1, dropout=0)
        self.gate_rep_video = AuViGateNet(video_in, args.v_lstm_hidden_size, \
                            num_layers=1, dropout=0)
        self.gate_rep_audio_video = AuViGateNet(audio_in + video_in, args.a_lstm_hidden_size + args.v_lstm_hidden_size, \
                            num_layers=1, dropout=0)

        self.gate_linear = nn.Linear(args.text_out + 2 * args.v_lstm_hidden_size + 2 * args.a_lstm_hidden_size, 1)
        self.sentiment_attn = nn.Linear(args.text_out + 2 * args.v_lstm_hidden_size + 2 * args.a_lstm_hidden_size, 1)
        self.added_gate = nn.Linear(args.text_out * 2, 1)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def forward(self, text, audio, video, max_poss_one, max_poss_one_index, extracted_indexs, max_value_one, extracted_sw_indicators):
        audio, audio_lengths = audio
        video, video_lengths = video

        # gating
        last_seq_hiddens = self.gate_network(text)

        # text shape : bz * 3 * max_lens

        # fake alignment
        avg_audio_seg = audio.size(1) // text.size(2)
        avg_video_seg = video.size(1) // text.size(2)

        audio_aligned = [torch.zeros(audio.size(0), 1, audio.size(2)).cuda()]
        video_aligned = [torch.zeros(video.size(0), 1, video.size(2)).cuda()]

        for segment in torch.split(audio, avg_audio_seg, 1)[:text.size(2)-1]:
            audio_aligned.append(torch.mean(segment, 1, keepdims=True))
        
        for segment in torch.split(video, avg_video_seg, 1)[:text.size(2)-1]:
            video_aligned.append(torch.mean(segment, 1, keepdims=True))
        
        audio_aligned = torch.cat(audio_aligned, 1)
        video_aligned = torch.cat(video_aligned, 1)

        # # represent audio/video using lstm
        _lengths = [text.size(2) for i in range(text.size(0))]
        audio_rep = self.gate_rep_audio(audio_aligned, _lengths)
        video_rep = self.gate_rep_video(video_aligned, _lengths)

        _lengths = [text.size(2) for i in range(text.size(0))]
        audio_video_rep = self.gate_rep_audio_video(torch.cat([audio_aligned, video_aligned], -1), _lengths)


        gate_input = torch.cat([audio_rep, video_rep, last_seq_hiddens, audio_video_rep], -1)

        gate_value = F.sigmoid(self.gate_linear(gate_input)) 

        embedding_map = self.text_model.get_input_embeddings()
        mask_id = torch.LongTensor([103]).cuda()
        mask_embedding = embedding_map(mask_id)



        # swcn add sentiment

        candidate_embeddings = embedding_map(extracted_indexs)

        top_n_num = extracted_indexs.size(1)

        # same length
        # audio/video
        audio_shape = audio_rep.size()
        _max_poss_one_index = max_poss_one_index + torch.arange(0, audio_shape[0]).cuda() * audio_shape[1]
        audio_rep = audio_rep.contiguous().view(audio_shape[0]*audio_shape[1], -1)
        # bz * hidden_size
        target_audio_embedding = torch.index_select(audio_rep, 0, _max_poss_one_index)

        video_shape = video_rep.size()
        _max_poss_one_index = max_poss_one_index + torch.arange(0, video_shape[0]).cuda() * video_shape[1]   
        video_rep = video_rep.contiguous().view(video_shape[0]*video_shape[1], -1)
        # bz * hidden_size
        target_video_embedding = torch.index_select(video_rep, 0, _max_poss_one_index) 

        target_audio_embedding = target_audio_embedding.unsqueeze(1).expand(audio_shape[0], top_n_num, audio_shape[2])
        target_video_embedding = target_video_embedding.unsqueeze(1).expand(video_shape[0], top_n_num, video_shape[2])

        # attention_input = torch.cat([target_audio_embedding, target_video_embedding, candidate_embeddings], -1)

        audio_video_shape = audio_video_rep.size()
        _max_poss_one_index = max_poss_one_index + torch.arange(0, audio_video_shape[0]).cuda() * audio_video_shape[1]
        audio_video_rep = audio_video_rep.contiguous().view(audio_video_shape[0]*audio_video_shape[1], -1)
        # bz * hidden_size
        target_audio_video_embedding = torch.index_select(audio_video_rep, 0, _max_poss_one_index)
        target_audio_video_embedding = target_audio_video_embedding.unsqueeze(1).expand(audio_video_shape[0], top_n_num, audio_video_shape[2])

        attention_input = torch.cat([target_audio_video_embedding, candidate_embeddings, target_audio_embedding, target_video_embedding,], -1)   
        score = self.sentiment_attn(attention_input).squeeze()

        # bz * topn 
        # only calculate on the candidate sentiment words
        # score = extracted_sw_indicators.float() * score
        attention_weights = F.softmax(score, 1)

        attention_weights = attention_weights.unsqueeze(2).expand_as(candidate_embeddings)
        final_candidate_embedding = torch.sum(attention_weights * candidate_embeddings, 1)
        input_embeddings = embedding_map(text[:,0,:].long())
        final_candidate_embedding = final_candidate_embedding.unsqueeze(1).expand_as(input_embeddings)

        sentiment_masking = []
        for mpo, mvo in zip(max_poss_one, max_value_one):
            s = [0 for i in range(input_embeddings.size(1))]
            if mvo > 0.5:
                s[mpo] = 1
            sentiment_masking.append(s)
        
        sentiment_masking = torch.Tensor(sentiment_masking).unsqueeze(2).cuda()

        # refining token embeddings
        gate_value = sentiment_masking * gate_value
        mask_embedding = mask_embedding.unsqueeze(0).expand_as(input_embeddings)
        added_gate_input = torch.cat([mask_embedding, final_candidate_embedding], -1)
        added_gate_value = F.sigmoid(self.added_gate(added_gate_input))

        added_embeddings = added_gate_value * final_candidate_embedding + (1 - added_gate_value) * mask_embedding
        
        gate_value = gate_value.expand_as(input_embeddings)
        refined_input_embeddings = (1 - gate_value) * input_embeddings + gate_value * added_embeddings

        text = self.text_model(text, refined_input_embeddings)[:,0,:]
        audio = self.audio_model(audio, audio_lengths)
        video = self.video_model(video, video_lengths)


        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'important_value': 1 - torch.max(gate_value, 1).values[:, 0]
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

        
class AuViGateNet(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViGateNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        final_states, _ = self.rnn(packed_sequence)
        final_states, _ = pad_packed_sequence(final_states, batch_first=True)
        return final_states
