"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from models.subNets.AlignNets import AlignSubNet
from models.singleTask import *
from models.multiTask import *

__all__ = ['AMIO']

MODEL_MAP = {
    # single-task
    'tfn': TFN,
    'lmf': LMF,
    'mfn': MFN,
    'ef_lstm': EF_LSTM,
    'lf_dnn': LF_DNN,
    'graph_mfn': Graph_MFN,
    'mult': MULT,
    'misa': MISA,
    'swrm': SWRM,
    'self_mm': SELF_MM
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_model_aligned = args.need_model_aligned
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args.seq_lens = self.alignNet.get_seq_len()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x, *args)
        return self.Model(text_x, audio_x, video_x, *args)