import os
import argparse
import json
from utils.functions import Storage
class ConfigRegression():
    def __init__(self, args): 
        HYPER_MODEL_MAP = {
            'tfn': self.__TFN,
            'lmf': self.__LMF,
            'mfn': self.__MFN,
            'ef_lstm': self.__EF_LSTM,
            'lf_dnn': self.__LF_DNN,
            'graph_mfn': self.__Graph_MFN,
            'mult': self.__MULT,
            'misa': self.__MISA,
            'swcn': self.__SWCN,
            'self_mm': self.__SELF_MM
        }
        if (args.expSetting).split("_")[0] == "ibm":
            HYPER_MODEL_MAP['swcn'] = self.__TSWCN

        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams(args)

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self, args):
        # dataset dir
        root_dataset_dir = args.datasetPath 

        # data file
        mosi_mapping = {
                        "speechbrain_asr_train": "MOSI/asr_speechbrain_unaligned_50.pkl",
                        "ibm_asr_train": "MOSI/asr_ibm_unaligned_50.pkl",
                        "xf_asr_train": "MOSI/asr_xf_unaligned_50.pkl",
                        "gold_asr_train": "MOSI/asr_gold_unaligned_50.pkl"
                        } 

        tmp = {
            'mosi':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, mosi_mapping[args.expSetting]),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            }   
        }
        return tmp

    def __MULT(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'early_stop': 8,
                'use_bert': False,
                # use finetune for bert
                'use_bert_finetune': False,
                # use attention mask for Transformer
                'attn_mask': True, 
                'update_epochs': 8,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.2,
                    'relu_dropout': 0.2,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (40, 10), 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 4,
                    'learning_rate': 1e-3,
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 4, 
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 1,
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    'attn_dropout': 0.1, # crossmodal attention block dropout
                    'output_dropout': 0.1,
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8, 
                    # when to decay learning rate (default: 20)
                    'patience': 20, 
                    'weight_decay': 0.0
                }
            },
        }
        return tmp
    
   

    def __TSWCN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 0.005,
                    'learning_rate_other': 0.001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                }
            },
        }

        for key in self.paras.keys():
            if ("dropout" in key) or ("weight_decay" in key) or ("learning_rate" in key) or ("H" in key):
                tmp["datasetParas"]["mosi"][key] = float(self.paras[key])
            else:
                tmp["datasetParas"]["mosi"][key] = int(float(self.paras[key]))

        return tmp


    def __TSWCN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.001,
                    'learning_rate_video': 0.0001,
                    'learning_rate_other': 0.0001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.01,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':64,
                    'post_audio_dim': 32,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                }
            },
        }
        return tmp


    def __SWCN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 0.005,
                    'learning_rate_other': 0.001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                }
            },
        }
        return tmp


    def __MISA(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_finetune': True,
                'use_bert': True,
                'early_stop': 8,
                'update_epochs': 2,
                'rnncell': 'lstm',
                'use_cmd_sim': True,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-4,
                    'hidden_size': 128,
                    'dropout': 0.2,
                    'reverse_grad_weight': 0.8,
                    'diff_weight': 0.3,
                    'sim_weight': 0.8,
                    'sp_weight': 0.0,
                    'recon_weight': 0.8,
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 1.0,
                    'weight_decay': 0.002,
                }
            },
        }
        return tmp
    
    def __TFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 16, 128),
                    'text_out': 128,
                    'post_fusion_dim': 32,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'batch_size': 32,
                    'learning_rate': 5e-4,
                }
            },
        }
        return tmp
    
    def __LMF(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (256, 32, 256),
                    'dropouts': (0.3, 0.3, 0.3, 0.3),
                    'rank': 4,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'factor_lr': 1e-3, # factor_learning_rate
                    'weight_decay': 1e-4,
                }
            },
        }
        return tmp
    
    def __MFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': True,
                'early_stop': 8,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (128, 32, 128),
                    'memsize': 400,
                    'windowsize': 2,
                    'NN1Config': {"drop": 0.2, "shapes": 128},
                    'NN2Config': {"drop": 0.7, "shapes": 64},
                    'gamma1Config': {"drop": 0.0, "shapes": 64},
                    'gamma2Config': {"drop": 0.0, "shapes": 256},
                    'outConfig': {"drop": 0.2, "shapes": 256},
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                }
            },
        }
        return tmp
    
    def __EF_LSTM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': False,
                'early_stop': 8,
            },
            'datasetParas':{
                'mosi':{
                    'hidden_dims': 256,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4,
                }
            },
        }
        return tmp
    
    def __LF_DNN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'early_stop': 8,
            },
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 16, 64),
                    'text_out': 256,
                    'post_fusion_dim': 16,
                    'dropouts': (0.4, 0.4, 0.4, 0.4),
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'weight_decay': 0.01,
                }
            },
        }
        return tmp
   
    def __Graph_MFN(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': False,
                'early_stop': 8,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    'hidden_dims': (64, 32, 64),
                    'memsize': 128,
                    'inner_node_dim': 32,
                    'NNConfig': {"drop": 0.0, "shapes": 128},
                    'gamma1Config': {"drop": 0.7, "shapes": 32},
                    'gamma2Config': {"drop": 0.7, "shapes": 64},
                    'outConfig': {"drop": 0.7, "shapes": 32},
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0,
                }
            },
        }
        return tmp

    
    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 0.005,
                    'learning_rate_other': 0.001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 3.0
                }
            },
        }
        return tmp

    def get_config(self):
        return self.args
