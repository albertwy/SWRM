# SWRM
Code for Findings of ACL 2022 Paper **"Sentiment Word Aware Multimodal Refinement for Multimodal Sentiment Analysis with ASR Errors"**


## Clone
Clone this repo and install requirements. 

    git clone 
    pip install -r requirements.txt

## Dataset
Download dataset files and put them in the *dataset* directory
- [Cow Transfer](https://cowtransfer.com/s/15d1bc5193a445) with code: r9etft

Download pre-trained BERT models（provided by [MMSA Project](https://github.com/thuiar/MMSA)）and specify the dir name in the Code(models/subNets/Bert*.py) 

- [BaiDu Cloud Drive](https://pan.baidu.com/s/1oksuDEkkd3vGg2oBMBxiVw) with code: ctgs
- [Google Cloud Drive](https://drive.google.com/drive/folders/1-LT7HtECyKAVrMcYI-OjMr4g3ISfTRzS)


## Run

    python run.py --modelName *** --expSetting ***

## Paper


## Acknowledgments
- [https://github.com/thuiar/MMSA](https://github.com/thuiar/MMSA)
    







