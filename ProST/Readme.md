# Progressive Spatio-Temporal Prototype Matching for Text-Video Retrieval

Our paper is accepted by ICCV2023.

We released a simple version of [Prost](https://www.modelscope.cn/models/damo/multi_modal_clip_vtretrieval_prost/summary) in ModelScope, which is convenient for people to use.
ModelScope is built on the concept of "Model as a Service" (MaaS). It aims to bring together state-of-the-art machine learning models from the AI community and simplify the process of utilizing AI models in practical applications.


## Requirements
```sh
# From TS2-Net
conda create -n prost python=3.8.8
. activate
conda activate prost
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12 transformers==4.15.0 fairscale==0.4.4 pycocoevalcap decord
conda install -y ruamel_yaml
pip install numpy opencv-python Pillow pyyaml requests scikit-image scipy tqdm regex easydict scikit-learn
pip install mmcv terminaltables tensorboardX python-magic faiss-gpu imageio-ffmpeg
pip install yacs Cython tensorboard gdown termcolor tabulate xlrd==1.2.0
pip install ffmpeg-python librosa pydub pytorch_lightning torchlibrosa
pip install gpustat einops ftfy boto3 pandas
pip install git+https://github.com/openai/CLIP.git
```


## Data Preparing 

Please refer to [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) to get data annotation.

According to compress_video.py, we convert the original video to mp4 format and set fps to 3.

## Usage
* For didemo, 
```sh
sh scripts/run_didemo.sh
```




## Acknowledge
* Our code is based on [TS2-Net](https://github.com/yuqi657/ts2_net) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
