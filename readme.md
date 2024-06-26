# Environment setup

## install cuda 11.3 at https://developer.nvidia.com/cuda-11.3.0-download-archive

## create new conda environment:
conda create -n chute_env python=3.8

## activate conda environment
conda activate chute_env

## install ultralytics
conda install -c conda-forge ultralytics

## install pytorch
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

## install additional packages
conda install configargparse\
pip install gdown\
conda install tensorboard\
pip install easydict\
pip install chardet\
pip install hub_sdk

## install packages for segmentation
pip install opencv-python pycocotools matplotlib onnxruntime onnx

# download classification models
https://drive.google.com/drive/folders/13D_ebQXyF7LqQrSTNoazjofbsmiIECcj?usp=sharing

# download detection model
https://drive.google.com/file/d/17nEVps8HzAJ93AmSNCKyIXgbrzdayPvR/view?usp=sharing

# download classification model 

# download segmentation models
vit_h:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
vit_l:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
vit_b:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
