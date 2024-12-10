# EmoNeXt: an Adapted ConvNeXt for facial Emotion Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/emonext-an-adapted-convnext-for-facial/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=emonext-an-adapted-convnext-for-facial)

*This repository contains the code referenced in the paper: ["EmoNeXt: an Adapted ConvNeXt for facial Emotion Recognition"](https://ieeexplore.ieee.org/abstract/document/10337732).*

## Abstract
Facial expressions play a crucial role in human communication serving as a powerful and impactful means to express a wide range of emotions. With advancements in artificial intelligence and computer vision, deep neural networks have emerged as effective tools for facial emotion recognition. In this paper, we propose EmoNeXt, a novel deep learning framework for facial expression recognition based on an adapted ConvNeXt architecture network. We integrate a Spatial Transformer Network (STN) to focus on feature-rich regions of the face and Squeeze-and-Excitation blocks to capture channel-wise dependencies. Moreover, we introduce a self-attention regularization term, encouraging the model to generate compact feature vectors. We demonstrate the superiority of our model over existing state-of-the-art deep learning models on the FER2013 dataset regarding emotion classification accuracy.

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
   
        pip install -r requirements.txt

5. put data into /data folder

# Pipeline:

1. Train 5 model wight
2. use tester to inference 5 csv (output{i}.csv)
3. use vote_csv.py to get vote result

# Command

## Training Command:
python train.py \
     --dataset-path='data/Images' \
     --batch-size=64 --lr=0.0001 \
     --epochs=300 \
     --amp \
     --in_22k \
     --num-workers=4 \
     --model-size='base' \
     --output-dir 'out' \
     --device
## Testing Command:
python tester.py \
     --dataset-path='data/Images' \
     --amp \
     --in_22k \
     --num-workers=1 \
     --model-size='base'
## Vote Commad:
python csv_vote.py





