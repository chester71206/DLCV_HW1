# NYCU Computer Vision 2026 HW1

-   **Student ID:** 314551178
-   **Name:** 陳鎮成 (chester)

## Introduction

The objective is to solve a 100-class plant image classification problem.

To maximize feature extraction capacity within the 100M parameter
constraint, the core methodology utilizes a heavily modified
**ResNet-152** as the backbone network. The original classifier was
replaced with **GeM (Generalized Mean) Pooling** and a massive custom
Multi-Layer Perceptron (MLP) head with GELU activations and strong
Dropout. The training pipeline is further enhanced by robust data
augmentation strategies, including **Mixup**, **AutoAugment**, and
**Random Erasing**, optimized via the Cosine Annealing with Warm
Restarts scheduler.

## Environment Setup

It is recommended to use Python 3.9 or higher. You can install all the
required dependencies using the provided `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

## Usage

### Training

execute the advanced training script:

``` bash
python train_advanced.py
```

### Inference

The inference script automatically applies Test-Time Augmentation (TTA) by averaging probabilities across original,
horizontally flipped, and vertically flipped images. Run the following
command:

``` bash
python inference.py
```


### Performance Snapshot
<img width="1435" height="281" alt="image" src="https://github.com/user-attachments/assets/43b24bfd-5692-4a3d-b747-4eab81a23762" />

