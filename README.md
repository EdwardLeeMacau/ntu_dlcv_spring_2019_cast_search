# Final Project - Cast Search by Portrait Challenge
Final Project #2  
Deep Learning for Computer Vision (107-2) Group 36  

## Task Definition
Given an image of a target query cast and some candidates from gallery, we are requested to search for all the instances belonging to that cast, and sort the result by confidence.

## Solutions and Experiment
1. Best Solution : Considered as Human Re-Identification and Re-Ranking problem, Extracting features to calculate distance between people.
2. Naive Solution : Considered as Classification problem, classifing candidates into different class of people.

## Workflow for best solution
1. Dataset Preprocessing : Cropping faces with pre-trained detection model "XXXX model" (refference : ).
2. Model Design : Resnet50 with Imagenet or VGGFace2 pre-trained fixed weights, and self-designed further features extracting layers.
3. Training : With Triplet Loss Function, train the model to be able to distinguish the distance between two features from diffrernt faces.
4. Inferencing : Extract features of all casts and candidates with trained model, and calaulate distance between all casts and candidates with either cosine similarity function or re-ranking function. Output the ranking order as csv format finally.
5. Validation : When inferencing on validation dataset, we has ground truth to calculate mAP scores to measure accuracy of our model.

## Workflow for naive solution
1. 
2. 
3. 



# Usage of codes to reproduce our results
### 0. Cloning th project repository
To start working on this project, clone this repository into your local machine by using the following command.

    git clone https://github.com/dlcv-spring-2019/final-yuchiang_little_fans-list.git

### 1. Downloading Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the origin dataset for this project. For Linux users, simply use the following command.

    sh ./get_dataset.sh IMDB
The shell script will automatically download the dataset and store the data in a folder called `IMDB`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/drive/folders/1GItzg9wJBiPFrDPBUXQdZgs1ac0Wwbju?usp=sharing
) and unzip the compressed file manually.

### 2.0 Dataset Preprocessing : Cropping
TODO :
description : 
command : 

    python3 ......

### 2.1 Download Cropped dataset directly
Because cropping dataset is very time consuming, you could download cropped dataset directly from google drive through the following command.

    sh get_resize_data.sh

TODO : shell script has to makedir ./IMDb_resize/ folder, unzip the zip files, and rm the zip files.

### 3. Training
Before starting training, download the pre-trained model first through the following command.

    sh get_res50model.sh
This command would download the pre-trained model to folder `pretrain`.

Then, to reproduce our best model, train the model with the following command.
  
    python3 train.py ........

TODO : finish the above args

### 4. Inferencing

Please run the code below to inference:

    python3 inference_csv.py 

### 5. Validation : Inference on val set, experimenting params :
|  k1   |  k2   | lambda | mAP with cosine similarity | mAP with reranking |
| :---: | :---: | :----: | :------------------------: | :----------------: |
|  20   |   6   |  0.15  |             x              |         x          |
|  20   |  10   |  0.15  |             x              |         x          |
|  40   |   6   |  0.15  |             x              |         x          |
|  ...  |  ...  |  ...   |            ...             |        ...         |


### 6. Visualization




### Packages
Below is a list of packages you are allowed to import in this assignment:

> [`python`](https://www.python.org/): 3.5+  
> [`torch`](https://pytorch.org/): 1.0  
> [`h5py`](https://www.h5py.org/): 2.9.0  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.0  
> [`torchvision`](https://pypi.org/project/torchvision/): 0.2.2  
> [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/), [`skimage`](https://scikit-image.org/), [`Pillow`](https://pillow.readthedocs.io/en/stable/), [`scipy`](https://www.scipy.org/)  
> [The Python Standard Library](https://docs.python.org/3/library/)

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above.


### Referrence

1. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)

2. [Re-ranking Person Re-identification with K-reciprocal Encoding](http://openaccess.thecvf.com/content_cvpr_2017/papersZhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)

3. [face.evoLVe: High-Performance Face Recognition Library based on PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)