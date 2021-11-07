# VFedPCA+VFedAKPCA
This is the official source code for the Paper: **Vertical Federated Principal Component Analysis and Its Kernel Extension on Feature-wise Distributed Data** based on [Pytorch](https://pytorch.org/) Framework. 

> Despite enormous research interest and rapid application of federated learning (FL) to various areas, existing studies mostly focus on supervised federated learning under the horizontally partitioned local dataset setting. This paper will study the unsupervised FL under the vertically partitioned dataset setting.

## Server-Clients Architecture
<p align="center">
  <img src="figs/sc_arc.png" alt="Server-Clients Architecture" width="600">
  <br>
  <b>Figure</b>: Server-Clients Architecture
</p>

## Master Branch
```
VFedPCA+VFedAKPCA                    
├── case                    // the configuration of the model
    ├── figs                   // save 
    ├── main.py          
    ├── model.py              
    └── utils.py                 
├── dataset                      // after download the dataset, put it on this folder
└── figs   
    ├── dataload              
    └── Train                     // run  
├── README.md               
├── main.py                   // transform .pth model to .onnx model
├── model.py                   // simply use for inference
└── utils.py                    // the information of how to run our Model 
```

## Environments

- python = 3.8.8
- numpy = 1.20.1
- pandas = 1.2.4
- scikit-learn = 0.24.1
- scipy = 1.6.2
- imageio = 2.9.0

## Prepare Dataset

Experiment Description

The main function file of the algorithm is: utils.py, the rest are supporting files. If you need run this code, please import utils first. A brief explanation of files is shown below:
[]()

Data
We use real datasets including structured datasets and image datasets. Experiment 1,2,3 use structured datasets and Experiment 4 uses image datasets for testing. You can also use other datasets for testing, just adjust the input size of the dataset.

Tips
Datasets used here are relatively small, which will not take too much time for running. The actual running time of the code is determined by the number of data samples. If the datasets with a larger number of samples, the running time of VFedPCA and VFedAKPCA algorithm will be relatively long.

you need to create a folder named dataset firstly

```bash
├── dataset                 
    ├── train                // the path for train images 
    ├── test                // the path for test images 
    └── label               // the path for label images
```

**Step 1: Download dataset from the Google Drive URL: https://data.vision.ee.ethz.ch/cvl/DIV2K/**
```bash
$ wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip 
$ wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
```

**Step 2: Crop your each images in datasets to low resolution images, such as 64 x 64** 
```bash
$ python crop_img.py --src ./dataset/DIV2K_train  --dst ./dataset/DIV2K_train_crop # as /dataset/train
$ python crop_img.py --src ./dataset/DIV2K_valid  --dst ./dataset/DIV2K_valid_crop  # as /dataset/test

# Step 3: Using the traditional image processing method Bicubic + Sharpening to get Super Resolution images as label dataset
# $ python inter_img.py --src ./dataset/DIV2K_train_crop --dst ./dataset/DIV2K_train_label
```
## Training Models
```bash
python train.py -net tinynet(default)
                -path ./dataset(default)   
                [-b 32]   
                [-warm 1]   
                [-lr 0.01]  
```

## Test Models
```bash
python test.py -net tinynet(default)  
               -weight ./checkpoint/*.pth  
               -path ./demo/images   
               -result ./demo/result   
               [-b 32]  
               [-rgb]   
```

## Demo Results
