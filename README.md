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
├── case                        // Case Studies
    ├── figs                    // Save experimental results' figures in '.eps' / '.png' format 
        ├── img_name*.eps              
        └── img_name*.png           
    ├── main.py          
    ├── model.py              
    └── utils.py                 
├── dataset                     // Put downloaded dataset in this folder
└── figs                        // Save experimental results' figures in '.eps' / '.png' format
    ├── img_name*.eps              
    └── img_name*.png           
├── README.md               
├── main.py                     // Experiment on Structured Dataset
├── model.py                   
└── utils.py                     
```

## Environments

- python = 3.8.8
- numpy = 1.20.1
- pandas = 1.2.4
- scikit-learn = 0.24.1
- scipy = 1.6.2
- imageio = 2.9.0

## Prepare Dataset
To demonstrate the superiority of our method, we utilized FIVE types of real-world datasets coming with distinct nature.  
1) structured datasets from different domains; 
2) medical image dataset; 
3) face image dataset; 
4) gait image dataset; 
5) person re-identification image dataset

**Step 1: Download Dataset from the [Google Drive URL](https://drive.google.com/drive/folders/1Rv_a02tBygvbO8FY05XxsY_lhXLiHQj6?usp=sharing)**

**Step 2: Specify Dataset Path by Command Argument** 

```bash
$ python main.py --data_path="./dataset/xxx"
```

## Experiments
We conduct extensive experiments on structured datasets to exmaines the effect of feature size, local iterations, warm-start power iterations, and weight scaling method on structed datasets. Furthermore, we investigate some case studies with image dataset to demonstrate the effectiveness of VFedPCA and VFedAKPCA.

### A. Experiment on Structured Dataset
First, you need to choose the dataset.
```bash
python main.py --data_path './dataset/College.csv' --batch_size 160 
```
Then, you only need to set different `flag`, `p_list`, `iter_list` and `sampler_num` to exmaines the effect of feature size, local iterations, warm-start power iterations, and weight scaling method on structed datasets. The example is as follows:
```
flag ='clients'
p_list = [3, 5, 10]         # the number of involved clients
iter_list = [100, 100, 100] # the number of local power iterations
sampler_num = 5
```

### B. Case Studies
```bash
python main.py --data_path '../dataset/Image/DeepLesion' --client_num 8 --iterations 100 --re_size 512
```

## Citation
```
@inproceedings{
title = {{Vertical Federated Principal Component Analysis and Its Kernel Extension on Feature-wise Distributed Data}},
author = {Yiu-ming Cheung, Fellow, IEEE, Feng Yu, and Jian Lou},
year = 2021
}
```