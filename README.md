# VFedPCA+VFedAKPCA

Python demo code for ''Vertical Federated Principal Component Analysis and Advanced Kernel Principal Component Analysis on Feature-wise Distributed Data''.

## The Framework of Sever-Clients Architecture
![](/Users/sophia/Desktop/c.png)
## Experiment Description

The main function file of the algorithm is: utils.py, the rest are supporting files. If you need run this code, please import utils first. A brief explanation of files is shown below:
[]()

#### Data
We use real datasets including structured datasets and image datasets. Experiment 1,2,3 use structured datasets and Experiment 4 uses image datasets for testing. You can also use other datasets for testing, just adjust the input size of the dataset.
#### Experiment1 
This experiment mainly examines the effect of feature size on VFedPCA:

* utils.py: main function of VFedPCA algorithm, including local power iteration function and federated function.
* -.py: test separately on different structured datasets.

#### Experiment2 
This experiment mainly examines the effect of local iterations on VFedPCA:

* utils.py: main function of VFedPCA algorithm, including local power iteration function and federated function.
* -.py: test separately on different structured datasets.

#### Experiment3 
This experiment mainly examines the effect of warm-start power iterations and weight scaling method:

* -.py: test separately on different structured datasets.

#### Experiment4
The experiment mainly examines the effect of VFedPCA and VFedAKPCA on the image data set and performs Image segmentation tests:

* -.py: test separately on different image datasets.


#### Tips
Datasets used here are relatively small, which will not take too much time for running. The actual running time of the code is determined by the number of data samples. If the datasets with a larger number of samples, the running time of VFedPCA and VFedAKPCA algorithm will be relatively long.
