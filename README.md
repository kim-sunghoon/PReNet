## CSED 441 Final Project (Fall, 2019)
## Attention and Progressive Network Based Raindrop Removal From a Single Image 
by Sunghoon Kim (20182560) and Jooyeong Yun (20160185)

### Abstract
test 


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- Please type `conda env create -f requirement.yaml`, when you make a virutal enviroment using conda 


## Datasets
The whole dataset can be find here:
https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K
* Training Set: 861 image pairs for training.
* Testing Set A: For quantitative evaluation where the alignment of image pairs is good. A subset of testing set B.
* Testing Set B: 239 image pairs for testing.

Please download and place the unzipped folders into `./datasets/train/` and  `./datasets/test/`. 

Folder tree example:
* For Train 
```
datasets/train
├── raindrop 
    ├── data
        ├── 0_rain.png
        ...
    ├── gt
        ├── 0_clean.png
        ...
```
* For test_a
```
datasets/test/
├── raindrop_test_a 
    ├── data
        ├── 0_rain.png
        ...
    ├── gt
        ├── 0_clean.png
 
```
* For test_b 
```
datasets/test/
├── raindrop_test_b 
    ├── data
        ├── 0_rain.png
        ...
    ├── gt
        ├── 0_clean.png
 
```

## Getting Started
### 1) Training

Run shell scripts to train the models:
```bash
sh train_APRN_r.sh  
```
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### 2) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
sh test_raindrop.sh      
```
All the results in the paper are also available at [Onedrive]().
You can place the downloaded results into `./results/`.

### 3) Evaluation metrics

We also provide the scripts to compute the average PSNR and SSIM values reported in the paper.
```bash
sh run_statistic.sh 
```
which is based on `statistic_psnr_ssim.py`


### 4) Results 
Average PSNR/SSIM values on the raindrop dataset (test_a):

Dataset    |Qian et al.[1] |Ren et al. (PRN_r) [2] |Ours
-----------|-----------|-----------|-----------
raindrop   |31.516/0.921|32.185/0.942|32.808/0.946


### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations (`train_APRN_r.py`)

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 4             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
num_mask               | 2             | The number of masks used in the attention map
which_mask             | 0             | Which mask is used in the Progressive network
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations (`test_APRN_r.py`)

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 4                | Number of recursive stages
num_mask               | 2                | The number of masks used in the attention map
which_mask             | 0                | Which mask is used in the Progressive network
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
gt_path                | N/A              | path to ground truth of testing images
save_path              | N/A              | path to save results

#### Statistic Configurations (`statistic_psnr_ssim.py`)
Option                 |Default        | Description
-----------------------|---------------|------------
gt_dir                 | N/A           | path to ground truth of testing images
result_dir             | N/A           | path to save results (save_path in testing mode)
store_eval_dir         | N/A           | path to save ssim and psnr results 


## References
[1] R. Qian, TF. Tan, W. Yang, J. Su and J. Liu "Attentive Generative Adversarial Network for Raindrop Removal from a Single Image", In IEEE CVPR 2018

[2] D. Ren, W. Zuo, Q. Hu, P. Zhu and D. Meng "Progressive Image Deraining Networks: A Better and Simpler Baseline", In IEEE CVPR 2019.

[3] W. Yang, RT. Tan, J. Feng, J. Liu, Z. Guo and S. Yan "Deep joint rain detection and removal from a single image", In IEEE CVPR 2017.

[4] Y. Li, RT. Tan, X. Guo, J. Lu and MS. Brown "Rain streak removal using layer priors", In IEEE CVPR 2016.

[5] X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley  "Removing rain from single images via a deep detail network", In IEEE CVPR 2017.

[6] X. Li, J. Wu, Z. Lin, H. Liu and H. Zha "Recurrent squeeze-and-excitation context aggregation net for single image deraining", In ECCV 2018.


