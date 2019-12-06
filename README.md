## Raindrop-removal Project 

Progressive Image Deraining Networks: A Better and Simpler Baseline 


### Abstract

## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)



## Datasets
The whole dataset can be find here:
https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K

####Training Set:
861 image pairs for training.

####Testing Set A:
For quantitative evaluation where the alignment of image pairs is good. A subset of testing set B.

####Testing Set B:
239 image pairs for testing.

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

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
sh test_raindrop.sh      
```
All the results in the paper are also available at [Onedrive]().
You can place the downloaded results into `./results/`, and directly compute all the [evaluation metrics](store_eval_dir/) in this paper.  

### 2) Evaluation metrics

We also provide the scripts to compute the average PSNR and SSIM values reported in the paper.
```bash
sh run_statistic.sh 
```
which is based on `statistic_psnr_ssim.py`
###
Average PSNR/SSIM values on four datasets:

Dataset    | PRN       |PReNet     |PRN_r      |PReNet_r   |Qian et al.[1] |Ours
-----------|-----------|-----------|-----------|-----------|-----------|-----------
raindrop   |28.07/0.884|29.46/0.899|27.43/0.874|28.98/0.892|x/x|x/x

### 3) Training

Run shell scripts to train the models:
```bash
sh train_APRN_r.sh  
```
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 6             | Number of recursive stages
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

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 6                | Number of recursive stages
num_mask               | 2                | The number of masks used in the attention map
which_mask             | 0                | Which mask is used in the Progressive network
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1]
[2]
[3] Yang W, Tan RT, Feng J, Liu J, Guo Z, Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[4] Li Y, Tan RT, Guo X, Lu J, Brown MS. Rain streak removal using layer priors. In IEEE CVPR 2016.

[5] Fu X, Huang J, Zeng D, Huang Y, Ding X, Paisley J. Removing rain from single images via a deep detail network. In IEEE CVPR 2017.

[6] Li X, Wu J, Lin Z, Liu H, Zha H. Recurrent squeeze-and-excitation context aggregation net for single image deraining.In ECCV 2018.


# Citation

```
 @inproceedings{ren2019progressive,
   title={Progressive Image Deraining Networks: A Better and Simpler Baseline},
   author={Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }
 ```
