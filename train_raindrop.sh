#! bash

## PRN
# python train_PRN.py --save_path logs/raindrop/PRN6 --data_path datasets/train/raindrop1 --batch_size 36 --gpu_id 0
# python train_PRN.py --save_path logs/raindrop/PRN6 --data_path datasets/train/raindrop --batch_size 36 --gpu_id 0 --preprocess True

## PRN_r
# python train_PRN_r.py --save_path logs/raindrop/PRN6_r --data_path datasets/train/raindrop2 --batch_size 36 --gpu_id 1

## PReNet
# python train_PReNet.py --save_path logs/raindrop/PReNet6 --data_path datasets/train/raindrop3 --batch_size 36 --gpu_id 2

## PReNet_r
python train_PReNet_r.py --save_path logs/raindrop/PReNet6_r --data_path datasets/train/raindrop4 --batch_size 36 --gpu_id 3
