#! bash

# Rain100H
# python train_PRN.py --save_path logs/Rain100H/PRN --data_path datasets/train/RainTrainH --gpu_id 0 --preprocess False

# Rain100L
# python train_PRN.py --save_path logs/Rain100L/PRN --data_path datasets/train/RainTrainL --gpu_id 1 --preprocess False
#
# # Rain12600
# python train_PRN.py --save_path logs/Rain1400/PRN --data_path datasets/train/Rain12600 --gpu_id 2 --preprocess False

## rain drop
python train_PRN.py --save_path logs/Raindrop/PRN --data_path datasets/train/raindrop --gpu_id 3 --preprocess True
