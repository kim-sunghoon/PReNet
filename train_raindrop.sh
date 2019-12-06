#! bash

batch=18
data_path=datasets/train/raindrop
## APRN_r
# echo "train ARPN6_r_2_l1"
# echo "python train_APRN_r.py --save_path logs/raindrop/APRN6_r --data_path datasets/train/raindrop2 --batch_size 18 --gpu_id 1"
# python train_APRN_r.py --save_path logs/raindrop/APRN6_r --data_path datasets/train/raindrop2 --batch_size 18 --gpu_id 1
# echo ", which is already done! where loss is ssim"
# echo "Too large attention module... "
# attention2_save_path=logs/raindrop/APRN6_r_2
# echo "python train_APRN_r.py --save_path $attention2_save_path --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_save_path --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path=logs/raindrop/APRN6_r_2_l1
echo "Now, start adapt different loss function"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path2=logs/raindrop/APRN6_r_2_l2
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path2 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path2 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path3=logs/raindrop/APRN6_r_2_l3
# echo "train $attention2_w_loss_save_path3"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path3 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path3 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path4=logs/raindrop/APRN6_r_2_l4
# echo "train $attention2_w_loss_save_path4"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path4 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path4 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path5=logs/raindrop/APRN6_r_2_l5
# echo "train $attention2_w_loss_save_path5"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path5 --data_path $data_path --batch_size $batch --gpu_id 3"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path5 --data_path $data_path --batch_size $batch --gpu_id 3

# attention2_w_loss_save_path6=logs/raindrop/APRN6_r_2_l6
# echo "train $attention2_w_loss_save_path6"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path6 --data_path $data_path --batch_size $batch --gpu_id 2"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path6 --data_path $data_path --batch_size $batch --gpu_id 2

# attention2_w_loss_save_path7=logs/raindrop/APRN6_r_2_l7
# echo "train $attention2_w_loss_save_path7"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path7 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path7 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path8=logs/raindrop/APRN6_r_2_l8
# echo "train $attention2_w_loss_save_path8"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path8 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path8 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path9=logs/raindrop/APRN6_r_2_l9
# echo "train $attention2_w_loss_save_path9"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path9 --data_path $data_path --batch_size $batch --gpu_id 1"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path9 --data_path $data_path --batch_size $batch --gpu_id 1

# attention2_w_loss_save_path10=logs/raindrop/APRN6_r_4_l10
# echo "train $attention2_w_loss_save_path10"
# echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path10 --data_path $data_path --batch_size $batch --gpu_id 3 --num_mask 4 --which_mask 3"
# python train_APRN_r.py --save_path $attention2_w_loss_save_path10 --data_path $data_path --batch_size $batch --gpu_id 3 --num_mask 4 --which_mask 3

attention2_w_loss_save_path11=logs/raindrop/APRN6_r_4_l11
echo "train $attention2_w_loss_save_path11"
echo "python train_APRN_r.py --save_path $attention2_w_loss_save_path11 --data_path $data_path --batch_size $batch --gpu_id 2 --num_mask 2 --which_mask 1"
python train_APRN_r.py --save_path $attention2_w_loss_save_path11 --data_path $data_path --batch_size $batch --gpu_id 2 --num_mask 2 --which_mask 1

#########################################################################################
## PRN
# python train_PRN.py --save_path logs/raindrop/PRN6 --data_path datasets/train/raindrop --batch_size $batch --gpu_id 0 --preprocess True
# python train_PRN.py --save_path logs/raindrop/PRN6 --data_path datasets/train/raindrop1 --batch_size $batch --gpu_id 0

## PRN_r
# python train_PRN_r.py --save_path logs/raindrop/PRN6_r --data_path datasets/train/raindrop2 --batch_size $batch --gpu_id 0

## PReNet
# python train_PReNet.py --save_path logs/raindrop/PReNet6 --data_path datasets/train/raindrop3 --batch_size $batch --gpu_id 0

## PReNet_r
# python train_PReNet_r.py --save_path logs/raindrop/PReNet6_r_test --data_path datasets/train/raindrop4 --batch_size $batch --gpu_id 0
