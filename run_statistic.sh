#! bash

###
# echo "APRN_r a and b statistic"
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_b

# echo "APRN_r_2 a and b statisics"
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_b
#
# echo "APRN_r_2_diffent loss1 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l1_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l1_b
#

# echo "APRN_r_2_diffent loss2 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l2_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l2_b

# echo "APRN_r_2_diffent loss3 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l3_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l3_b

# echo "APRN_r_2_diffent loss4 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l4_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l4_b

# echo "APRN_r_2_diffent loss5 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l5_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l5_b

echo "APRN_r_2_diffent loss6 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_b

# echo "APRN_r_2_diffent loss6 iteration4 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_i4_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_i4_b

echo "APRN_r_2_diffent loss6 iteration2 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_i2_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_i2_b

echo "APRN_r_2_diffent loss6 iteration1 with test a and b "
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_i1_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_i1_b

# echo "APRN_r_2_diffent loss6 iteration1 with test a and b  wm = $wm"
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_b
### 12.05 13:40 
num_mask=4
wm=0
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_tm${num_mask}_a
wm=1
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_tm${num_mask}_a
wm=2
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_tm${num_mask}_a
wm=3
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_tm${num_mask}_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l6_m${wm}_b

############# 
### for test_a
# PRN
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PRN6_a
#
# # PRN_r
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PRN_r_a
#
# # PReNet
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PReNet_a
#
# # PReNet_r
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PReNet_r_a
#
#
# ### for test_b
# # PRN
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PRN6_b
#
# # PRN_r
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PRN_r_b
#
# # PReNet
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PReNet_b
#
# # PReNet_r
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PReNet_r_b
#
