#! bash

###
### for test_a
# PRN
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PRN6_a

# PRN_r
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PRN_r_a

# PReNet
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PReNet_a

# PReNet_r
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/PReNet_r_a


### for test_b
# PRN
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PRN6_b

# PRN_r
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PRN_r_b

# PReNet
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PReNet_b

# PReNet_r
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/PReNet_r_b

