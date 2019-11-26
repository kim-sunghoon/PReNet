#! bash

###
# echo "APRN_r a and b statistic"
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_b

echo "APRN_r_2 a and b statisics"
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_a
# python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_b

echo "APRN_r_2_diffent loss with test a and b "
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_a/gt --result_dir results/raindrop/APRN_r_2_l1_a
python ./statistic_psnr_ssim.py --gt_dir datasets/test/raindrop_test_b/gt --result_dir results/raindrop/APRN_r_2_l1_b

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
