#! bash 
# echo "APRN_r with test a"
# python test_APRN_r.py --logdir logs/raindrop/APRN6_r --save_path results/raindrop/APRN_r_a --data_path datasets/test/raindrop_test_a/data
# # PRN_r
# echo "APRN_r with test b"
# python test_APRN_r.py --logdir logs/raindrop/APRN6_r --save_path results/raindrop/APRN_r_b --data_path datasets/test/raindrop_test_b/data
#
# attention2_save_path=logs/raindrop/APRN6_r_2
# echo "APRN_r_2 with test a"
# python test_APRN_r.py --logdir $attention2_save_path --save_path results/raindrop/APRN_r_2_a --data_path datasets/test/raindrop_test_a/data
# # PRN_r
# echo "APRN_r_2 with test b"
# python test_APRN_r.py --logdir $attention2_save_path --save_path results/raindrop/APRN_r_2_b --data_path datasets/test/raindrop_test_b/data

echo "raise Not implemented error"
attention2_w_loss_save_path=logs/raindrop/APRN6_r_2_l1
echo "APRN_r_2_diffent loss with test a $attention2_w_loss_save_path"
python test_APRN_r.py --logdir $attention2_w_loss_save_path --save_path results/raindrop/APRN_r_2_l1_a --data_path datasets/test/raindrop_test_a/data
echo "APRN_r_2_diffent loss with test b $attention2_w_loss_save_path"
python test_APRN_r.py --logdir $attention2_w_loss_save_path --save_path results/raindrop/APRN_r_2_l1_b --data_path datasets/test/raindrop_test_b/data

# ### for test_a
# # PRN
# echo "PRN with test a"
# python test_PRN.py --logdir logs/raindrop/PRN6 --save_path results/raindrop/PRN6_a --data_path datasets/test/raindrop_test_a/data
#
# # PRN
# echo "PRN with test b"
# python test_PRN.py --logdir logs/raindrop/PRN6 --save_path results/raindrop/PRN6_b --data_path datasets/test/raindrop_test_b/data
#
# # PRN_r
# echo "PRN_r with test a"
# python test_PRN_r.py --logdir logs/raindrop/PRN6_r --save_path results/raindrop/PRN_r_a --data_path datasets/test/raindrop_test_a/data
# # PRN_r
# echo "PRN_r with test b"
# python test_PRN_r.py --logdir logs/raindrop/PRN6_r --save_path results/raindrop/PRN_r_b --data_path datasets/test/raindrop_test_b/data
#
# # PReNet
# echo "PReNet with test a"
# python test_PReNet.py --logdir logs/raindrop/PReNet6 --save_path results/raindrop/PReNet_a --data_path datasets/test/raindrop_test_a/data
# # PReNet
# echo "PReNet with test b"
# python test_PReNet.py --logdir logs/raindrop/PReNet6 --save_path results/raindrop/PReNet_b --data_path datasets/test/raindrop_test_b/data
#
#
# # PReNet_r
# echo "PReNet_r with test a"
# python test_PReNet_r.py --logdir logs/raindrop/PReNet6_r --save_path results/raindrop/PReNet_r_a --data_path datasets/test/raindrop_test_a/data
# # PReNet_r
# echo "PReNet_r with test b"
# python test_PReNet_r.py --logdir logs/raindrop/PReNet6_r --save_path results/raindrop/PReNet_r_b --data_path datasets/test/raindrop_test_b/data
#
#
