#! bash 

### for test_a
# PRN
python test_PRN.py --logdir logs/raindrop/PRN6 --save_path results/raindrop/PRN6_a --data_path datasets/test/raindrop_test_a/data

# PRN_r
python test_PRN_r.py --logdir logs/raindrop/PRN6_r --save_path results/raindrop/PRN_r_a --data_path datasets/test/raindrop_test_a/data

# PReNet
python test_PReNet.py --logdir logs/raindrop/PReNet6 --save_path results/raindrop/PReNet_a --data_path datasets/test/raindrop_test_a/data

# PReNet_r
python test_PReNet_r.py --logdir logs/raindrop/PReNet6_r --save_path results/raindrop/PReNet_r_a --data_path datasets/test/raindrop_test_a/data


### for test_b
# PRN
python test_PRN.py --logdir logs/raindrop/PRN6 --save_path results/raindrop/PRN6_b --data_path datasets/test/raindrop_test_b/data

# PRN_r
python test_PRN_r.py --logdir logs/raindrop/PRN6_r --save_path results/raindrop/PRN_r_b --data_path datasets/test/raindrop_test_b/data

# PReNet
python test_PReNet.py --logdir logs/raindrop/PReNet6 --save_path results/raindrop/PReNet_b --data_path datasets/test/raindrop_test_b/data

# PReNet_r
python test_PReNet_r.py --logdir logs/raindrop/PReNet6_r --save_path results/raindrop/PReNet_r_b --data_path datasets/test/raindrop_test_b/data

