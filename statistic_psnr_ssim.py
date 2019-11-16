import os
import argparse
import glob
import cv2
import skimage
import natsort
import numpy as np
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim

gt_dirs = ['datasets/test/raindrop_test_a/gt', 'datasets/test/raindrop_test_b/gt']

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def calc_psnr(im1, im2):
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)
    return compare_ssim(im1_y, im2_y, multichannel=True)

def parse_opt():
    parser = argparse.ArgumentParser(description="evaluation metric!")
    parser.add_argument('--gt_dir', type=str, 
            choices = gt_dirs,
            default='./datasets/test/raindrop_test_a/gt', help='ground truth dir \n' + ' | '.join(gt_dirs))
    parser.add_argument('--result_dir', type=str, default='./results/raindrop/PRN6_a', help='test result files')
    parser.add_argument('--store_eval_dir', type=str, default='store_eval_dir')
    print("gt a and result a should be matched")
    return parser.parse_args()

if __name__ == "__main__":

    opts = parse_opt()
    print(opts)
    create_dir(opts.store_eval_dir)

    gt_list = glob.glob(os.path.join(opts.gt_dir, "*"))
    gt_list = natsort.natsorted(gt_list, reverse=False)
    processed_list = glob.glob(os.path.join(opts.result_dir, "*"))
    processed_list = natsort.natsorted(processed_list, reverse=False)

    print("gt_list: {}, processed_list: {}".format(len(gt_list), len(processed_list)))
    assert len(gt_list) == len(processed_list)

    store_csv_name = os.path.join(opts.store_eval_dir, "{}.csv".format(opts.result_dir.split("/")[-1]))

    csv = open(store_csv_name, 'w')
    csv.write("gt,processed,psnr,ssim\n")
    csv.close()

    accum_psnr = np.zeros(1)
    accum_ssim = np.zeros(1)

    csv = open(store_csv_name, 'a')

    for gt, processed in tqdm(zip(gt_list, processed_list)):
        psnr = calc_psnr(gt, processed)
        ssim = calc_ssim(gt, processed)
        accum_psnr = accum_psnr + psnr
        accum_ssim = accum_ssim + ssim
        csv.write("{},{},{:.3f},{:.3f}\n".format(gt.split("/")[-1], processed.split("/")[-1], psnr, ssim))

    avg_psnr = accum_psnr/len(processed_list)
    avg_ssim = accum_ssim/len(processed_list)

    csv.write("Cal_avg,t {} files,{:.3f},{:.3f}\n".format(len(processed_list), avg_psnr[0], avg_ssim[0]))
    csv.close()


