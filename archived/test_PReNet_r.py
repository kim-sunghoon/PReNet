import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time 
import matplotlib.pyplot as plt
import glob
import natsort
from tqdm import tqdm

gt_dirs = ['datasets/test/raindrop_test_a/gt', 'datasets/test/raindrop_test_b/gt']

parser = argparse.ArgumentParser(description="PReNet_r Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/media/r/BC580A85580A3F20/dataset/rain/peku/Rain100H/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet", help='path to save results')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--gt_dir', type=str, 
    choices = gt_dirs,
    default='datasets/test/raindrop_test_a/gt', help='ground truth dir \n' + ' | '.join(gt_dirs))
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet_r(opt.recurrent_iter, opt.use_gpu)
    #  model = PRN(opt.recurrent_iter, opt.use_gpu)
    print_network(model)
    if opt.use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()
    summary_csv_name = os.path.join(opt.save_path, "{}_inf_time.csv".format(opt.logdir.split("/")[-1]))
    with open(summary_csv_name, "w") as csv_out:
        csv_out.write("img,inf_time\n")

    time_test = 0
    count = 0
    csv_out = open(summary_csv_name, "a")

    gt_list = glob.glob(os.path.join(opt.gt_dir, "*.jpg"))
    gt_list.extend(glob.glob(os.path.join(opt.gt_dir, "*.png")))
    gt_list = natsort.natsorted(gt_list, reverse=False)
    input_list = glob.glob(os.path.join(opt.data_path, "*.jpg"))
    input_list.extend(glob.glob(os.path.join(opt.data_path, "*.png")))
    input_list = natsort.natsorted(input_list, reverse=False)

    print("gt_list: {}, input_list: {}".format(len(gt_list), len(input_list)))
    assert len(gt_list) == len(input_list)

    #  print(gt_list)
    #  print(input_list)

    for img_path, gt_path  in zip(input_list, gt_list):
        if is_image(img_path):
            #  img_path = os.path.join(opt.data_path, img_name)
            ext = os.path.splitext(img_path)[-1]
            img_name = os.path.splitext(img_path)[0].split("/")[-1]
            #  print(img_name, ext)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            ## gt image 
            #  gt = cv2.imread(gt_path)
            #  b, g, r = cv2.split(gt)
            #  gt = cv2.merge([r, g, b])
            #  gt = normalize(np.float32(gt))
            #  gt = np.expand_dims(gt.transpose(2, 0, 1), 0)
            #  gt = Variable(torch.Tensor(gt))

            if opt.use_gpu:
                y = y.cuda()
                #  gt = gt.cuda()

            with torch.no_grad(): #
                if opt.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                #  out, _, _, mask_list = model(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name + ext, ': ', dur_time)
                csv_out.write("{},{}\n".format(img_name, dur_time))


            if opt.use_gpu:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, "{}{}".format(img_name, ext)), save_out)

            count += 1

    print('Avg. time:', time_test/count)
    csv_out.write("avg_{}imgs, {}\n".format(count, time_test/count))
    csv_out.close()



if __name__ == "__main__":
    main()

