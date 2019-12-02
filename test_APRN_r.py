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

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/datasets/test/raindrop_test_a/data", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/r/works/derain_arxiv/release/results/PReNet", help='path to save results')
parser.add_argument('--gt_dir', type=str, 
    choices = gt_dirs,
    default='datasets/test/raindrop_test_a/gt', help='ground truth dir \n' + ' | '.join(gt_dirs))
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--num_mask", type=int, default=2, help='number of masks in attention map')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id



#  def save_mask_plt(img1, img2, img3, mask_list, mask_save_path, img_name, img_ext, threshold=0.3):
def save_mask_plt(mask_list, mask_save_path, img_name, img_ext, threshold=0.3):
    """
    -- compute_attentive_rnn_loss:
    :img1 - processed image (deraindroped?))
    :img2 - target image (clean image, ground truth)
    :img3 - unprocessed image - original raindrop image
    :mask_list - generated mask list
    """

    #  diff = torch.mean(torch.abs(img2 - img3), dim=1, keepdim=True)
    #  mask_label = (diff > threshold).float()
    #
    #  if opt.use_GPU:
    #      mask_label = mask_label.data.cpu().numpy()   #back to cpu
    #  else:
    #      mask_label = mask_label.data.numpy()

    #  ## To see the 'mask_label' in a image figure, comment out below
    #  np_mask = mask_label.cpu()
    #  np_mask = np_mask.numpy()
    #  #  num_mask = len(np_mask)
    #  plt.imshow(np_mask[0].squeeze()) # shows only the first mask of a batch
    #  plt.show()

    for idx, attention_map in enumerate(mask_list):
        #  plt.imshow(mask_label[idx].squeeze())
        #  plt.savefig(os.path.join(mask_save_path, "{}_mask{}{}".format(img_name, idx, img_ext)))
        if opt.use_GPU:
            mask_label = attention_map.data.cpu().numpy().squeeze()   #back to cpu
        else:
            mask_label = attention_map.data.numpy().squeeze()
        plt.imshow(mask_label)
        #  plt.show()
        plt.savefig(os.path.join(mask_save_path, "{}_mask{}{}".format(img_name, idx, img_ext)))



def main():

    os.makedirs(opt.save_path, exist_ok=True)
    mask_save_path = os.path.join(opt.save_path, "attention_map")
    os.makedirs(mask_save_path, exist_ok=True)

    summary_csv_name = os.path.join(opt.save_path, "{}_inf_time.csv".format(opt.logdir.split("/")[-1]))
    with open(summary_csv_name, "w") as csv_out:
        csv_out.write("img,inf_time\n")

    # Build model
    print('Loading model ...\n')
    model = APRN_r(opt.num_mask, opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()

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

            if opt.use_GPU:
                y = y.cuda()
                #  gt = gt.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _, _, mask_list = model(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name + ext, ': ', dur_time)
                csv_out.write("{},{}".format(img_name, dur_time))
                #  for idx, mask in enumerate(mask_list):
                    #  csv_out.write("{}".format(mask))
                    #  if (idx-1) != len(mask_list):
                    #      csv_out.write(",")

                csv_out.write("\n")

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, "{}{}".format(img_name, ext)), save_out)
            #  save_mask_plt(img1=out, img2=gt, img3=y, mask_list=mask_list, mask_save_path=mask_save_path, img_name=img_name, img_ext=ext)
            save_mask_plt(mask_list=mask_list, mask_save_path=mask_save_path, img_name=img_name, img_ext=ext)

            count += 1

    print('Avg. time:', time_test/count)
    csv_out.write("avg_{}imgs, {}\n".format(count, time_test/count))
    csv_out.close()


if __name__ == "__main__":
    main()

