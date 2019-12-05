import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM, SSIM_attention_loss
from networks import *


parser = argparse.ArgumentParser(description="APRN_r_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainL",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--num_mask", type=int, default=2, help='number of masks in attention map')
parser.add_argument("--which_mask", type=int, default=0, help='which mask is used in attention map') 
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = APRN_r(num_mask=opt.num_mask, recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu, which_mask=opt.which_mask)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    #  criterion = SSIM()
    criterion = SSIM_attention_loss()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, _, _, mask_list = model(input_train)


            ### TODO:
            ### out_train - processed image
            ### target_train - ground truth
            ### input_train - original image
            ssim_metric, attention_loss, ssim_mask_metric  = criterion(out_train, target_train, input_train, mask_list)

            #  attention_loss = torch.log10(attention_loss)/10
            #  loss = -torch.add(ssim_metric, attention_loss).cuda()
            #  loss = -torch.add(ssim_metric, ssim_mask_metric*0.01).cuda()
            loss = -ssim_metric

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train, _, _, _= model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, ssim_metric: %.4f, attention_loss: %.4f, ssim_mask_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), ssim_metric.item(), attention_loss.item(), ssim_mask_metric.item(), psnr_train))

            ### gen gt_mask
            diff = torch.mean(torch.abs(target_train - input_train), dim=1, keepdim=True)
            gt_mask = (diff > 0.3).float()
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('ssim_metric', ssim_metric.item(), step)
                writer.add_scalar('attention_loss', attention_loss.item(), step)
                writer.add_scalar('ssim_mask_metric', ssim_mask_metric.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)

                gt_mask_grid = utils.make_grid(gt_mask.data, nrow=8, normalize=False, scale_each=False)
                writer.add_image('gt_mask', gt_mask_grid, step)

                for idx, attention_map in enumerate(mask_list):
                    if opt.use_gpu:
                        mask_label = attention_map.data #.cpu().numpy().squeeze()   #back to cpu
                    else:
                        mask_label = attention_map.data #.numpy().squeeze()
                    #  print("batch_size: {}, c: {}, r: {}, h: {}".format(attention_map.size(0), attention_map.size(1), attention_map.size(2), attention_map.size(3)))
                    im_mask = utils.make_grid(mask_label, nrow=8, normalize=False, scale_each=False)
                    writer.add_image('mask_{}'.format(idx), im_mask, step)
            step += 1
        ## epoch training end

        # log the images
        model.eval()
        out_train, _, _, mask_list = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)
        #  for idx, attention_map in enumerate(mask_list):
        #      if opt.use_gpu:
        #          mask_label = attention_map.data #.cpu().numpy().squeeze()   #back to cpu
        #      else:
        #          mask_label = attention_map.data #.numpy().squeeze()
        #      #  print("batch_size: {}, c: {}, r: {}, h: {}".format(attention_map.size(0), attention_map.size(1), attention_map.size(2), attention_map.size(3)))
        #      im_mask = utils.make_grid(mask_label, nrow=8, normalize=False, scale_each=False)
        #      writer.add_image('mask_{}'.format(idx), im_mask, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        elif opt.data_path.find('raindrop') != -1:
            prepare_data_raindrop(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
