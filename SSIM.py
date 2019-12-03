import torch
import tensorflow as tf
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import matplotlib.pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)  
    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM_attention_loss(torch.nn.Module):

    def __init__(self, window_size = 11, loss_decay = 0.8, size_average = True):
        super(SSIM_attention_loss, self).__init__()
        self.loss_decay = loss_decay
        self.threshold = 0.3
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def cal_ssim_loss(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)



    ## TODO: main compute attention rnn loss
    def compute_attention_rnn_loss(self, img1, img2, img3, mask_list):

        diff = torch.mean(torch.abs(img2 - img3), dim=1, keepdim=True)
        mask_label = (diff > self.threshold).float()

        ## To see the 'mask_label' in a image figure, comment out below
        #  np_mask = mask_label.cpu()
        #  np_mask = np_mask.numpy()
        #  plt.imshow(np_mask[0].squeeze()) # shows only the first mask of a batch
        #  plt.show()

        loss = torch.zeros([1], dtype=torch.float32).cuda()
        n = len(mask_list)
        for index, attention_map in enumerate(mask_list):
                loss_decay = torch.Tensor([self.loss_decay]).cuda()
                exponent = torch.Tensor([n-index+1]).cuda()
                mse_loss = torch.pow(loss_decay, exponent)* F.mse_loss(attention_map,mask_label).cuda()
                loss = torch.add(loss, mse_loss).cuda()

        ssim_mask_loss = self.cal_ssim_loss(mask_list[-1], mask_label)
        return loss, mask_list[-1], ssim_mask_loss


    def forward(self, img1, img2, img3, mask_list):
        """
        -- compute_attentive_rnn_loss:
        :img1 - processed image (deraindroped?))
        :img2 - target image (clean image, ground truth)
        :img3 - unprocessed image - original raindrop image
        :mask_list - generated mask list
        """
        ssim_loss = self.cal_ssim_loss(img1, img2)
        #  print("ssim loss")
        #  print(ssim_loss)

        ## TODO::
        attention_loss, _, ssim_mask_metric = self.compute_attention_rnn_loss(img1, img2, img3, mask_list)
        #  print("attention loss")
        #  print(attention_loss)


        return ssim_loss,  attention_loss, ssim_mask_metric
