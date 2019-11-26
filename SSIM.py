import torch
import tensorflow as tf
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
 
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


    ## TODO: need to chnage tf to pytorch
    def build_attentive_rnn(self, img1, mask_list):
        """
        """
        batch_size, row, col = img1.size(0), img1.size(2), img1.size(3)

        init_attention_map = Variable(torch.ones(batch_size, 1, row, col)/2).cuda()
        init_cell_state = Variable(torch.zeros( batch_size, 32, row, col)).cuda()
        init_lstm_feats = Variable(torch.zeros( batch_size, 32, row, col)).cuda()

        for i in range(4):
            attention_input = tf.concat((input_tensor, init_attention_map), axis=-1)
            conv_feats = self._residual_block(input_tensor=attention_input,
                                              name='residual_block_{:d}'.format(i + 1))
            lstm_ret = self._conv_lstm(input_tensor=conv_feats,
                                       input_cell_state=init_cell_state,
                                       name='conv_lstm_block_{:d}'.format(i + 1))
            init_attention_map = lstm_ret['attention_map']
            init_cell_state = lstm_ret['cell_state']
            init_lstm_feats = lstm_ret['lstm_feats']

            attention_map_list.append(lstm_ret['attention_map'])

        ret = {
            'final_attention_map': init_attention_map,
            'final_lstm_feats': init_lstm_feats,
            'attention_map_list': attention_map_list
        }

        return ret

    ## TODO: main compute attention rnn loss
    def compute_attention_rnn_loss(self, img1, img2, mask_list):

        inference_ret = self.build_attentive_rnn(img1, mask_list)

        loss = tf.constant(0.0, tf.float32)
        n = len(inference_ret['attention_map_list'])
        for index, attention_map in enumerate(inference_ret['attention_map_list']):
            mse_loss = tf.pow(0.8, n - index + 1) * \
                       tf.losses.mean_squared_error(labels=label_tensor,
                                                    predictions=attention_map)
            loss = tf.add(loss, mse_loss)

        return loss, inference_ret['final_attention_map']


    def forward(self, img1, img2, mask_list):
        """
        -- compute_attentive_rnn_loss:
        :img1 - input (raindrop image)
        :img2 - target image (clean image)
        :mask_list - generated mask list
        """
        ssim_loss = self.cal_ssim_loss(img1, img2)
        print("ssim loss")
        print(ssim_loss)

        ## TODO::
        attention_loss, _ = self.compute_attention_rnn_loss(img1, img2, mask_list)


        return ssim_loss + attention_loss
