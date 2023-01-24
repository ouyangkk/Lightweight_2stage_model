import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as func
from torch.utils import data




def scale(x, mean, std):
    return (x - mean) / (std)

def inv_scale(x, mean, std):
    return (x * std + mean)

def mse_loss(output, target):
    # loss = torch.mean(torch.abs(output - target))
    loss_f2 = nn.MSELoss(size_average=True, reduce=True, reduction='mean' )
    loss = loss_f2(output, target)
    return loss

class Adaptive_weight_loss(nn.Module):
    def __init__(self):
        super(Adaptive_weight_loss, self).__init__()
    def forward(self, weight, obj_speech, ref_specch, res_noise):
        dis_speech_power = torch.mean(torch.mean(torch.pow(obj_speech-ref_specch, 2), dim=-1), dim=-1)
        res_noise_power = torch.mean(torch.mean(torch.pow(res_noise, 2), dim=-1), dim=-1)
        weight_noise = torch.ones_like(weight) - weight
        loss = torch.mean(weight*dis_speech_power + weight_noise*res_noise_power)
        return loss

class Adaptive_weight_loss1(nn.Module):
    def __init__(self):
        super(Adaptive_weight_loss1, self).__init__()
    def forward(self, weight, obj_speech, ref_specch,ref2_speech, res_noise):
        dis_speech_power = torch.mean(torch.mean(torch.pow(obj_speech-ref_specch, 2), dim=-1), dim=-1)
        res_noise_power = torch.mean(torch.mean(torch.pow(res_noise, 2), dim=-1), dim=-1)
        res_clean_power = torch.mean(torch.mean(torch.pow(ref2_speech, 2), dim=-1), dim=-1)
        weight_noise = torch.ones_like(weight) - weight
        loss = torch.mean(weight*dis_speech_power + weight_noise*res_noise_power + weight_noise*res_clean_power)
        return loss



class loadDatasetToMamory(data.Dataset):
    def __init__(self, npyname, npy_path, time_stems):
        super(loadDatasetToMamory, self).__init__()
        self.npyname = npyname
        self.npy_path = npy_path
    def __getitem__(self, index):
        npyname = self.npyname[index]
        load_ut = np.load(os.path.join(self.npy_path, npyname))
        row, list = np.shape(load_ut)
        clean = load_ut[0:row, 0: 257]                  #clean speech
        noisy = load_ut[0:row, 257: 257*2]               #noisy
        noise = load_ut[0:row, 257*2: 257 * 3]           #noise
        returns = [torch.FloatTensor(clean),torch.FloatTensor(noisy), torch.FloatTensor(noise)]

        return returns

    def __len__(self):
        return len(self.npyname)


def calculate_scalar(x):
    if x.ndim == 2:
        mean_ = np.mean(x, axis=0)
        std_ = np.std(x, axis=0)
    elif x.ndim == 3:
        mean_ = np.mean(x, axis=(0, 1))
        std_ = np.std(x, axis=(0, 1))
    return mean_, std_


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)
