#_*conding:utf-8_*_
''' 
__author__ = YuyongKang, NenghengZheng
 __date__ = '2020/11/7' 
 __filename__ = 'define_loss.py'
 __IDE__ = 'PyCharm'
 __copyright__ = Shenzhen University ,Electronic and infromation college,
 Intelligent speech and artificial hearing Lab
 '''

import torch
import torch.nn as nn
import torch.nn.functional as func

class Adaptive_weight_loss(nn.Module):
    def __init__(self):
        super(Adaptive_weight_loss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, weight, obj_speech, ref_specch):
        power = torch.pow(obj_speech-ref_specch, 2)
        power = power.mul(weight)
        dis_speech_power = torch.mean(torch.mean(power, dim=-1), dim=-1)

        return dis_speech_power