#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : ResidualAttention.py
# @Author : yfor
# @Mail   : xxxx@mail.com
# @Date   : 2021/08/14 17:02:17
# @Docs   : 
'''

import torch
from torch import nn

class ResidualAttention(nn.Module):
    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y_raw = self.fc(x).flatten(2) # b, num_class, h*w
        y_avg = torch.mean(y_raw, dim=2) # b, num_class
        y_max = torch.max(y_raw, dim=2)[0] # b, num_class
        score = y_avg + self.la * y_max
        return score

if __name__ == '__main__':

    channel = 4
    num_class = 3
    batchsize = 1
    input = torch.randn(batchsize, channel, 3, 3)
    resatt = ResidualAttention(channel=channel, num_class=num_class, la=0.2)
    output = resatt(input)
    print(output.shape)