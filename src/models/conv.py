import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def conv3x3_dilated(in_channels, out_channels, dilation = 1):
    """dilated 3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation = dilation)

def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def conv64x1(in_channels, out_channels):
    """64x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(64,1))

def conv1x64(in_channels, out_channels):
    """1x64 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1,64))


class Block(nn.Module):
    def __init__(self, in_channels, dilation=1, dropout=0.15):
        super(Block, self).__init__()
        self.norm1    = nn.BatchNorm2d(in_channels)
        self.project_down = conv1x1(in_channels, in_channels//2)
        self.norm2    = nn.BatchNorm2d(in_channels//2)
        self.dilation = conv3x3_dilated(in_channels//2, in_channels//2, dilation=dilation)
        self.norm3    = nn.BatchNorm2d(in_channels//2)
        self.project_up = conv1x1(in_channels//2, in_channels)
        self.elu      = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.elu(out)
        out = self.project_down(out)
        out = self.norm2(out)
        out = self.elu(out)   
        out = self.dilation(out) 
        out = self.dropout(out)
        out = self.norm3(out)
        out = self.elu(out)
        out = self.project_up(out)
        return out + identity