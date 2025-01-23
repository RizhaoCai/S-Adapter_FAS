# -*- coding: utf-8 -*-
"""
Created on Thursday Sep 3 9:25:26 2020
Generate piecewise linear histogram layer
@author: jpeeples

https://github.com/GatorSense/Histogram_Layer/blob/f77b71b6be7cc46b1f65193e1bf6c2f70fb1d904/Utils/RBFHistogramPooling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class HistogramLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dim=2, num_bins=4,
                 stride=1, padding=0, normalize_count=True, normalize_bins=True,
                 count_include_pad=False,
                 ceil_mode=False, groups=196):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=True)
            self.bin_widths_conv.bias.data.fill_(1)
            self.bin_widths_conv.bias.requires_grad = False
            self.hist_pool = nn.AvgPool1d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Image Data
        elif self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=True)
            self.bin_widths_conv.bias.data.fill_(1)
            self.bin_widths_conv.bias.requires_grad = False
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=True)
            self.bin_widths_conv.bias.data.fill_(1)
            self.bin_widths_conv.bias.requires_grad = False
            self.hist_pool = nn.AvgPool3d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window
        # import pdb; pdb.set_trace()
        # Pass through first convolution to learn bin centers: |x-center|
        xx = torch.abs(self.bin_centers_conv(xx))

        # Pass through second convolution to learn bin widths 1-w*|x-center|
        xx = self.bin_widths_conv(xx)

        # Pass through relu
        xx = F.relu(xx)

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n, c, l = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Image Data
        elif self.dim == 2:
            n, c, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n, c, d, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

        return xx



class RFBHistogramLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dim=2, num_bins=4,
                 stride=1, padding=0, normalize_count=True, normalize_bins=True,
                 count_include_pad=False,
                 ceil_mode=False):

        # inherit nn.module
        super(RFBHistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool1d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Image Data
        elif self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels, self.numBins * self.in_channels, 1,
                                              groups=self.in_channels, bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins * self.in_channels,
                                             self.numBins * self.in_channels, 1,
                                             groups=self.numBins * self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool3d(self.filt_dim, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)

        # Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)

        # Pass through radial basis function
        xx = torch.exp(-(xx ** 2))

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n, c, l = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Image Data
        elif self.dim == 2:
            n, c, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

            # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n, c, d, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        else:
            raise RuntimeError('Invalid dimension for histogram layer')

        return xx

class ConvpassHist(nn.Module):
    def __init__(self, dim=8, xavier_init=True, conv_type='conv', cdc_theta=0.7):
        super().__init__()

        self.conv_type = conv_type
        self.adapter_hist = HistogramLayer(3*196, kernel_size=3, dim=2, padding=1,num_bins=4, groups=196)

        if 'conv' in conv_type:
            self.adapter_conv = nn.Conv2d(3*196, 3*196, 3, 1, 1, groups=196)
            self.adapter_conv_down = nn.Conv2d(12*196, 3*196, 1, 1, 0, groups=196)

            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.weight)
                self.adapter_conv.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv.bias)

            #self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
            #self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            #nn.init.xavier_uniform_(self.adapter_down.weight)
            #nn.init.zeros_(self.adapter_down.bias)
            #nn.init.zeros_(self.adapter_up.weight)
            #nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward3(self, x):

        B, N, C = x.shape
        patch_size = 16
        x_cls = x[:, 0, :]
        x_patch = x[:, 1:, :]
        x_patch = x_patch.reshape(B,N-1, 3, patch_size,patch_size) # N 197
        # x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv # 197,down # B, 196, 8

        x_hist = []
        for i in range(196):
            patch_token_i = self.adapter_conv(x_patch[:, i, :,:,:]) # B, 3, N, N


            patch_token_i = self.act(patch_token_i)
            patch_token_i = self.adapter_hist(patch_token_i) # B, 768 #  # B, 3, N, N

            patch_token_i = self.act(patch_token_i)
            patch_token_i = self.adapter_conv_down(patch_token_i).reshape(B,768)
            x_hist.append(patch_token_i)

        x_out = torch.stack([x_cls]+ x_hist, dim=1)

        return x_out

    def forward(self, x):

        B, N, C = x.shape
        patch_size = 16
        x_cls = x[:, 0:1, :]
        x_patch = x[:, 1:, :]
        # import pdb;
        # pdb.set_trace()
        x_patch = x_patch.reshape(B,(N-1)*3, patch_size,patch_size) # N 197
        # x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv # 197,down # B, 196, 8


        patch_token_i = self.adapter_conv(x_patch) # B, 3, N, N


        patch_token_i = self.act(patch_token_i)
        patch_token_i = self.adapter_hist(patch_token_i) # B, 768 #  # B, 3, N, N

        patch_token_i = self.act(patch_token_i)
        patch_token_i = self.adapter_conv_down(patch_token_i).reshape(B,196, 768)

        x_out = torch.cat([x_cls, patch_token_i], dim=1)

        return x_out

# x = torch.zeros(2, 197, 768)
# y = torch.zeros(2,3,16,16)
# hist = HistogramLayer(in_channels=3, kernel_size=3, dim=2, padding=1,num_bins=4)
# import IPython; IPython.embed()
# c=ConvpassHist()
#
# import IPython; IPython.embed()