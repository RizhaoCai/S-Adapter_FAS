import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from visualizer import get_local

import timm
import math
from torch.nn import functional as F
from .cdc_matrix import Conv2d_cd_pixel_difference_matrix5x5_unshared, Conv2d_cd_pixel_difference_matrix5x5_shared, Conv2d_cd_pixel_difference_matrix4x4_unshared
import logging
from .hist import HistogramLayer
get_local.activate()

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class DC_Conv(nn.Module):
    def __init__(self, conv1, conv2):
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x):
        return (self.conv1(x) + self.conv2(x))/2




class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:

            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_cd_hist(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7, num_bins=1):

        super(Conv2d_cd_hist, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.hist = HistogramLayer(in_channels, kernel_size=3, dim=2, padding=1,num_bins=num_bins)

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:

            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            out = out_normal - self.theta * out_diff
            out = self.hist(out)
            return out

class Conv2d_cd_hist_cat_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd_hist_cat_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.hist = HistogramLayer(in_channels, kernel_size=3, dim=2, padding=1,num_bins=1) # ?

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            out = out_normal
            return out_normal
        else:

            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            out = out_normal - self.theta * out_diff
            hist_out = self.hist(out)

            out = torch.cat([out,hist_out],1)
            return out





# @get_local('adapter_attn')
# def forward_block(self, x):
#     adapter_attn = self.adapter_attn(self.norm1(x))
#     x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(adapter_attn) * self.s
#     x = x + self.drop_path2(self.mlp(self.norm2(x))) + self.drop_path2(self.adapter_mlp(self.norm2(x))) * self.s
#     return x

def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x

def forward_block_attn(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='conv', cdc_theta=0.7, num_bins=1, reduction_dim=768):
        super().__init__()

        self.conv_type = conv_type

        self.num_bins = num_bins
        if conv_type=='conv':
            self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)

            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.weight)
                self.adapter_conv.conv.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv.bias)

            #import pdb; pdb.set_trace()
            self.adapter_down = nn.Linear(reduction_dim, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, reduction_dim)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

        elif conv_type=='cdc':
            logging.info('cdc_theta={}'.format(cdc_theta))
            self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1, theta=cdc_theta)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            #nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(reduction_dim, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim, reduction_dim)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
        #     nn.init.zeros_(self.adapter_up.bias)

        elif conv_type == 'cdc_hist':
            logging.info('cdc_theta={}'.format(cdc_theta))
            logging.info('num_bin={}'.format(num_bins))
            # import pdb;
            # pdb.set_trace()
            self.adapter_conv = Conv2d_cd_hist(dim, dim, 3, 1, 1, theta=cdc_theta, num_bins=num_bins)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.conv.weight)
                self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
            # nn.init.zeros_(self.adapter_conv.conv.bias)

            self.adapter_down = nn.Linear(reduction_dim, dim)  # equivalent to 1 * 1 Conv
            self.adapter_up = nn.Linear(dim*num_bins, reduction_dim)  # equivalent to 1 * 1 Conv
            #self.adapter_up_cls = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)


        # elif conv_type == 'cdc_hist+conv':
        #     logging.info('cdc_theta={}'.format(cdc_theta))
        #     self.adapter_conv = None # Conv2d_cd_hist(dim, dim, 3, 1, 1, theta=cdc_theta) # todo
        #     if xavier_init:
        #         nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
        #     else:
        #         nn.init.zeros_(self.adapter_conv.conv.weight)
        #         self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        #     # nn.init.zeros_(self.adapter_conv.conv.bias)
        #
        #     self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        #
        #     self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        #
        #     self.adapter_up_cls = nn.Linear(dim*2, 768)  # equivalent to 1 * 1 Conv
        #     nn.init.xavier_uniform_(self.adapter_down.weight)
        #     nn.init.zeros_(self.adapter_down.bias)
        #     nn.init.zeros_(self.adapter_up.weight)
        #     nn.init.zeros_(self.adapter_up.bias)
        #
        # elif conv_type == 'cdc_hist_cat_conv':
        #     logging.info('cdc_theta={}'.format(cdc_theta))
        #     self.adapter_conv = Conv2d_cd_hist_cat_conv(dim, dim, 3, 1, 1, theta=cdc_theta)
        #     if xavier_init:
        #         nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
        #     else:
        #         nn.init.zeros_(self.adapter_conv.conv.weight)
        #         self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        #     # nn.init.zeros_(self.adapter_conv.conv.bias)
        #
        #     self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        #     self.adapter_up = nn.Linear(dim*2, 768)  # equivalent to 1 * 1 Conv
        #     self.adapter_up_cls = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        #     nn.init.xavier_uniform_(self.adapter_down.weight)
        #     nn.init.zeros_(self.adapter_down.bias)
        #     nn.init.zeros_(self.adapter_up.weight)
        #     nn.init.zeros_(self.adapter_up.bias)
        #
        #
        #
        # elif 'cdc_matrix' in conv_type:
        #     logging.info('cdc_theta={}'.format(cdc_theta))
        #     if conv_type == 'cdc_matrix_5x5unshared':
        #         self.adapter_conv = Conv2d_cd_pixel_difference_matrix5x5_unshared(dim, dim, 3, 1, 1, theta=cdc_theta)
        #     elif conv_type == 'cdc_matrix_5x5shared':
        #         self.adapter_conv = Conv2d_cd_pixel_difference_matrix5x5_shared(dim, dim, 3, 1, 1, theta=cdc_theta)
        #     elif conv_type == 'cdc_matrix_4x4unshared':
        #         self.adapter_conv = Conv2d_cd_pixel_difference_matrix4x4_unshared(dim, dim, 3, 1, 1, theta=cdc_theta)
        #
        #
        #     if xavier_init:
        #         nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
        #     else:
        #         nn.init.zeros_(self.adapter_conv.conv.weight)
        #         self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        #     # nn.init.zeros_(self.adapter_conv.conv.bias)
        #
        #     self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        #     self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        #     nn.init.xavier_uniform_(self.adapter_down.weight)
        #     nn.init.zeros_(self.adapter_down.bias)
        #     nn.init.zeros_(self.adapter_up.weight)
        #     nn.init.zeros_(self.adapter_up.bias)


        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def __call__(self, x):
        if self.conv_type == 'cdc_hist_cat_conv':
            return self.forward_hist_cat_conv(x)
        else:
            return self.forward(x)


    def forward_hist_cat_conv(self, x):
        B, N, C = x.shape


        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim*2)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)

        x_cls = self.adapter_conv.conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_patch = self.act(x_patch)
        x_patch = self.dropout(x_patch)
        x_patch_up = self.adapter_up(x_patch)

        x_cls_up = self.act(x_cls)
        x_cls_up = self.dropout(x_cls_up)
        x_cls_up = self.adapter_up_cls(x_cls_up)
        x_up = torch.cat([x_cls_up, x_patch_up], dim=1)

        # x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up



    def forward(self, x):
        B, N, C = x.shape
        # import pdb;
        #pdb.set_trace()
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim*self.num_bins)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)

        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim*self.num_bins)

        #x_cls = sel

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up




def set_Convpass(model, method, dim=8, s=1, xavier_init=False, conv_type='conv', cdc_theta=0.7, num_bins=1, reduction_dim=768):

    if method == 'convpass':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                # import pdb; pdb.set_trace()
                _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta, num_bins=num_bins, reduction_dim=reduction_dim)
                _.adapter_mlp = Convpass(dim, xavier_init,conv_type=conv_type, cdc_theta=cdc_theta, num_bins=num_bins,reduction_dim=reduction_dim)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta, num_bins=num_bins,reduction_dim=reduction_dim)
    # elif method == 'vladpass':
    #     for _ in model.children():
    #         if type(_) == timm.models.vision_transformer.Block:
    #             _.adapter_attn = PatchNetVLAD()
    #             _.adapter_mlp = PatchNetVLAD()
    #             _.s = s
    #             bound_method = forward_block.__get__(_, _.__class__)
    #             setattr(_, 'forward', bound_method)
    #         elif len(list(_.children())) != 0:
    #             set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta)
    # elif method == 'histpass':
    #     for _ in model.children():
    #         if type(_) == timm.models.vision_transformer.Block:
    #             _.adapter_attn = ConvpassHist()
    #             _.adapter_mlp = ConvpassHist()
    #             _.s = s
    #             bound_method = forward_block.__get__(_, _.__class__)
    #             setattr(_, 'forward', bound_method)
    #         elif len(list(_.children())) != 0:
    #             set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta)
    #
    # else:
    #     for _ in model.children():
    #         if type(_) == timm.models.vision_transformer.Block:
    #             _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type)
    #             _.s = s
    #             bound_method = forward_block_attn.__get__(_, _.__class__)
    #             setattr(_, 'forward', bound_method)
    #         elif len(list(_.children())) != 0:
    #             set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type)



#vlad = NetVLAD(num_cluster=6, dim=128)