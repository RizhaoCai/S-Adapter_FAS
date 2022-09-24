import math
import pdb
import torch
from torch import nn
import numpy as np

from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.models as models
from torch.nn import functional as F
import pdb

def get_model_from_torchvision(arch_name, imagetnet_pretrain):
    """
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    googlenet = models.googlenet(pretrained=True)
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    mnasnet = models.mnasnet1_0(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    """
    net = models.__dict__[arch_name](pretrained=imagetnet_pretrain)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    return net


def get_ghost_net():
    from models.bc.ghost_net import GhostNet, ghostnet
    return ghostnet(num_classes=2)

def build_net(config):
    """

    :param config:
    :return:
    """
    imagetnet_pretrain = config.MODEL.IMAGENET_PRETRAIN
    model_arch = config.MODEL.ARCH
    if model_arch.lower() in models.__dict__.keys():
        return get_model_from_torchvision(model_arch, imagetnet_pretrain)

    elif model_arch.lower() == 'ghost_net':
        return get_ghost_net()

    else:
        exit('No valid arch name provided!')


if __name__ == '__main__':
    pass

