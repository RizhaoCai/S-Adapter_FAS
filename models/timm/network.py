

import torchvision.models as models
from torch.nn import functional as F
import pdb
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models import create_model
from functools import partial

import torch.nn as nn
import torch.optim as optim



def build_net(arch_name, pretrained=True):
    """

    :param config:
    :return:
    """


    if 'vit' in arch_name:
        #model = VisionTransformer(
        #    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        #    norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=True)  # , **kwargs)
        #model.default_cfg = _cfg()

        """
        Options:
            'vit_base_patch16_224', 
        """
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=2).cuda()


        return model




if __name__ == '__main__':
    pass

