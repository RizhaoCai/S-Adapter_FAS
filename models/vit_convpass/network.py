from .timm_vit import _create_vit_adapter
import logging
import timm
from timm.models.vision_transformer import vit_base_patch16_224
from .convpass import set_Convpass




def build_net(arch_name, pretrained, **kwargs):
    if arch_name == 'vit_base_patch16_224':
        model = vit_base_patch16_224(pretrained)

    if kwargs['conv_type'] == 'cdc':
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type=kwargs['conv_type'])
    else:
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'])
    #import pdb; pdb.set_trace()

    return model

if __name__ == '__main__':
    build_net('vit_base_patch16_224')

