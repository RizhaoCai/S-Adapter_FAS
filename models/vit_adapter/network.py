from .timm_vit import _create_vit_adapter
import logging

def vit_base_patch16_224(pretrained=True, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """

    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vit_adapter('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    #   import pdb; pdb.set_trace()
    return model


def build_net(arch_name, pretrained, num_classes):
    if arch_name == 'vit_base_patch16_224':
        return vit_base_patch16_224(pretrained, num_classes=num_classes)

if __name__ == '__main__':
    build_net('vit_base_patch16_224')

