# from .timm_vit import _create_vit_adapter
# import logging
# import timm
from timm.models.vision_transformer import vit_base_patch16_224, vit_large_patch16_224, vit_tiny_patch16_224, vit_small_patch16_224
from .convpass import set_Convpass





def build_net(arch_name, pretrained, **kwargs):
    num_classes = kwargs['num_classes']
    # import pdb;pdb.set_trace()
    if arch_name == 'vit_base_patch16_224':
        reduction_dim = 768
        model = vit_base_patch16_224(pretrained, num_classes=num_classes) # todo

    if arch_name == 'vit_large_patch16_224':
        reduction_dim = 1024
        model = vit_large_patch16_224(pretrained, num_classes=num_classes) # todo

    if arch_name == 'vit_small_patch16_224':
        reduction_dim = 384
        model = vit_small_patch16_224(pretrained, num_classes=num_classes) # todo

    if arch_name == 'vit_tiny_patch16_224':
        reduction_dim = 192
        model = vit_tiny_patch16_224(pretrained, num_classes=num_classes) # todo

    if 'vlad' in kwargs['conv_type']:
        set_Convpass(model, 'vladpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'],
                     cdc_theta=kwargs['cdc_theta'])

    elif 'histpass' in kwargs['conv_type']:
        set_Convpass(model, 'histpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'],
                     cdc_theta=kwargs['cdc_theta'])

    elif 'cdc' in kwargs['conv_type']:
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type=kwargs['conv_type'], cdc_theta=kwargs['cdc_theta'], num_bins=kwargs['num_bins'], reduction_dim=reduction_dim)
    else:
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'], num_bins=kwargs['num_bins'], reduction_dim=reduction_dim)




    #import pdb; pdb.set_trace()

    return model



if __name__ == '__main__':
    import  torch
    from thop import profile
    from functools import reduce

    model_vit = vit_base_patch16_224(True, num_classes=2)
    model = build_net('vit_base_patch16_224', pretrained=True, conv_type='cdc_hist',num_classes=2, cdc_theta=0.7, num_bins=1)
    x=torch.rand(2,3,224,224)
    y=model(x)

    macs, params = profile(model, inputs=(x,))

    n = 0
    for name, p in model.named_parameters():
        #import pdb; pdb.set_trace()
        shape = p.data.shape
        n+=reduce(lambda x, y: x * y, shape)
    import IPython;

    IPython.embed()



