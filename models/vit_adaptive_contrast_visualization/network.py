
from .timm_vit import vit_base_patch16_224
from .convpass import set_Convpass
from .timm_adapter import _create_vit_adapter




def build_net(arch_name, pretrained, **kwargs):
    num_classes = kwargs['num_classes']
    adaptive_type = kwargs['adaptive_type']
    if kwargs['conv_type'] == 'adapter':
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
        model = _create_vit_adapter('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
        return model

    if arch_name == 'vit_base_patch16_224':
        model = vit_base_patch16_224(pretrained, num_classes=num_classes) # todo



    if 'cdc' in kwargs['conv_type']:
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type=kwargs['conv_type'], cdc_theta=kwargs['cdc_theta'], adaptive_type=adaptive_type)
    else:
        set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'])
    #import pdb; pdb.set_trace()
    # import pdb;pdb.set_trace()
    return model

if __name__ == '__main__':
    import  torch
    model = build_net('vit_base_patch16_224', pretrained=True, conv_type='cdc_matrix')
    x=torch.rand(2,3,224,224)
    y=model(x)
    import IPython; IPython.embed()

