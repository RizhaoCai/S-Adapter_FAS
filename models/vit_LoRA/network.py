# from .timm_vit import _create_vit_adapter
# import logging
# import timm
#from timm.models.vision_transformer import vit_base_patch16_224
#from .convpass import set_Convpass
from .timm_lora import vit_base_patch16_224



def build_net(arch_name, pretrained, **kwargs):
    num_classes = kwargs['num_classes']

    model_cfg = {
        'super_LoRA_dim': kwargs['super_LoRA_dim'],
        'super_prompt_tuning_dim': kwargs['super_prompt_tuning_dim'],
        'super_adapter_dim': kwargs['super_adapter_dim'],
        'super_prefix_dim': kwargs['super_prefix_dim'],
    }
    if arch_name == 'vit_base_patch16_224':
        model = vit_base_patch16_224(pretrained, **kwargs) # todo

    #if 'cdc' in kwargs['conv_type']:
    #    set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=False, conv_type=kwargs['conv_type'], cdc_theta=kwargs['cdc_theta'])
    #else:
    #    set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True, conv_type=kwargs['conv_type'])
    #import pdb; pdb.set_trace()

    return model

if __name__ == '__main__':
    import  torch
    model = build_net('vit_base_patch16_224', pretrained=True, conv_type='cdc_matrix')
    x=torch.rand(2,3,224,224)
    y=model(x)
    import IPython; IPython.embed()

