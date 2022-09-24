import timm


def build_net(arch_name, pretrained=True):
    """

    :param config:
    :return:
    """

    if 'vit' in arch_name:
        # model = VisionTransformer(
        #    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        #    norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=True)  # , **kwargs)
        # model.default_cfg = _cfg()

        """
        Options:
            'vit_base_patch16_224', 
        """
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=2).cuda()

        return model


if __name__ == '__main__':
    pass
