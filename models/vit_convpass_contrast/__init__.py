"""
    bc contain many networks with different architecture. But all these networks are for binary classification and use cross encropy as loss for optimization
"""
from models.vit_convpass_contrast.trainer import Trainer
from models.vit_convpass_contrast.custom_config import _C as custom_cfg


__all__ = [
    'Trainer',
    'custom_cfg'
]

