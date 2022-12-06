"""
    bc contain many networks with different architecture. But all these networks are for binary classification and use cross encropy as loss for optimization
"""
from models.vit_convpass_sample.trainer import Trainer
from models.vit_convpass_sample.custom_config import _C as custom_cfg
import torch


__all__ = [
    'Trainer',
    'custom_cfg'
]

