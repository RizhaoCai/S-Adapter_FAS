"""
    vit_convpass_ds: dual stream
"""
from models.vit_convpass_ds.trainer import Trainer
from models.vit_convpass_ds.custom_config import _C as custom_cfg
import torch


__all__ = [
    'Trainer',
    'custom_cfg'
]

