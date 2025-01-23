"""
   vit_s_adapter
    
"""
from models.vit_s_adapter.trainer import Trainer
from models.vit_s_adapter.custom_config import _C as custom_cfg
import torch


__all__ = [
    'Trainer',
    'custom_cfg'
]

