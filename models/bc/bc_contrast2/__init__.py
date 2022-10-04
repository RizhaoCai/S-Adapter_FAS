"""
    bc_contrast: binary classification and contrastive learning and probabily ewc
"""
from models.bc_contrast.trainer import Trainer
from models.bc_contrast.custom_config import _C as custom_cfg



__all__ = [
    'Trainer',
    'custom_cfg'
]

