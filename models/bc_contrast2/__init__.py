"""
    bc_contrast2: use standar contrastive loss to cluster
"""
from models.bc_contrast.trainer import Trainer
from models.bc_contrast.custom_config import _C as custom_cfg



__all__ = [
    'Trainer',
    'custom_cfg'
]

