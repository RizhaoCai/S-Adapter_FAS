"""
    bc_contrast2: use standar contrastive loss to cluster
"""
from models.bc_contrast2.trainer import Trainer
from models.bc_contrast2.custom_config import _C as custom_cfg



__all__ = [
    'Trainer',
    'custom_cfg'
]

