"""
    bc_contrast2: use standar contrastive loss to cluster
"""
from models.bc_contrast2_cdc_adaptive.trainer import Trainer
from models.bc_contrast2_cdc_adaptive.custom_config import _C as custom_cfg



__all__ = [
    'Trainer',
    'custom_cfg'
]

