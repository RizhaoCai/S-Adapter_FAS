"""
    adaptive_elastic weight
"""
from models.bc_ewc.trainer import Trainer
from models.bc_ewc.custom_config import _C as custom_cfg
import torch


__all__ = [
    'Trainer',
    'custom_cfg'
]

