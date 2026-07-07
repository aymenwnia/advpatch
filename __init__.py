"""Attaque par patch adversarial pour détection aérienne (squelette 2D)."""
from .config import Config, EOTConfig
from .patch import AdversarialPatch
from .model import DetectorWrapper
from .attack import PatchAttack
from . import evaluate, losses, transforms, placement, data

__all__ = [
    "Config", "EOTConfig", "AdversarialPatch", "DetectorWrapper",
    "PatchAttack", "evaluate", "losses", "transforms", "placement", "data",
]
