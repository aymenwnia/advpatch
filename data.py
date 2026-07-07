"""Chargement des données : scène + quad de placement du patch.

Chaque échantillon = (image à la taille réseau, quad des 4 coins où poser le
patch sur le plan-sol / sur la cible). Le quad vient des annotations (bbox du
véhicule) projetées sur le plan-sol.

>>> STUB : brancher ici VisDrone / xView. Un dataset synthétique permet de
faire tourner le squelette de bout en bout sans données réelles. <<<
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader


def quad_from_box(box_xyxy: torch.Tensor, shrink: float = 0.5) -> torch.Tensor:
    """Construit un quad de placement centré dans la bbox cible.

    box_xyxy : (4,) = x1,y1,x2,y2. shrink : fraction de la bbox couverte par le patch.
    Ici quad = rectangle simple ; en oblique réel il serait non-rectangulaire
    (l'homographie s'en charge), mais l'EOT perspective_jitter injecte déjà
    cette variation à l'entraînement.
    """
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) * shrink / 2
    hh = (y2 - y1) * shrink / 2
    return torch.tensor([[cx - hw, cy - hh], [cx + hw, cy - hh],
                         [cx + hw, cy + hh], [cx - hw, cy + hh]])


class SyntheticPatchDataset(Dataset):
    """Fond bruité + une bbox centrale — pour valider la plomberie uniquement."""

    def __init__(self, n: int, net_size: int, patch_px: int, seed: int = 0):
        self.n, self.net_size, self.patch_px = n, net_size, patch_px
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.rand(3, self.net_size, self.net_size, generator=self.g) * 0.3 + 0.35
        c = self.net_size / 2
        r = self.patch_px * 1.6
        box = torch.tensor([c - r, c - r, c + r, c + r])
        return {"image": img, "quad": quad_from_box(box)}


class RealPatchDataset(Dataset):
    """TODO : VisDrone/xView. Charger image, resample au GSD opérationnel,
    letterbox à net_size, lire la bbox de la classe cible -> quad."""

    def __init__(self, root: str, cfg):
        raise NotImplementedError("Brancher le chargement VisDrone/xView ici.")


def make_loader(cfg, real: bool = False) -> DataLoader:
    if real:
        ds = RealPatchDataset(root="/data/...", cfg=cfg)
    else:
        ds = SyntheticPatchDataset(256, cfg.net_size, cfg.patch_px_net(), cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
