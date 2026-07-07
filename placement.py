"""Placement différentiable du patch par homographie + alpha-compositing.

Hypothèse aérienne clé : le patch est plan et posé sur le plan-sol, donc sa
projection vers n'importe quelle caméra aérienne est une homographie. Pas
besoin de NeRF ni de rendu différentiable — un warp_perspective suffit.

Dépendance : kornia (warp différentiable). Fallback torch.nn.functional si absent.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    import kornia
    _HAS_KORNIA = True
except Exception:  # pragma: no cover
    _HAS_KORNIA = False


def _unit_corners(size_px: int, B: int, device) -> torch.Tensor:
    """Coins source du patch (repère pixel du patch)."""
    c = torch.tensor(
        [[0, 0], [size_px - 1, 0], [size_px - 1, size_px - 1], [0, size_px - 1]],
        dtype=torch.float32, device=device,
    )
    return c.unsqueeze(0).repeat(B, 1, 1)


def warp_and_composite(
    scene: torch.Tensor,        # (B, 3, H, W) en [0,1]
    patch: torch.Tensor,        # (3, ph, pw) en [0,1]
    dst_quads: torch.Tensor,    # (B, 4, 2) coins destination dans la scène
) -> torch.Tensor:
    """Projette le patch sur chaque scène via l'homographie source->dst, puis composite.

    Renvoie la scène modifiée (B, 3, H, W), différentiable p/r au patch.
    """
    if not _HAS_KORNIA:
        raise ImportError("kornia requis pour le warp différentiable : pip install kornia")

    B, _, H, W = scene.shape
    ph, pw = patch.shape[-2:]
    dev = scene.device

    patch_b = patch.unsqueeze(0).repeat(B, 1, 1, 1)                  # (B,3,ph,pw)
    alpha = torch.ones(B, 1, ph, pw, device=dev)                    # masque plein

    src = _unit_corners(ph, B, dev)                                 # (B,4,2)
    Hm = kornia.geometry.get_perspective_transform(src, dst_quads)  # (B,3,3)

    warped_patch = kornia.geometry.warp_perspective(patch_b, Hm, (H, W))
    warped_alpha = kornia.geometry.warp_perspective(alpha, Hm, (H, W))

    # alpha-compositing : scene * (1 - a) + patch * a
    return scene * (1.0 - warped_alpha) + warped_patch * warped_alpha
