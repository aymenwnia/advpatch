"""Expectation Over Transformation (EOT).

Trois familles de perturbations, toutes différentiables :
  1. géométrique  : jitter de perspective/rotation/échelle autour du placement
  2. photométrique: luminosité / contraste / bruit
  3. résolution   : downsample->upsample pour émuler la perte capture au GSD

Les plages viennent de Config.EOTConfig et DOIVENT être calibrées sur
l'enveloppe réelle du capteur (angle de visée nadir<->oblique, altitude).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .config import EOTConfig


def _rand(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def perspective_jitter_quads(base_quad: torch.Tensor, mag: float) -> torch.Tensor:
    """Perturbe aléatoirement les 4 coins du quad de placement.

    base_quad : (B, 4, 2) coins destination dans l'image.
    Renvoie des quads bruités, modélisant la variation de point de vue.
    """
    if mag <= 0:
        return base_quad
    B = base_quad.shape[0]
    span = base_quad.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
    noise = (torch.rand_like(base_quad) - 0.5) * 2 * mag * span
    return base_quad + noise


def photometric_jitter(x: torch.Tensor, cfg: EOTConfig) -> torch.Tensor:
    """Luminosité additive, contraste multiplicatif, bruit gaussien. x en [0,1]."""
    B = x.shape[0]
    dev = x.device
    b = _rand(*cfg.brightness, B, dev).view(B, 1, 1, 1)
    c = _rand(*cfg.contrast, B, dev).view(B, 1, 1, 1)
    x = (x - 0.5) * c + 0.5 + b
    if cfg.noise_std > 0:
        x = x + torch.randn_like(x) * cfg.noise_std
    return x.clamp(0.0, 1.0)


def resolution_degrade(x: torch.Tensor, cfg: EOTConfig) -> torch.Tensor:
    """Downsample puis upsample bilinéaire -> émule la limite de résolution GSD.

    Empêche le patch d'apprendre un détail haute-fréquence irréaliste qui ne
    survivrait pas à la chaîne capture->redimensionnement réseau.
    """
    lo, hi = cfg.res_degrade
    f = float(_rand(lo, hi, 1, x.device).item())
    if f >= 0.999:
        return x
    _, _, H, W = x.shape
    h, w = max(1, int(H * f)), max(1, int(W * f))
    down = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
