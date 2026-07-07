"""Évaluation : ASR (Attack Success Rate) + balayage de points de vue.

ASR = fraction des instances cibles détectées SANS patch qui deviennent
non-détectées AVEC patch. On ne compte que celles vues au départ, pour ne pas
gonfler le score avec des objets déjà manqués par le détecteur.
"""
from __future__ import annotations
import torch

from .config import Config
from .placement import warp_and_composite
from . import transforms as T


@torch.no_grad()
def compute_asr(cfg: Config, detector, patch: torch.Tensor, loader) -> float:
    seen = 0
    suppressed = 0
    p = patch.to(cfg.device)
    for batch in loader:
        images = batch["image"].to(cfg.device)
        quads = batch["quad"].to(cfg.device)

        base = detector.detect_count(images, cfg.conf_thresh)      # (B,)
        scene = warp_and_composite(images, p, quads)
        after = detector.detect_count(scene, cfg.conf_thresh)      # (B,)

        was_seen = base > 0
        now_gone = after == 0
        seen += was_seen.sum().item()
        suppressed += (was_seen & now_gone).sum().item()
    return suppressed / max(seen, 1)


@torch.no_grad()
def viewpoint_sweep(cfg: Config, detector, patch: torch.Tensor, loader,
                    scales=(0.5, 0.75, 1.0, 1.25),
                    rots=(-20, -10, 0, 10, 20)) -> dict:
    """ASR par (échelle, rotation) : donne la baseline par point de vue à reporter.

    Fait varier le quad de placement de façon déterministe (pas d'EOT aléatoire)
    pour isoler la sensibilité au point de vue.
    """
    p = patch.to(cfg.device)
    results = {}
    for s in scales:
        for r in rots:
            seen = supp = 0
            for batch in loader:
                images = batch["image"].to(cfg.device)
                quads = _scale_rotate(batch["quad"].to(cfg.device), s, r)
                base = detector.detect_count(images, cfg.conf_thresh)
                scene = warp_and_composite(images, p, quads)
                after = detector.detect_count(scene, cfg.conf_thresh)
                was = base > 0
                seen += was.sum().item()
                supp += (was & (after == 0)).sum().item()
            results[(s, r)] = supp / max(seen, 1)
    return results


def _scale_rotate(quads: torch.Tensor, scale: float, rot_deg: float) -> torch.Tensor:
    """Applique échelle + rotation aux quads autour de leur centre."""
    dev = quads.device
    theta = torch.tensor(rot_deg * 3.14159265 / 180.0, device=dev)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]], device=dev)
    ctr = quads.mean(dim=1, keepdim=True)
    return ctr + (quads - ctr) * scale @ R.T
