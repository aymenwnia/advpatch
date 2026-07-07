"""Fonctions de perte : suppression de détection (Thys) + TV + NPS.

L = w_det * L_det + w_tv * L_tv + w_nps * L_nps
"""
from __future__ import annotations
import torch


def detection_loss(conf: torch.Tensor) -> torch.Tensor:
    """Perte de suppression façon Thys et al.

    conf : (B, N) confiances de la classe cible sur les N ancres/détections.
    On minimise la confiance MAX par image : si le meilleur candidat tombe
    sous le seuil, l'objet disparaît. Le max (plutôt que la moyenne) concentre
    la pression sur la détection la plus difficile à supprimer.
    """
    if conf.numel() == 0:
        return conf.new_zeros(())
    per_image_max = conf.amax(dim=1)          # (B,)
    return per_image_max.mean()


def tv_loss(patch: torch.Tensor) -> torch.Tensor:
    """Total variation : favorise des transitions de couleur lisses (imprimables).

    patch : (3, H, W) en [0,1].
    """
    dh = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    dw = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return dh + dw


def nps_loss(patch: torch.Tensor, printable: torch.Tensor) -> torch.Tensor:
    """Non-Printability Score : distance à la palette imprimable la plus proche.

    patch     : (3, H, W) en [0,1].
    printable : (K, 3) couleurs reproductibles (imprimante, ou palette camouflage).

    NOTE modalité : n'a de sens qu'en RGB. Si le capteur cible est IR/multispectral,
    remplacer par une contrainte d'émissivité/bande adaptée (cf. spec Kust4K).
    """
    if printable is None or printable.numel() == 0:
        return patch.new_zeros(())
    px = patch.permute(1, 2, 0).reshape(-1, 3)            # (H*W, 3)
    d = torch.cdist(px, printable.to(px.device))          # (H*W, K)
    return d.min(dim=1).values.mean()


def total_loss(conf, patch, printable, w_det, w_tv, w_nps):
    l_det = detection_loss(conf)
    l_tv = tv_loss(patch)
    l_nps = nps_loss(patch, printable)
    total = w_det * l_det + w_tv * l_tv + w_nps * l_nps
    return total, {"det": l_det.item(), "tv": l_tv.item(), "nps": float(l_nps)}
