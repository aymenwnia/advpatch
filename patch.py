"""Le patch adversarial, paramétré par sigmoïde pour rester dans [0, 1]."""
from __future__ import annotations
import torch
import torch.nn as nn


class AdversarialPatch(nn.Module):
    """Patch RGB optimisable.

    On optimise des logits libres et on applique une sigmoïde : la contrainte
    [0, 1] est ainsi respectée en permanence sans clamp destructif du gradient.
    Le patch est carré, dimensionné à la résolution réseau effective (cf.
    Config.patch_px_net) plutôt qu'à une résolution arbitraire.
    """

    def __init__(self, size_px: int, init: str = "gray", seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if init == "random":
            logits = torch.randn(3, size_px, size_px, generator=g)
        else:  # gris neutre : sigmoid(0) = 0.5
            logits = torch.zeros(3, size_px, size_px)
        self.logits = nn.Parameter(logits)

    def forward(self) -> torch.Tensor:
        """Retourne le patch en [0, 1], shape (3, H, W)."""
        return torch.sigmoid(self.logits)

    @torch.no_grad()
    def image(self) -> torch.Tensor:
        """Copie détachée pour visualisation / sauvegarde."""
        return self.forward().detach().cpu()
