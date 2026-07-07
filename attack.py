"""Boucle d'entraînement du patch : assemble tous les modules."""
from __future__ import annotations
import torch

from .config import Config
from .patch import AdversarialPatch
from .model import DetectorWrapper
from .placement import warp_and_composite
from . import transforms as T
from .losses import total_loss


class PatchAttack:
    def __init__(self, cfg: Config, detector: DetectorWrapper):
        self.cfg = cfg
        self.det = detector
        torch.manual_seed(cfg.seed)
        self.patch = AdversarialPatch(cfg.patch_px_net(), init="random",
                                      seed=cfg.seed).to(cfg.device)
        self.opt = torch.optim.Adam(self.patch.parameters(), lr=cfg.lr)
        pc = cfg.printable_colors
        self.printable = (torch.tensor(pc, dtype=torch.float32)
                          if pc else torch.empty(0, 3)).to(cfg.device)

    def _apply(self, images: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
        """EOT complet : patch -> jitter géométrique/photométrique/résolution -> composite."""
        cfg, eot = self.cfg, self.cfg.eot
        p = self.patch()                                      # (3,ph,pw) [0,1]
        p = T.resolution_degrade(p.unsqueeze(0), eot).squeeze(0)
        quads = T.perspective_jitter_quads(quads, eot.perspective)
        scene = warp_and_composite(images, p, quads)          # (B,3,H,W)
        scene = T.photometric_jitter(scene, eot)
        return scene

    def train(self, loader, log_every: int = 50):
        cfg = self.cfg
        it = 0
        while it < cfg.iters:
            for batch in loader:
                if it >= cfg.iters:
                    break
                images = batch["image"].to(cfg.device)
                quads = batch["quad"].to(cfg.device)

                scene = self._apply(images, quads)
                conf = self.det.raw_forward(scene)            # (B, N) diff
                loss, parts = total_loss(conf, self.patch(), self.printable,
                                         cfg.w_det, cfg.w_tv, cfg.w_nps)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if it % log_every == 0:
                    print(f"[{it:5d}] loss={loss.item():.4f} "
                          f"det={parts['det']:.4f} tv={parts['tv']:.4f} "
                          f"nps={parts['nps']:.4f}")
                it += 1
        return self.patch
