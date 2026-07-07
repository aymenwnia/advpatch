"""Wrapper de détecteur exposant un forward DIFFÉRENTIABLE.

Sépare le reste du code du détecteur concret. En expérimentation on cible
YOLOv8 (Ultralytics) ; en scénario opérationnel c'est le modèle client (ou un
surrogate), potentiellement boîte-noire -> seul le forward change ici.

>>> SEAM PRINCIPAL À ADAPTER : extract_conf() ci-dessous. <<<
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DetectorWrapper(nn.Module):
    def __init__(self, weights: str = "yolov8n.pt", target_class: int = 2,
                 device: str = "cuda"):
        super().__init__()
        self.target_class = target_class
        self.device = device
        self.model = self._load(weights)
        # Détecteur gelé : on optimise le patch, pas les poids.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _load(self, weights: str) -> nn.Module:
        from ultralytics import YOLO
        yolo = YOLO(weights)
        return yolo.model.to(self.device)

    def raw_forward(self, images: torch.Tensor) -> torch.Tensor:
        """images : (B,3,H,W) en [0,1]. Renvoie conf classe-cible (B, N), diff."""
        preds = self.model(images)          # tenseur brut (ou tuple selon version)
        return self.extract_conf(preds)

    # ------------------------------------------------------------------------
    def extract_conf(self, preds) -> torch.Tensor:
        """Extrait la confiance de la classe cible, différentiable.

        YOLOv8 : la sortie brute est (B, 4+nc, N) -> [4 box, nc classes].
        TODO client : adapter le parsing (objectness séparée pour v7, format
        DETR différent, sigmoïde déjà appliquée ou non, etc.).
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        # (B, 4+nc, N) -> scores classes (B, nc, N)
        cls_scores = preds[:, 4:, :]
        conf = cls_scores[:, self.target_class, :]      # (B, N)
        # TODO: si la tête ne sigmoïde pas en interne, décommenter :
        # conf = conf.sigmoid()
        return conf

    @torch.no_grad()
    def detect_count(self, images: torch.Tensor, conf_thresh: float) -> torch.Tensor:
        """Nb de détections classe-cible au-dessus du seuil, par image (B,)."""
        conf = self.raw_forward(images)
        return (conf > conf_thresh).sum(dim=1)
