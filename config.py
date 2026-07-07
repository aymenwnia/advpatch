"""Configuration centrale de l'attaque par patch (cas aérien 2D).

Toute la spec discutée est ici : la taille du patch n'est PAS un nombre de
pixels arbitraire, elle se dérive de la taille physique + du GSD opérationnel,
puis du ratio de redimensionnement vers l'entrée du détecteur.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import yaml


@dataclass
class EOTConfig:
    """Enveloppe EOT. À CALIBRER sur l'enveloppe capteur/plateforme du client."""
    rot_deg: Tuple[float, float] = (-20.0, 20.0)      # rotation dans le plan
    scale: Tuple[float, float] = (0.25, 1.25)          # échelle relative
    perspective: float = 0.10                          # jitter perspective (0..1)
    brightness: Tuple[float, float] = (-0.1, 0.1)
    contrast: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02
    # Dégradation de résolution : émule la perte capture->capteur au GSD.
    # C'est le point clé — sans ça le détail HF adversarial est appris mais
    # détruit par l'aliasing avant d'atteindre le détecteur.
    res_degrade: Tuple[float, float] = (0.5, 1.0)      # facteur de downsample


@dataclass
class Config:
    # --- Géométrie physique -> pixels ---------------------------------------
    patch_size_m: float = 2.0        # côté physique du patch (m), ~toit de voiture
    gsd_m_per_px: float = 0.05       # GSD opérationnel (m/px). xView=0.30, drone~0.05
    frame_px: int = 1500             # dimension de la frame brute capteur
    net_size: int = 640              # entrée du détecteur (letterbox carré)

    # --- Cible ---------------------------------------------------------------
    target_class: int = 2            # id de classe à supprimer (ex. 'car')
    conf_thresh: float = 0.25        # seuil de détection pour l'ASR

    # --- Pertes --------------------------------------------------------------
    w_det: float = 1.0
    w_tv: float = 2.5
    w_nps: float = 0.05
    printable_colors: List[List[float]] = field(default_factory=list)  # NPS palette

    # --- Optimisation --------------------------------------------------------
    lr: float = 0.03
    iters: int = 1000
    batch_size: int = 8
    seed: int = 0
    device: str = "cuda"

    eot: EOTConfig = field(default_factory=EOTConfig)

    # ------------------------------------------------------------------------
    def patch_px_raw(self) -> int:
        """Footprint du patch dans la frame BRUTE (avant redimensionnement réseau)."""
        return round(self.patch_size_m / self.gsd_m_per_px)

    def patch_px_net(self) -> int:
        """Footprint EFFECTIF en entrée réseau : c'est là que vit le gradient.

        C'est ce nombre qu'il faut comparer au plancher de résolution (~30 px)
        pour savoir si l'attaque est faisable au GSD visé.
        """
        return round(self.patch_px_raw() * self.net_size / self.frame_px)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        eot = EOTConfig(**raw.pop("eot", {}))
        return cls(eot=eot, **raw)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
