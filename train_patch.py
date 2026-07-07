#!/usr/bin/env python
"""Entraîne un patch adversarial.

Exemple :
    python scripts/train_patch.py --config configs/default.yaml \
        --weights yolov8n.pt --out patch.pt
"""
from __future__ import annotations
import argparse
import torch

from adv_patch import Config, DetectorWrapper, PatchAttack
from adv_patch.data import make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--out", default="patch.pt")
    ap.add_argument("--real", action="store_true", help="utiliser le dataset réel")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"Patch : {cfg.patch_px_net()} px réseau "
          f"(brut {cfg.patch_px_raw()} px, {cfg.patch_size_m} m @ {cfg.gsd_m_per_px} m/px)")
    if cfg.patch_px_net() < 30:
        print("  ! footprint < 30 px : sous le plancher de résolution, attaque "
              "probablement infaisable à ce GSD (cf. spec).")

    detector = DetectorWrapper(args.weights, cfg.target_class, cfg.device)
    loader = make_loader(cfg, real=args.real)

    attack = PatchAttack(cfg, detector)
    patch = attack.train(loader)

    torch.save(patch.image(), args.out)
    print(f"Patch sauvegardé -> {args.out}")


if __name__ == "__main__":
    main()
