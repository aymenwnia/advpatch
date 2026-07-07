#!/usr/bin/env python
"""Évalue un patch : ASR global + balayage de points de vue.

Exemple :
    python scripts/eval_patch.py --config configs/default.yaml \
        --weights yolov8n.pt --patch patch.pt
"""
from __future__ import annotations
import argparse
import torch

from adv_patch import Config, DetectorWrapper
from adv_patch.data import make_loader
from adv_patch.evaluate import compute_asr, viewpoint_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--patch", required=True)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    detector = DetectorWrapper(args.weights, cfg.target_class, cfg.device)
    loader = make_loader(cfg, real=args.real)
    patch = torch.load(args.patch)

    asr = compute_asr(cfg, detector, patch, loader)
    print(f"ASR global : {asr:.1%}")

    print("\nBalayage points de vue (échelle, rotation) -> ASR :")
    sweep = viewpoint_sweep(cfg, detector, patch, loader)
    for (s, r), v in sorted(sweep.items()):
        print(f"  échelle={s:>4}  rot={r:>4}°  ASR={v:.1%}")


if __name__ == "__main__":
    main()
