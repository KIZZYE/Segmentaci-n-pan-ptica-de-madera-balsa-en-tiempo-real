"""Reproduce las metricas de un modelo ya entrenado (sin volver a entrenar).

    python evaluar.py --pesos models/weights/effps_taller2_best.pt --tag effps_taller2
    python evaluar.py --pesos models/weights/effps_taller2_best.pt --data data/dataset_camara --tag effps_camara

Escribe outputs/metrics_<tag>.json (IoU, precision, recall, F1, mAP por clase y AP50 de piezas).
"""
import argparse

import torch

from models.effps_model import EffPS
from src.config import DATA_DIR
from src.data import make_loaders
from src.trainer import test_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True, help="archivo .pt entrenado")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--tag", default="evaluacion")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args.data, 4, num_workers=args.workers, p_multi=0.0)
    test_best(EffPS, loaders, device, args.tag, weights_path=args.pesos)


if __name__ == "__main__":
    main()
