"""Entrena y evalua EffPS-Lite para segmentacion panoptica de madera balsa.

Ejemplos (en Colab con GPU o en terminal):
    # modelo del proyecto: inicia con los pesos del mejor modelo del Taller 2
    python train.py --tag effps_taller2 --init taller2 --init-weights models/weights/taller2_effps_512_best.pt
    # control: misma arquitectura iniciada solo con ImageNet (para comparar)
    python train.py --tag effps_imagenet --init imagenet
"""
import argparse

import torch

from models.effps_model import EffPS
from src.config import DATA_DIR, IMG_H, IMG_W, SEED
from src.data import make_loaders
from src.trainer import fit, plot_history, test_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--tag", default="effps_taller2", help="nombre del experimento (archivos de salida)")
    ap.add_argument("--init", choices=["taller2", "imagenet"], default="taller2")
    ap.add_argument("--init-weights", default=None, help=".pt del Taller 2 (EffPS-Lite 512 px) si --init taller2")
    ap.add_argument("--variant", default="efficientnet_b3", help="debe coincidir con el del Taller 2 para cargar sus pesos")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--p-multi", type=float, default=0.5, help="probabilidad de composicion con varias piezas en train")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--img-h", type=int, default=IMG_H)
    ap.add_argument("--img-w", type=int, default=IMG_W)
    ap.add_argument("--no-pretrained", action="store_true", help="no descargar pesos ImageNet (solo para pruebas sin internet)")
    args = ap.parse_args()
    if args.init == "taller2" and not args.init_weights:
        ap.error("--init taller2 necesita --init-weights <ruta al .pt del Taller 2>")

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)
    size = (args.img_h, args.img_w)
    _, loaders = make_loaders(args.data, args.batch, size, args.workers, args.p_multi)
    model = EffPS.build(device, args.variant, pretrained=not args.no_pretrained, img_size=size,
                        init_weights=args.init_weights if args.init == "taller2" else None)
    fit(model, loaders, device, epochs=args.epochs, lr=args.lr, patience=args.patience, tag=args.tag)
    test_best(EffPS, loaders, device, args.tag)
    plot_history([args.tag], f"outputs/curvas_{args.tag}.png")


if __name__ == "__main__":
    main()
