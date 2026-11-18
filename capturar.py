"""Toma fotos con la camara (webcam, DroidCam o celular) y las guarda en una carpeta.

Sirve para armar fotos con varias piezas de balsa: cambias las piezas de lugar y pulsas una tecla.

Ejemplos:
    python capturar.py                                         # webcam 0, guarda en fotos_multi/
    python capturar.py --fuente 1                              # otra camara (p. ej. DroidCam Client)
    python capturar.py --fuente http://192.168.1.20:4747/video # DroidCam por WiFi
    python capturar.py --auto 2                                # ademas, una foto cada 2 s al activar el modo auto

Teclas:  ESPACIO o s = guardar una foto    a = activar/desactivar modo automatico    q o ESC = salir
"""
import argparse
import os
import time
from pathlib import Path

import cv2


def abrir(fuente, ancho=None, alto=None):
    """Abre una camara (indice o URL) o un video; en Windows usa DirectShow para los indices."""
    src = int(fuente) if str(fuente).isdigit() else fuente
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (os.name == "nt" and isinstance(src, int)) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente '{fuente}'. Revisa el indice de la camara o la URL de DroidCam.")
    if ancho and alto:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # menos retraso con DroidCam
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0", help="indice de camara, URL de DroidCam o video")
    ap.add_argument("--carpeta", default="fotos_multi", help="carpeta donde se guardan las fotos")
    ap.add_argument("--auto", type=float, default=2.0, help="segundos entre fotos en modo automatico (tecla a)")
    ap.add_argument("--ancho", type=int, default=None)
    ap.add_argument("--alto", type=int, default=None)
    ap.add_argument("--sin-ventana", action="store_true", help="solo para pruebas: arranca en modo auto sin ventana")
    ap.add_argument("--max", type=int, default=None, help="parar tras guardar N fotos")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    n = len(list(carpeta.glob("*.jpg")))          # sigue la numeracion si la carpeta ya tenia fotos
    cap = abrir(args.fuente, args.ancho, args.alto)
    auto, t_ultima = args.sin_ventana, 0.0

    def guardar(frame):
        nonlocal n, t_ultima
        n += 1
        ruta = carpeta / f"foto_{n:04d}.jpg"
        cv2.imwrite(str(ruta), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        t_ultima = time.time()
        print("Guardada:", ruta)

    print(f"Guardando en: {carpeta.resolve()}  |  ESPACIO/s = foto, a = auto cada {args.auto:g}s, q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se termino el video o se perdio la camara.")
            break
        if auto and time.time() - t_ultima >= args.auto:
            guardar(frame)
        if args.max and n >= args.max:
            break
        if args.sin_ventana:
            continue
        vista = frame.copy()
        estado = f"Fotos: {n}   AUTO {'ON' if auto else 'OFF'}"
        cv2.rectangle(vista, (0, 0), (vista.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(vista, estado, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if auto else (255, 255, 255), 2)
        cv2.imshow("Captura de fotos", vista)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord(" "), ord("s")):
            guardar(frame)
        elif k == ord("a"):
            auto = not auto
            t_ultima = time.time()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo: {n} fotos en {carpeta.resolve()}")


if __name__ == "__main__":
    main()
