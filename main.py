"""Segmentacion panoptica de madera balsa en tiempo real (camara, video o carpeta de imagenes).

Muestra, sobre cada fotograma:
  * "things": cada pieza de balsa por separado, con su largo y ancho (px o cm);
  * "stuff": las regiones del entorno (superficie de trabajo y fondo).

Ejemplos:
    python main.py                                   # camara 0 (webcam)
    python main.py --fuente http://192.168.1.20:4747/video   # celular con DroidCam
    python main.py --fuente videos/prueba.mp4        # archivo de video
    python main.py --fuente data/dataset_limpio/test/images  # carpeta de imagenes
    python main.py --cm-por-px 0.0185                # medidas en cm (escala conocida)

Camara: el video va fluido y el modelo trabaja en segundo plano (--sincrono lo desactiva).
Teclas:  z = marcar zona de trabajo (ignora lo de afuera)   c = calibrar (px -> cm)   g = mostrar/ocultar stuff   m = mostrar la mascara real   s = guardar captura
         espacio = pausa           q o ESC = salir
"""
import argparse
import glob
import os
import threading
import time

import cv2

from src.config import OUTPUTS_DIR, WEIGHTS_DIR
from src.inferencia import Segmentador
from src.movimiento import Compensador
from src.medidas import Seguidor, cargar_calibracion, cargar_zona, guardar_calibracion, guardar_zona, medir
from src.visual import dibujar

VENTANA = "Segmentacion panoptica - madera balsa"
ZONA = {"rect": None}      # zona de trabajo (fracciones x0,y0,x1,y1): lo que cae fuera se ignora


def fotogramas(fuente):
    """Generador de fotogramas BGR desde camara (indice o URL), archivo de video o carpeta de imagenes."""
    if os.path.isdir(fuente):                                    # carpeta de imagenes
        for p in sorted(glob.glob(os.path.join(fuente, "*.jpg")) + glob.glob(os.path.join(fuente, "*.png"))):
            yield cv2.imread(p)
        return
    src = int(fuente) if str(fuente).isdigit() else fuente
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (os.name == "nt" and isinstance(src, int)) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente '{fuente}'. Revisa el indice de camara, la URL o la ruta.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


class CamaraViva:
    """Lee la camara en un hilo aparte y guarda solo el ULTIMO fotograma (sin cola => sin retraso acumulado)."""

    def __init__(self, fuente):
        src = int(fuente) if str(fuente).isdigit() else fuente
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (os.name == "nt" and isinstance(src, int)) else cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"No se pudo abrir la fuente '{fuente}'. Revisa el indice de camara, la URL o la ruta.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.activo, self.lock = None, True, threading.Lock()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        while self.activo:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def leer(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def cerrar(self):
        self.activo = False
        time.sleep(0.05)
        self.cap.release()


class Trabajador:
    """Segmenta en un hilo aparte el ultimo fotograma disponible; la ventana nunca espera al modelo."""

    def __init__(self, modelo, seguidor, camara):
        self.modelo, self.seguidor, self.camara = modelo, seguidor, camara
        self.salida, self.fps, self.n, self.activo, self.lock = None, None, 0, True, threading.Lock()
        threading.Thread(target=self._bucle, daemon=True).start()

    def _bucle(self):
        while self.activo:
            frame = self.camara.leer()
            if frame is None:
                time.sleep(0.01)
                continue
            t = time.time()
            res = self.modelo.segmentar(frame)
            piezas = self.seguidor.actualizar(piezas_medidas(res))
            dt = max(time.time() - t, 1e-6)
            with self.lock:
                self.salida = (res, piezas, frame)
                self.n += 1
                self.fps = 1.0 / dt if self.fps is None else 0.8 * self.fps + 0.2 / dt

    def resultado(self):
        with self.lock:
            return (None, [], None, None, 0) if self.salida is None else (*self.salida[:2], self.fps, self.salida[2], self.n)


def es_camara(fuente):
    return str(fuente).isdigit() or str(fuente).startswith(("http://", "https://", "rtsp://"))


def flujo_fluido(fuente, modelo, seguidor, compensar=True):
    """Camara en vivo: video fluido + resultado del modelo en segundo plano, con el contorno siguiendo el movimiento."""
    cam = CamaraViva(fuente)
    trab = Trabajador(modelo, seguidor, cam)
    comp, ultimo = Compensador(), 0
    try:
        while True:
            frame = cam.leer()
            if frame is None:
                time.sleep(0.01)
                continue
            res, piezas, fps_m, origen, n = trab.resultado()
            if res is not None and compensar:
                if n != ultimo:                                  # llego un resultado nuevo del modelo
                    comp.fijar(origen, res, piezas)
                    ultimo = n
                res, piezas = comp.aplicar(frame)
            yield frame, res, piezas, fps_m
    finally:
        trab.activo = False
        cam.cerrar()


def flujo_sincrono(fuente, modelo, seguidor, cada):
    """Video o carpeta: procesa cada fotograma en orden (no se salta ninguno)."""
    res, piezas = None, []
    for i, frame in enumerate(fotogramas(fuente)):
        if i % max(cada, 1) == 0:
            t = time.time()
            res = modelo.segmentar(frame)
            piezas = seguidor.actualizar(piezas_medidas(res))
        yield frame, res, piezas, None


def piezas_medidas(res):
    """Mide cada pieza (thing) detectada: largo y ancho en pixeles. Descarta astillas, objetos enormes y lo que
    queda fuera de la zona de trabajo (por ejemplo una silla al fondo)."""
    out = []
    H, W = res["seg"].shape
    z = ZONA["rect"]
    for s in res["segments"]:
        if not s["is_thing"]:
            continue
        m = medir(res["seg"] == s["id"])
        if not m or m["ancho"] < 0.03 * min(H, W) or m["largo"] * m["ancho"] > 0.35 * H * W:
            continue
        if m["relleno"] < 0.82:                                    # un bloque llena casi todo su rectangulo; personas/objetos no
            continue
        cx, cy = m["centro"]
        if z and not (z[0] * W <= cx <= z[2] * W and z[1] * H <= cy <= z[3] * H):
            continue
        m["id"] = s["id"]
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0", help="indice de camara, URL (DroidCam), video o carpeta de imagenes")
    ap.add_argument("--pesos", default=str(WEIGHTS_DIR / "effps_taller2_best.pt"))
    ap.add_argument("--score", type=float, default=0.5, help="confianza minima para aceptar una pieza")
    ap.add_argument("--cada", type=int, default=1, help="(solo video/carpeta) segmentar 1 de cada N fotogramas")
    ap.add_argument("--sincrono", action="store_true", help="camara sin hilos (el video va al ritmo del modelo)")
    ap.add_argument("--sin-flujo", action="store_true", help="no compensar el movimiento entre resultados del modelo")
    ap.add_argument("--escala", type=float, default=1.0, help="tamano de entrada del modelo (0.6 = mas rapido, algo menos preciso)")
    ap.add_argument("--hilos", type=int, default=None, help="hilos de CPU para PyTorch")
    ap.add_argument("--cm-por-px", type=float, default=None, help="escala conocida; si no, usa calibracion.json")
    ap.add_argument("--salida", default=None, help="guardar el resultado en un video .mp4")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (para pruebas)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    modelo = Segmentador(args.pesos, score_thr=args.score, hilos=args.hilos, escala=args.escala)
    print(f"Modelo cargado en {modelo.device} (entrada {modelo.h}x{modelo.w}). Teclas: z zona de trabajo, c calibrar, g stuff, s captura, q salir")
    seguidor = Seguidor()
    ZONA["rect"] = cargar_zona()
    cm = args.cm_por_px or cargar_calibracion()
    fps, t0, writer, stuff, pausa, mascara = None, time.time(), None, True, False, False
    OUTPUTS_DIR.mkdir(exist_ok=True)
    if es_camara(args.fuente) and not args.sincrono:
        flujo = flujo_fluido(args.fuente, modelo, seguidor, not args.sin_flujo)
    else:
        flujo = flujo_sincrono(args.fuente, modelo, seguidor, args.cada)

    for i, (frame, res, piezas, fps_m) in enumerate(flujo):
        if args.max_frames and i >= args.max_frames:
            break
        if res is None:                                          # el modelo aun no termina el primer fotograma
            img = frame.copy()
            cv2.putText(img, "Cargando modelo...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            img = dibujar(frame, res, piezas, cm, fps, stuff, fps_m, mascara)
        if ZONA["rect"]:
            hh, ww = img.shape[:2]
            z = ZONA["rect"]
            cv2.rectangle(img, (int(z[0] * ww), int(z[1] * hh)), (int(z[2] * ww), int(z[3] * hh)), (0, 255, 255), 1)
        t1 = time.time()
        fps = 1.0 / max(t1 - t0, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(t1 - t0, 1e-6)
        t0 = t1
        if args.salida:
            if writer is None:
                writer = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"), 10, (img.shape[1], img.shape[0]))
            writer.write(img)
        if args.sin_ventana:
            continue
        cv2.imshow(VENTANA, img)
        while True:
            k = cv2.waitKey(0 if pausa else 1) & 0xFF
            if k == ord(" "):
                pausa = not pausa
            if not pausa or k != 255:
                break
        if k in (ord("q"), 27):
            break
        elif k == ord("z"):                                       # marcar la zona de trabajo con el raton
            x, y, w, h = cv2.selectROI("Zona de trabajo (ENTER confirma)", frame, showCrosshair=False)
            cv2.destroyWindow("Zona de trabajo (ENTER confirma)")
            fh, fw = frame.shape[:2]
            ZONA["rect"] = (x / fw, y / fh, (x + w) / fw, (y + h) / fh) if w > 10 and h > 10 else None
            guardar_zona(ZONA["rect"])
            print("Zona de trabajo:", "borrada" if ZONA["rect"] is None else "guardada")
        elif k == ord("m"):
            mascara = not mascara
        elif k == ord("g"):
            stuff = not stuff
        elif k == ord("s"):
            ruta = OUTPUTS_DIR / f"captura_{int(time.time())}.png"
            cv2.imwrite(str(ruta), img)
            print("Captura guardada:", ruta)
        elif k == ord("c") and piezas:                           # calibracion: la pieza mas grande tiene largo conocido
            p = max(piezas, key=lambda q: q["largo"])
            try:
                real = float(input(f"Largo REAL de la pieza {p['num']} en cm (mide con regla): "))
                cm = real / p["largo"]
                guardar_calibracion(cm)
                print(f"Calibrado: {cm:.5f} cm por pixel (guardado en calibracion.json)")
            except ValueError:
                print("Valor no valido, no se calibro.")
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
