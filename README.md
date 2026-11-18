# Segmentación panóptica de madera balsa en tiempo real

Proyecto de Corte (3° Corte) – Electiva IV Deep Computer Vision (USTA).
Modelo: **EffPS-Lite** (EfficientNet-B3 + FPN + cabeza semántica + Mask R-CNN para instancias + fusión panóptica),
adaptado del mejor modelo del Taller 2 (Cartografía Automatizada).

- **Things:** cada pieza de balsa, con su largo y ancho (px o cm).
- **Stuff:** `Fondo` y `Superficie de trabajo`.

## 1. Instalación
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
Para solo ejecutar la app basta con `torch torchvision opencv-python numpy scipy`.

## 2. Estructura
```
data/dataset_limpio/{train,val,test}/{images,semantic,instances}   # datos (no incluidos en el zip)
models/effps_model.py     # arquitectura, pérdidas y fusión panóptica
models/weights/           # pesos .pt (aquí va effps_taller2_best.pt)
src/config.py             # clases, tamaño de entrada, mínimos exigidos
src/data.py, composite.py # dataset, aumentos y composiciones con varias piezas
src/metrics.py, trainer.py# métricas, bucle de entrenamiento, curvas
src/prelabel_sam.py       # pre-etiquetado con SAM (solo herramienta de anotación)
src/prelabel_cajas.py, mezclar.py          # etiquetado por cajas (webcam) y mezcla de datasets
src/inferencia.py, medidas.py, movimiento.py, visual.py   # app en tiempo real
src/reporte.py, dataset_camara.py          # tablas, curvas y figuras del informe; prueba por cámara
train.py   evaluar.py   generar_reporte.py   main.py   capturar.py
notebooks/Proyecto_balsa_completo_colab.ipynb   # todo el flujo de entrenamiento y resultados en Colab (solo para usar GPU)
```

## 3. Preparación de datos
1. Fotos de las piezas desde varios ángulos y celulares (iPhone 11, iPhone 15 Pro Max, DroidCam).
2. `notebooks/01_prelabel_colab.ipynb` (Colab, GPU) genera máscaras con **SAM** guiado por semillas de color;
   SAM se usa **solo para anotar**, no forma parte del modelo entregado.
3. Se revisaron a mano las hojas de contacto y se descartaron las máscaras malas. Queda un dataset de 336 imágenes con
   `semantic/` (0 fondo, 1 superficie de trabajo, 2 pieza) e `instances/` (un id por pieza).
4. División en bloques de fotogramas consecutivos (train 240 / val 48 / test 48) para evitar fugas entre imágenes casi iguales.
5. Piezas múltiples: como solo hay una pieza por foto real, `src/composite.py` pega 1–3 piezas extra en las imágenes de
   entrenamiento (copy-paste) y arma un **test sintético fijo** de 48 imágenes con piezas del test.

### 3.1 Fotos con la webcam (escena de la demo)
Para que el modelo funcione en la escena real de la demo (webcam 640x480, piezas en las manos, ventana al fondo) se
tomaron 34 fotos con `capturar.py` y se etiquetaron con `src/prelabel_cajas.py`:
- `data/cajas_webcam.json` guarda, por foto, una caja `[x1, y1, x2, y2]` por pieza, dibujada a mano.
- SAM recorta la forma exacta de cada pieza dentro de su caja; lo que no es pieza queda como `Fondo`.
- Las fotos 4:3 se rellenan a 16:9 (borde repetido), igual que hace la app al inferir.
- `src/mezclar.py` une este conjunto con `dataset_limpio` (las fotos de la webcam se repiten 4 veces con aumentos distintos).
```bash
python src/prelabel_cajas.py --fotos fotos_multi --cajas data/cajas_webcam.json --out data/dataset_webcam
python src/mezclar.py --base data/dataset_limpio --extra data/dataset_webcam --out data/dataset_mixto --rep 4
python train.py --data data/dataset_mixto --tag effps_webcam --init taller2 --init-weights models/weights/effps_taller2_best.pt --epochs 30 --lr 5e-5
python evaluar.py --pesos models/weights/effps_webcam_best.pt --data data/dataset_webcam --tag webcam_despues
```
(`notebooks/03_reentrenar_webcam_colab.ipynb` ejecuta estos mismos comandos en Colab con GPU.)

## 4. Entrenar
En Colab con GPU (`notebooks/Proyecto_balsa_completo_colab.ipynb`, que además hace la prueba por cámara, el reentrenamiento con la webcam y reúne los resultados) o en una terminal con GPU:
```bash
# modelo del proyecto: inicia con los pesos del mejor modelo del Taller 2
python train.py --tag effps_taller2 --init taller2 --init-weights models/weights/taller2_effps_512_best.pt --epochs 40
# control: misma arquitectura, solo ImageNet
python train.py --tag effps_imagenet --init imagenet --epochs 40
```
Genera `models/weights/<tag>_best.pt`, `outputs/history_<tag>.json`, `outputs/curvas_<tag>.png` (pérdida de
entrenamiento vs validación) y `outputs/metrics_<tag>.json`.

## 5. Ejecutar en tiempo real
Copia `effps_taller2_best.pt` a `models/weights/`. Luego:
```bash
python main.py                                       # webcam 0
python main.py --fuente 1                            # otra cámara
python main.py --fuente http://IP_DEL_CELULAR:4747/video   # DroidCam
python main.py --fuente videos/prueba.mp4            # video
python main.py --fuente data/dataset_limpio/test/images    # carpeta de imágenes
```
Teclas: `c` calibrar px→cm · `g` mostrar/ocultar stuff · `s` guardar captura · espacio pausa · `q`/ESC salir.

**Medidas en cm:** pon una pieza cuyo largo conozcas (mídela con regla), pulsa `c` y escribe el largo real en cm en la
terminal. Se guarda `calibracion.json` y desde ahí todas las piezas se muestran en cm. También puedes pasar
`--cm-por-px 0.0185`. La escala es válida mientras la cámara y la distancia a la mesa no cambien, y las piezas estén
sobre el mismo plano que la pieza de referencia.

**Sin GPU:** el modelo corre a ~2–3 FPS en CPU. Con cámara, el video se muestra fluido y el modelo trabaja en segundo plano (el contorno se actualiza ~3 veces por segundo y entre actualizaciones sigue el movimiento de cada pieza con flujo óptico; `--sin-flujo` lo desactiva). Para ir más rápido usa `--escala 0.75` o `0.6` (entrada más chica). Con la tecla `z` se marca la zona de trabajo con el ratón: lo que quede fuera (sillas, objetos del fondo) se ignora; se guarda en `zona.json`; `--hilos N` ajusta los hilos de CPU y `--sincrono` desactiva este modo. Con video o carpeta de imágenes, `--cada N` segmenta 1 de cada N fotogramas.

## 6. Reproducir métricas
```bash
python evaluar.py --pesos models/weights/effps_taller2_best.pt --tag effps_taller2
```
Escribe `outputs/metrics_effps_taller2.json` con IoU, precisión, recall, F1 y mAP (por clase y global) en test real y
test sintético, y AP50 de piezas. Mínimos del proyecto: IoU ≥ 0.7, mAP ≥ 0.6, F1 ≥ 0.8.

Los promedios "overall" se calculan **solo sobre las clases que existen en el conjunto evaluado** (`src/metrics.py`,
clave `clases_promediadas`). En las fotos de la webcam no hay mesa de trabajo, así que `Superficie de trabajo` no tiene
píxeles de referencia: no se puede evaluar y, si se promediara como 0, bajaría el resultado sin que el modelo falle.
Las métricas por clase siempre se reportan completas.

### Tablas y figuras del informe
```bash
python generar_reporte.py --pesos models/weights/effps_webcam_best.pt
```
Lee `outputs/` (sin entrenar) y deja en `outputs/reporte/`: tabla comparativa (CSV y Excel), resultados originales del Taller 2
(aparte), curvas de pérdida de entrenamiento y validación, ejemplos del dataset con sus anotaciones y ejemplos de segmentación.

## 7. Notas
- Los pesos del Taller 2 son de clases satelitales: no se pueden evaluar directamente sobre balsa. Se usan como
  inicialización (se omite la última capa semántica) y los resultados originales del taller se reportan aparte.
- Limitaciones: el test sale de las mismas escenas que el entrenamiento; la prueba por cámara (`data/dataset_camara`,
  entrenar sin iPhone 15 y probar en él) da la medida de generalización. Las piezas muy claras están poco representadas.
