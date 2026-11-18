"""Genera las tablas y figuras del informe a partir de outputs/ (no entrena nada).

    python generar_reporte.py [--pesos models/weights/effps_webcam_best.pt]
Salida: outputs/reporte/  (curvas_perdida.png, dataset_anotaciones.png, tabla_comparativa.csv, tablas_informe.xlsx, ...)
"""
import argparse

from src import reporte as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_limpio")
    ap.add_argument("--pesos", default=None, help="si se indica, tambien genera ejemplos de segmentacion con esos pesos")
    a = ap.parse_args()
    print(R.resumen_dataset(a.data), "\n")
    print(R.tabla_taller2_original().round(3), "\n")
    print(R.tabla_comparativa().drop(columns="Análisis").round(3).to_string(), "\n")
    R.fig_curvas(); R.fig_dataset(a.data); R.guardar_tablas()
    if a.pesos:
        R.fig_predicciones(a.pesos, a.data, R.REPORTE_DIR / "resultados_segmentacion.png")
    print("Listo: figuras y tablas en", R.REPORTE_DIR)


if __name__ == "__main__":
    main()
