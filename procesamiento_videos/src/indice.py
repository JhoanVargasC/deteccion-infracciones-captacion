"""Índice de los videos de entrada: una fila por archivo, sin copiar los videos.

Columnas: video_id, sesion, fecha, periodo, hora_num, archivo, ruta, bytes, ancho, alto,
cuadros, fps_contenedor, duracion_s, y, si existe la lectura del reloj en pantalla
(datos/lecturas_reloj.csv), hora_cuadro0, fps_reloj y fps (el que usa el flujo).

Lectura del reloj (una fila por video): k_ini es el primer cuadro en que el reloj muestra
un segundo nuevo y t_ini esa hora; k_fin y t_fin, lo mismo cerca del final. Entonces
    hora del cuadro 0 = t_ini - k_ini / fps        fps_reloj = (k_fin - k_ini) / (t_fin - t_ini)

Entrada: carpeta de videos (rutas.videos_entrada) y datos/lecturas_reloj.csv.
Salida: datos/indice_videos.csv.

Uso: python src/indice.py
"""
import argparse
import re
from datetime import datetime, timedelta

import cv2
import pandas as pd

import configuracion


def escanear(p):
    cfg = p["indice"]
    filas = []
    for ruta in sorted(p["rutas"]["videos_entrada"].glob(f"*{cfg['extension']}")):
        m = re.search(cfg["patron_nombre"], ruta.name)
        if not m:
            print(f"Se omite (el nombre no sigue el patrón): {ruta.name}")
            continue
        fecha, hh, mm, ss, n = m.groups()
        periodo = "tarde" if int(hh) >= cfg["hora_corte_tarde"] else "manana"
        sesion = f"{fecha}_{periodo}"
        cap = cv2.VideoCapture(str(ruta))
        cuadros, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        filas.append({"video_id": f"{sesion}_h{int(n)}", "sesion": sesion, "fecha": fecha, "periodo": periodo,
                      "hora_num": int(n), "archivo": ruta.name, "ruta": str(ruta), "bytes": ruta.stat().st_size,
                      "ancho": ancho, "alto": alto, "cuadros": cuadros, "fps_contenedor": fps,
                      "duracion_s": round(cuadros / fps, 2)})
    return pd.DataFrame(filas).sort_values(["fecha", "periodo", "hora_num"]).reset_index(drop=True)


def agregar_reloj(ind, p):
    ruta = p["rutas"]["lecturas_reloj"]
    if not ruta.exists():
        print(f"Sin {ruta.name}: el índice queda sin hora de inicio ni fps medidos.")
        return ind
    lec = pd.read_csv(ruta, dtype={"t_ini": str, "t_fin": str})
    ind = ind.merge(lec, on="video_id", how="left")
    seg = lambda f, t: datetime.fromisoformat(f"{f}T{t}")  # noqa: E731
    ind["fps_reloj"] = [(kf - ki) / (seg(f, tf) - seg(f, ti)).total_seconds()
                        for f, ki, ti, kf, tf in zip(ind.fecha, ind.k_ini, ind.t_ini, ind.k_fin, ind.t_fin)]
    ind["fps"] = ind[f"fps_{p['indice']['fuente_fps']}"]
    # El contenedor declara un cuadro menos de los que se leen hasta el final (medido en los videos de la cámara)
    ind = ind.rename(columns={"cuadros": "cuadros_contenedor"})
    ind["cuadros"] = ind.cuadros_reales.fillna(ind.cuadros_contenedor).astype(int)
    ind["duracion_s"] = (ind.cuadros / ind.fps).round(2)
    ind["hora_cuadro0"] = [(seg(f, ti) - timedelta(seconds=ki / fps)).strftime("%H:%M:%S.%f")[:-4]
                           for f, ti, ki, fps in zip(ind.fecha, ind.t_ini, ind.k_ini, ind.fps)]
    return ind


def leer(p):
    """Índice como DataFrame indexado por video_id."""
    return pd.read_csv(p["rutas"]["indice_videos"], dtype={"hora_cuadro0": str}).set_index("video_id", drop=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    p = configuracion.cargar(a.config)
    ind = agregar_reloj(escanear(p), p)
    p["rutas"]["indice_videos"].parent.mkdir(parents=True, exist_ok=True)
    ind.to_csv(p["rutas"]["indice_videos"], index=False)
    print(ind.drop(columns=["ruta", "archivo"]).to_string())


if __name__ == "__main__":
    main()
