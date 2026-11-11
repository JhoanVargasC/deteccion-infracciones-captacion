"""Período de iluminación (día o noche) de cada bloque de 10 minutos, medido en el video.

La hora del reloj no basta: la hora 17:57-18:57 cruza la puesta de sol (en Lima, a inicios de abril, hacia
las 18:10). Por eso el período se mide con la luminosidad de la escena. En cada bloque se toman
`cuadros_por_bloque` cuadros repartidos en el bloque y se calcula la mediana del brillo (canal V de HSV,
0-255) en la franja superior de la imagen (cielo y fachadas, `franja_cielo_px`). El brillo de la vía no
sirve: la cámara ajusta su exposición y de noche la vía queda iluminada por faros y postes, de modo que
ambos períodos tienen valores parecidos. El cielo no depende de la exposición de la vía.
El umbral entre día y noche es el de Otsu sobre los tramos de todos los videos, el mismo método que se usa
para el estado del tránsito. No se ajusta con etiquetas. La posición del cuadro
con seek es aproximada en HEVC, lo que no afecta la medida del brillo de un bloque de 10 minutos.

Entrada: videos del procesamiento (índice de videos).
Salida: rutas.dataset/iluminacion.csv (video_id, bloque, brillo, periodo) e iluminacion.json (umbral)
Uso: python src/iluminacion.py
"""
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import comun as c

IL = c.P["iluminacion"]


def otsu(x):
    """Umbral de Otsu sobre valores continuos (histograma de 256 intervalos)."""
    h, e = np.histogram(x, bins=256)
    centros = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * centros) / np.maximum(w0, 1)
    m1 = ((h * centros).sum() - np.cumsum(h * centros)) / np.maximum(w1, 1)
    return float(centros[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def main():
    inicio = {s: datetime.fromisoformat(f"{g.fecha.iloc[0]}T{g.hora_cuadro0.iloc[0]}") for s, g in c.IND.groupby("sesion")}
    filas = []
    for vid, f in c.IND.iterrows():
        cap = cv2.VideoCapture(str(f.ruta))
        fps, n = float(f.fps), int(f.cuadros)
        t0 = (datetime.fromisoformat(f"{f.fecha}T{f.hora_cuadro0}") - inicio[f.sesion]).total_seconds()
        blo = 60 * c.P["dataset"]["bloque_min"]
        for b in range(int(t0 // blo), int((t0 + n / fps) // blo) + 1):
            a, z = max(0.0, b * blo - t0), min(n / fps, (b + 1) * blo - t0)
            if z - a < IL["min_segundos_bloque"]:
                continue
            v = []
            for s in np.linspace(a, z, IL["cuadros_por_bloque"] + 2)[1:-1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
                ok, im = cap.read()
                if ok:
                    v.append(np.median(cv2.cvtColor(im[:IL["franja_cielo_px"]], cv2.COLOR_BGR2HSV)[..., 2]))
            if v:
                filas.append({"video_id": vid, "sesion": f.sesion, "bloque": b, "brillo": round(float(np.median(v)), 1)})
        cap.release()
        print(vid, [x["brillo"] for x in filas if x["video_id"] == vid], flush=True)
    d = pd.DataFrame(filas)
    u = otsu(d.brillo.to_numpy())
    d["periodo"] = np.where(d.brillo >= u, "dia", "noche")
    d.to_csv(c.R["dataset"] / "iluminacion.csv", index=False)
    c.guardar_json({"umbral_otsu_brillo": round(u, 1), "bloques": len(d),
                    "por_video": d.groupby("video_id").periodo.agg(lambda s: s.value_counts().to_dict()).to_dict()},
                   c.R["dataset"] / "iluminacion.json")
    print(f"umbral de Otsu: {u:.1f}")
    print(d.pivot_table(index="video_id", columns="periodo", values="bloque", aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
