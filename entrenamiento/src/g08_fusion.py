"""Peso de la fusión tardía de G08, estimado por apilamiento (stacking) con las predicciones out-of-fold de train.

El puntaje de una ventana combina los puntajes estandarizados de las dos ramas,
    s(w) = α · s_a(w) + (1 − α) · s_i(w),
y el puntaje del automóvil es el máximo de sus ventanas. Se evalúan los valores de α entre 0 y 1
(`g08.fusion.paso_rejilla`) y se elige el que maximiza la AP de los candidatos de train.
Cada partición de train se puntúa con el α estimado en las otras dos, y el conjunto test, con el α estimado
en las tres.
"""
import numpy as np
import pandas as pd

import comun as c

CLAVE = ["video_id", "track_id"]


def rejilla():
    paso = c.P["g08"]["fusion"]["paso_rejilla"]
    return np.round(np.arange(0, 1 + paso / 2, paso), 4)


def puntaje_autos(pw, alfa):
    """S(T) = máx sobre las ventanas de α·s_a + (1 − α)·s_i."""
    s = alfa * pw.s_apariencia.to_numpy() + (1 - alfa) * pw.s_interaccion.to_numpy()
    return pw.assign(s=s).groupby(CLAVE).s.max().rename("S")


def ap_alfa(pw, autos, alfa):
    x = autos.join(puntaje_autos(pw, alfa), on=CLAVE, how="inner")
    return c.average_precision(x.etiqueta.to_numpy(bool), x.S.to_numpy(float))


def estimar(pw, autos):
    """α que maximiza la AP de `autos` (candidatos con etiqueta). Ante empates se elige el α más cercano a 0,5."""
    vals = [(a, ap_alfa(pw, autos, a)) for a in rejilla()]
    mejor = max(v for _, v in vals)
    a = min((a for a, v in vals if v >= mejor - 1e-12), key=lambda a: abs(a - 0.5))
    return float(a), {float(k): round(float(v), 4) for k, v in vals}


def aplicar(pw, pa, cand_mask):
    """Devuelve (pa con la columna S recalculada, pw con s_fusion recalculada, detalle de los α).
    Train (fold >= 0): α estimado en las otras particiones. Test (fold == -1): α de todo train."""
    pa, pw = pa.copy(), pw.copy()
    dev = pa[(pa.fold >= 0) & cand_mask]
    folds = sorted(dev.fold.unique())
    alfa_total, curva = estimar(pw[pw.fold >= 0], dev[CLAVE + ["etiqueta"]].set_index(CLAVE))
    alfa_fold = {}
    for k in folds:
        otras = dev[dev.fold != k]
        alfa_fold[int(k)], _ = estimar(pw[pw.fold.isin(otras.fold.unique())], otras[CLAVE + ["etiqueta"]].set_index(CLAVE))
    alfa_w = pw.fold.map(lambda f: alfa_fold.get(int(f), alfa_total)).to_numpy(float)
    pw["s_fusion"] = alfa_w * pw.s_apariencia + (1 - alfa_w) * pw.s_interaccion
    S = pw.groupby(CLAVE).s_fusion.max().rename("S_nuevo")
    pa = pa.merge(S, on=CLAVE, how="left")
    pa["S"] = pa.S_nuevo
    # puntaje con el α de todo train: es el que recibe test, por lo que el umbral de alerta se fija sobre él
    pa = pa.merge(puntaje_autos(pw, alfa_total).rename("S_alfa_total"), on=CLAVE, how="left")
    pa = pa.drop(columns="S_nuevo")
    return pa, pw, {"alfa_train": alfa_total, "alfa_por_particion": alfa_fold,
                    "ap_candidatos_por_alfa": curva}
