"""Utilidades compartidas por los scripts: parámetros, rutas, lectura de las salidas del procesamiento
y métricas.

Lee config/parametros.yaml y los parámetros de los módulos vecinos procesamiento_videos y etiquetado
(rutas.procesamiento y rutas.etiquetado). No se ejecuta directamente.

Las métricas resuelven los empates como scikit-learn: el AUROC se calcula por Mann-Whitney con empates
a 1/2, y la AP como la suma de la precisión en cada umbral distinto, ponderada por el aumento de recall.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
R = {k: (Path(v) if Path(v).is_absolute() else (RAIZ / v).resolve()) for k, v in P["rutas"].items()}
PROC, ETQ = R["procesamiento"], R["etiquetado"]
sys.path.insert(0, str(PROC / "src"))
import configuracion as _conf  # noqa: E402
import indice as _indice  # noqa: E402
import zonas as _zonas  # noqa: E402

PP = _conf.cargar()                     # parámetros del procesamiento (zonas, fps, recortes)
IND = _indice.leer(PP)
ZONAS = _zonas.cargar(PP["rutas"]["zonas"])
PE = yaml.safe_load((ETQ / "config" / "parametros.yaml").read_text(encoding="utf-8"))


def videos():
    """Videos con etiquetas finales: los que se revisaron completos (cobertura del etiquetado)."""
    return sorted(pd.read_csv(ETQ / "salidas" / "consolidado" / "cobertura.csv").video_id.unique())


def carpeta_video(vid):
    return PP["rutas"]["salidas"] / IND.loc[vid, "sesion"] / vid


def fps(vid):
    return float(IND.loc[vid, "fps"])


def leer_csv(vid, sufijo, **kw):
    return pd.read_csv(carpeta_video(vid) / f"{vid}_{sufijo}", **kw)


def etiquetas(inf):
    return pd.read_csv(ETQ / "salidas" / "final" / f"etiquetas_{inf.lower()}.csv")


def guardar_json(obj, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def markdown(df):
    """Tabla markdown sin depender de `tabulate`."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    filas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    filas += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(filas)


def salida(*partes):
    d = R["resultados"].joinpath(*partes)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- métricas

def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _curva(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    if len(y) == 0:
        return np.zeros(0), np.zeros(0), 0.0, np.zeros(0)
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    ult = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[ult]
    return tp, (ult + 1) - tp, y.sum(), s[ult]


def average_precision(y, s):
    tp, fp, n_pos, _ = _curva(y, s)
    if n_pos == 0:
        return float("nan")
    prec, rec = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def umbral_max_f1(y, s):
    """Umbral (el puntaje más bajo que se declara positivo) que maximiza F1, y ese F1."""
    tp, fp, n_pos, umb = _curva(y, s)
    if n_pos == 0:
        return float("inf"), 0.0
    f1 = 2 * tp / (tp + fp + n_pos)
    i = int(np.argmax(f1))
    return float(umb[i]), float(f1[i])


def prf(y, d):
    """Precisión, recall y F1 de decisiones binarias."""
    y, d = np.asarray(y, bool), np.asarray(d, bool)
    tp, fp, fn = int((y & d).sum()), int((~y & d).sum()), int((y & ~d).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"vp": tp, "fp": fp, "fn": fn, "vn": int((~y & ~d).sum()), "precision": p, "recall": r, "f1": f}


def horas_video(vids):
    return float(sum(IND.loc[v, "cuadros"] / IND.loc[v, "fps"] for v in set(vids)) / 3600)


def umbral_alertas(puntajes, horas, alertas_por_hora):
    """Puntaje mínimo para emitir `alertas_por_hora` alertas por hora en el conjunto dado (train)."""
    s = np.sort(np.asarray(puntajes, float))[::-1]
    k = int(round(alertas_por_hora * horas))
    return float(s[min(k, len(s)) - 1]) if len(s) else float("inf")


def punto_operacion(y, s, umbral, horas):
    y, d = np.asarray(y, bool), np.asarray(s, float) >= umbral
    return {"umbral": round(umbral, 4), "alertas_por_hora": round(float(d.sum() / horas), 1),
            "eventos_encontrados": int((y & d).sum()), "eventos": int(y.sum()),
            "recall": round(float((y & d).sum() / max(1, y.sum())), 4),
            "precision": round(float((y & d).sum() / max(1, d.sum())), 4)}


def bootstrap(y, fn, b, semilla, grupos=None):
    """Intervalo percentil al 95 % de fn(índices), remuestreando positivos y negativos por separado.
    Si hay `grupos` (p. ej. el vehículo), se remuestrean grupos enteros, para respetar la dependencia
    entre unidades del mismo vehículo."""
    y = np.asarray(y, bool)
    rng = np.random.default_rng(semilla)
    if grupos is None:
        ip, i0 = np.nonzero(y)[0], np.nonzero(~y)[0]
        sorteo = lambda: np.r_[rng.choice(ip, len(ip)), rng.choice(i0, len(i0))]  # noqa: E731
    else:
        cod, _ = pd.factorize(np.asarray(grupos))
        orden = np.argsort(cod, kind="stable")
        largo = np.bincount(cod)
        ini = np.r_[0, np.cumsum(largo)[:-1]]
        tiene = np.bincount(cod, weights=y.astype(float)) > 0
        gp, g0 = np.nonzero(tiene)[0], np.nonzero(~tiene)[0]

        def sorteo():
            g = np.r_[rng.choice(gp, len(gp)), rng.choice(g0, len(g0))]
            n = largo[g]
            desde = np.repeat(ini[g] - np.r_[0, np.cumsum(n)[:-1]], n)
            return orden[desde + np.arange(n.sum())]
    v = np.array([fn(sorteo()) for _ in range(b)], float)
    v = v[np.isfinite(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else [None, None]
