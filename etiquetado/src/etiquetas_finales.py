"""Etiquetas finales por infracción: anotaciones manuales válidas.

Fuentes, en orden de prioridad:
  - salidas/consolidado/etiquetas_<inf>.csv  anotaciones `ok` de los anotadores
  - salidas/anotacion_previa/                anotaciones manuales previas del mismo video, si existen
  - config/revision_manual/<inf>_*.csv       casos anotados en la revisión manual, con su decisión
Cada fila lleva `origen` y `confianza` (alta, media).
Un vehículo con intervalos que se solapan entre fuentes se cuenta una vez.
Salida: salidas/final/etiquetas_<inf>.csv y salidas/final/resumen.csv
Uso: python src/etiquetas_finales.py
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((RAIZ / P["rutas"]["procesamiento"]).resolve() / "src"))
import configuracion, indice  # noqa: E402

SAL = RAIZ / P["rutas"]["salidas"]
VER = RAIZ / "config" / "revision_manual"
COLS = ["video_id", "track_id", "k_ini", "k_fin", "hora_inicio", "hora_fin", "origen", "confianza", "detalle"]


def horas(df, ind):
    t0 = {v: (datetime.fromisoformat(f"{f.fecha}T{f.hora_cuadro0}"), float(f.fps)) for v, f in ind.iterrows()}
    h = lambda v, k: (t0[v][0] + timedelta(seconds=k / t0[v][1])).strftime("%H:%M:%S")  # noqa: E731
    df["hora_inicio"] = [h(v, k) for v, k in zip(df.video_id, df.k_ini)]
    df["hora_fin"] = [h(v, k) for v, k in zip(df.video_id, df.k_fin)]
    return df


def anotadas(inf):
    e = pd.read_csv(SAL / "consolidado" / f"etiquetas_{inf.lower()}.csv")
    e = e.assign(origen=e.anotador, confianza="alta", detalle=e.get("tipo_evento", ""))
    return e


def main():
    ind = indice.leer(configuracion.cargar())
    out = SAL / "final"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for inf in ("G10", "G41", "G08", "M21"):
        partes = [anotadas(inf)]
        ant = SAL / "anotacion_previa" / f"etiquetas_{inf.lower()}_anterior.csv"
        if ant.exists():
            a = pd.read_csv(ant)
            if inf == "G10":
                a = a[a.positivo].rename(columns={"f_ini": "k_ini", "f_fin": "k_fin"})
            elif inf == "G41":
                a = a[a.positivo].copy()
                a["k_ini"] = a.frame_ini.fillna(a.f_ini_episodio)
                a["k_fin"] = a.frame_fin.fillna(a.f_fin_episodio)
            elif inf == "G08":
                a = a.rename(columns={"f_ini": "k_ini", "f_fin": "k_fin"})
            else:
                a = a.iloc[0:0]   # M21 anterior: otra definición, no se usa
            partes.append(a.assign(origen="anotacion_previa", confianza="alta", detalle=a.get("tipo", "")))
        for v in sorted(VER.glob(f"{inf.lower()}_*.csv")):
            x = pd.read_csv(v)
            partes.append(x)
        e = pd.concat(partes, ignore_index=True)
        e = e.dropna(subset=["k_ini", "k_fin"]) if inf != "G10" else e
        e["track_id"] = e.track_id.astype(int)
        # intervalos solapados del mismo vehículo se cuentan una vez (prioridad: anotadores > previa > revisión)
        prio = {"anotador_a": 0, "anotador_b": 0, "anotacion_previa": 1}
        e["p"] = e.origen.map(prio).fillna(2)
        e = e.sort_values(["video_id", "track_id", "p"])
        if inf == "G10":
            e = e.drop_duplicates(["video_id", "track_id"])
        else:
            keep, ult = [], {}
            for r in e.itertuples():
                k = (r.video_id, r.track_id)
                if any(r.k_ini <= b and r.k_fin >= a_ for a_, b in ult.get(k, [])):
                    continue
                ult.setdefault(k, []).append((r.k_ini, r.k_fin))
                keep.append(r.Index)
            e = e.loc[keep]
        tr = {}
        if inf == "G10":   # G10 sin horas: el tramo del vehículo dentro del carril (de candidatos.py)
            for vid in e.video_id.unique():
                c = pd.read_csv(SAL / "candidatos" / f"{vid}_G10.csv").set_index("track_id")
                tr[vid] = c
            falta = e.k_ini.isna()
            for c in ("k_ini", "k_fin"):
                e.loc[falta, c] = pd.to_numeric(pd.Series([tr[v][c].get(t, np.nan) for v, t in zip(e.video_id[falta], e.track_id[falta])],
                                                          index=e.index[falta]), errors="coerce")
        sin = e.k_ini.isna() | e.k_fin.isna()
        if sin.any():
            print(f"{inf}: {int(sin.sum())} filas sin tramo dentro del carril (no pasan 7 cuadros en él); quedan con toda la vida del vehículo")
            for vid in e.video_id[sin].unique():
                f = ind.loc[vid]
                t = pd.read_csv(configuracion.cargar()["rutas"]["salidas"] / f.sesion / vid / f"{vid}_trayectorias_vehiculos.csv.gz")
                vida = t.groupby("track_id").frame.agg(["min", "max"])
                m = sin & (e.video_id == vid)
                e.loc[m, "k_ini"] = e.track_id[m].map(vida["min"])
                e.loc[m, "k_fin"] = e.track_id[m].map(vida["max"])
        e = e.dropna(subset=["k_ini", "k_fin"])
        e[["k_ini", "k_fin"]] = e[["k_ini", "k_fin"]].astype(int)
        if inf == "G08":   # confianza de la revisión visual de los eventos (registro_g08_anotados.csv)
            rg = pd.read_csv(VER / "registro_g08_anotados.csv")
            nv = set(zip(rg.video_id[rg.decision == "no_visible"], rg.track_id[rg.decision == "no_visible"].astype(int)))
            e.loc[[(v, int(t)) in nv for v, t in zip(e.video_id, e.track_id)], "confianza"] = "media"
        e = horas(e, ind)
        e[COLS].sort_values(["video_id", "k_ini"]).to_csv(out / f"etiquetas_{inf.lower()}.csv", index=False)
        res[inf] = e.groupby(["video_id", "origen"]).size()
    r = pd.concat(res, names=["infraccion"]).rename("filas").reset_index()
    r.to_csv(out / "resumen.csv", index=False)
    print(r.pivot_table(index=["infraccion", "origen"], columns="video_id", values="filas", fill_value=0).astype(int).to_string())


if __name__ == "__main__":
    main()
