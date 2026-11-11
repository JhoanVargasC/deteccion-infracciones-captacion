"""Señales de detención, posición y tránsito por video (Etapa 3).

Sobre las trayectorias de vehículos y personas se calculan:
  - detenciones estrictas: el punto de contacto suavizado permanece a menos de D alturas de caja del punto
    donde empezó el intervalo, durante al menos S segundos, con hasta 3 fotogramas sin detección;
  - cajas falsas F: tracklets que permanecen en su lugar en al menos el 95 % de sus fotogramas, en un lugar
    donde se repite otra caja así a menos de 20 px;
  - distancia vertical del punto de contacto al borde superior del crucero (prof);
  - estado del tránsito por segundo: proporción de vehículos en movimiento en el área de reconocimiento,
    antes del crucero y fuera del carril exclusivo, suavizada con una media móvil de 5 s;
  - otra detención estricta simultánea hasta 1,5 alturas por delante (detenido_delante).
Además, para cada automóvil detenido, cuenta las personas cuyo tracklet empieza o termina junto a él.

Salidas (salidas/candidatos/): <video_id>_detenciones.csv, con una fila por detención estricta, y
<video_id>_<infracción>.csv, con las unidades que cumplen la condición principal de cada regla (se usan para
elegir los casos de control de la validación por experto).
Uso: python src/candidatos.py [--videos ...]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((RAIZ / P["rutas"]["procesamiento"]).resolve() / "src"))
import configuracion, indice, zonas  # noqa: E402

C = P["candidatos"]


def suavizar(t):
    t = t.sort_values("frame").copy()
    t["px"], t["py"] = (t.x1 + t.x2) / 2, t.y2
    t["h"] = (t.y2 - t.y1).clip(lower=1)
    t["sx"] = t.px.rolling(C["ventana_contacto"], center=True, min_periods=1).median()
    t["sy"] = t.py.rolling(C["ventana_contacto"], center=True, min_periods=1).median()
    cx, cy = t.px, (t.y1 + t.y2) / 2
    t["vel"] = (np.hypot(cx.diff(), cy.diff()) / t.frame.diff() / t.h).rolling(C["ventana_vel"], center=True, min_periods=3).median()
    return t


def detenciones(t, fps):
    """Intervalos [k0, k1] de detención estricta (índices de cuadro)."""
    f, x, y, h = t.frame.values, t.sx.values, t.sy.values, t.h.values
    out, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and f[j + 1] - f[j] <= C["max_hueco"] + 1 and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= C["radio_D"] * h[i]:
            j += 1
        if (f[j] - f[i] + 1) / fps >= C["duracion_S"] and j - i + 1 >= C["min_cuadros_obs"]:
            out.append((int(f[i]), int(f[j])))
            i = j + 1
        else:
            i += 1
    return out


def borde_crucero(cruce, alto):
    """y del borde superior del crucero para cada columna x (np.inf donde no hay crucero)."""
    m = np.zeros((alto, cruce[:, 0].max() + 2), np.uint8)
    cv2.fillPoly(m, [cruce], 1)
    top = np.full(m.shape[1], np.inf)
    cols = np.where(m.any(0))[0]
    top[cols] = m[:, cols].argmax(0)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    a = ap.parse_args()
    p = configuracion.cargar()
    ind = indice.leer(p)
    z = zonas.cargar(p["rutas"]["zonas"])
    carril, cruce, area = z[C["zona_carril"]], z[C["zona_crucero"]], z[p["zonas"]["area"]]
    top = borde_crucero(cruce, p["video"]["resolucion"][1])
    xmin, xmax = int(cruce[:, 0].min()), int(cruce[:, 0].max())
    sal = RAIZ / P["rutas"]["salidas"] / "candidatos"
    sal.mkdir(parents=True, exist_ok=True)
    cons = RAIZ / P["rutas"]["salidas"] / "consolidado"
    vids = a.videos or sorted(pd.read_csv(cons / "cobertura.csv").video_id.unique())
    for vid in vids:
        f = ind.loc[vid]
        fps = float(f.fps)
        c = p["rutas"]["salidas"] / f.sesion / vid
        tr = pd.read_csv(c / f"{vid}_trayectorias_vehiculos.csv.gz")
        per = pd.read_csv(c / f"{vid}_trayectorias_personas.csv.gz")
        tr = pd.concat([suavizar(t) for _, t in tr.groupby("track_id")], ignore_index=True)
        tr["en_area"] = zonas.dentro(tr.px, tr.py, area)
        tr["en_carril"] = zonas.dentro(tr.px, tr.py, carril)
        xi = tr.sx.clip(xmin, xmax).round().astype(int).values
        tr["prof"] = np.where((tr.sx >= xmin) & (tr.sx <= xmax), tr.sy.values - top[xi], np.nan)
        clase = tr.groupby("track_id").clase.agg(lambda s: (s == 5).mean())
        es_bus = clase >= C["max_frac_bus"]
        # cajas falsas: casi inmóviles en un sitio donde se repiten otras así
        g = tr.groupby("track_id")
        med = g[["px", "py"]].median()
        quieto = tr.join(med, on="track_id", rsuffix="_m")
        quieto = (np.hypot(quieto.px - quieto.px_m, quieto.py - quieto.py_m) <= C["radio_D"] * quieto.h).groupby(quieto.track_id).mean()
        q_ids = quieto[quieto >= C["frac_quieto_falsa"]].index
        F = set()
        for tid in q_ids:
            d = np.hypot(med.loc[q_ids].px - med.loc[tid].px, med.loc[q_ids].py - med.loc[tid].py)
            if (d <= C["radio_sitio_falsa_px"]).sum() - 1 >= C["min_repeticiones_falsa"]:
                F.add(tid)
        # estado del tránsito por segundo
        ref = tr[tr.en_area & ~tr.en_carril & (tr.prof.isna() | (tr.prof < 0))].copy()
        ref["seg"] = (ref.frame / fps).astype(int)
        ref["mov"] = ref.vel >= C["umbral_vel"]
        tran = ref.groupby(["seg", "track_id"]).mov.mean().gt(0.5).groupby("seg").mean()
        tran = tran.reindex(range(int(f.cuadros / fps) + 1)).interpolate().rolling(C["ventana_transito_s"], center=True, min_periods=1).mean()
        # detenciones estrictas de todos los tracklets
        filas = []
        for tid, t in tr.groupby("track_id"):
            for k0, k1 in detenciones(t, fps):
                w = t[(t.frame >= k0) & (t.frame <= k1)]
                if w.en_area.sum() < C["min_cuadros_obs"]:
                    continue
                s0, s1 = int(k0 / fps), int(k1 / fps)
                filas.append({"track_id": tid, "k_ini": k0, "k_fin": k1, "dur_s": round((k1 - k0 + 1) / fps, 1),
                              "prof_mediana": float(np.nanmedian(w.prof)) if w.prof.notna().any() else np.nan,
                              "frac_carril": round(w.en_carril.mean(), 2), "transito": round(float(tran.loc[s0:s1].median()), 3),
                              "bus": bool(es_bus.get(tid, False)), "falsa": tid in F,
                              "x": float(w.sx.median()), "y": float(w.sy.median()), "h": float(w.h.median())})
        det = pd.DataFrame(filas)
        # ¿otro vehículo detenido delante, en su carril? (delante = más cerca de la cámara: y mayor)
        delante = []
        for r in det.itertuples():
            o = det[(det.track_id != r.track_id) & (det.k_ini <= r.k_fin) & (det.k_fin >= r.k_ini) & ~det.falsa]
            o = o[(o.y > r.y) & (o.y - r.y <= C["delante_altos"] * r.h) & (abs(o.x - r.x) <= C["delante_lateral_altos"] * r.h)]
            delante.append(len(o) > 0)
        det["detenido_delante"] = delante
        det.to_csv(sal / f"{vid}_detenciones.csv", index=False)
        g41 = det[(det.prof_mediana > 0) & ~det.falsa]
        m21 = det[~det.bus & ~det.falsa & (det.transito >= C["theta_transito"]) & ~det.detenido_delante]
        # G08: personas que aparecen o desaparecen junto a un auto detenido
        g08 = []
        autos = det[~det.bus & ~det.falsa]
        p_ini = per.sort_values("frame").groupby("track_id").head(1)
        p_fin = per.sort_values("frame").groupby("track_id").tail(1)
        for r in autos.itertuples():
            v = tr[(tr.track_id == r.track_id) & (tr.frame >= r.k_ini) & (tr.frame <= r.k_fin)]
            if v.empty:
                continue
            b = v[["x1", "y1", "x2", "y2"]].median()
            m = C["margen_persona_altos"] * (b.y2 - b.y1)
            x1, y1, x2, y2 = b.x1 - m, b.y1 - m, b.x2 + m, b.y2 + m
            n_ev = 0
            for q in (p_ini, p_fin):
                q = q[(q.frame >= r.k_ini - C["holgura_persona"]) & (q.frame <= r.k_fin + C["holgura_persona"])]
                pies = ((q.x1 + q.x2) / 2 >= x1) & ((q.x1 + q.x2) / 2 <= x2) & (q.y2 >= y1) & (q.y2 <= y2)
                n_ev += int(pies.sum())
            if n_ev:
                g08.append({**r._asdict(), "personas_aparecen_o_desaparecen": n_ev})
        g08 = pd.DataFrame(g08).drop(columns=["Index"], errors="ignore")
        g10 = tr[tr.en_carril].groupby("track_id").agg(cuadros=("frame", "size"), k_ini=("frame", "min"), k_fin=("frame", "max"))
        g10 = g10[(g10.cuadros >= C["min_cuadros_carril"]) & ~g10.index.isin(es_bus[es_bus].index)].reset_index()
        for nombre, x in (("G41", g41), ("M21", m21), ("G08", g08), ("G10", g10)):
            x.to_csv(sal / f"{vid}_{nombre}.csv", index=False)
        print(f"{vid}: detenciones {len(det)}, falsas {len(F)}, G41 {len(g41)}, M21 {len(m21)}, G08 {len(g08)}, G10 {len(g10)}")


if __name__ == "__main__":
    main()
