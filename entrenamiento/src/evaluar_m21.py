"""Regla de M21 y su evaluación (Etapas 4 y 5).

Sobre las detenciones estrictas de vehículos que no son bus ni caja falsa, la regla exige que:
  1. el tránsito circundante esté en movimiento: mediana de tránsito(t) durante la detención >= θ;
  2. no haya otra detención estricta simultánea hasta 1,5 alturas por delante (etiquetado/src/candidatos.py);
  3. la detención dure al menos T, el umbral de Otsu del logaritmo de la duración de las detenciones que cumplen
     1 y 2;
  4. el vehículo esté fuera de la intersección;
  5. no haya un vehículo detenido delante en su carril, entre 0,3 y 5 alturas (carriles trazados);
  6. al menos la mitad de los vehículos de los carriles contiguos, a menos de 3 alturas, esté en movimiento;
  7. el vehículo no cambie de rumbo 45° o más con el frente a menos de 150 px del crucero.
Ningún umbral usa etiquetas.

Evaluación: la precisión tiene como referencia la revisión visual de todas las detenciones que señala la regla;
el recall tiene como referencia la anotación: proporción de los eventos anotados que coinciden con una detención
señalada del mismo vehículo. Como las dos medidas usan referencias distintas, no se combinan en un F1. Los
intervalos de confianza son de Wilson al 95 %. Cada caso revisado es un clip con el vehículo marcado
(etiquetado/src/exportar_clips.py), en orden aleatorio y con nombres neutros; el revisor responde si el vehículo
cumple la definición: si, no o dudoso.

Paso 1  python src/evaluar_m21.py --muestra    -> clips y hoja revision_m21.csv en rutas.verificacion_m21
Paso 2  python src/evaluar_m21.py              -> rutas.resultados/m21/metricas_m21.json y .md
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import comun as c
import iluminacion as il

M = c.P["m21"]
CP = c.PE["candidatos"]


def detenciones():
    filas = []
    for vid in c.videos():
        d = pd.read_csv(c.ETQ / "salidas" / "candidatos" / f"{vid}_detenciones.csv").assign(video_id=vid)
        pj = pd.read_csv(c.ETQ / "salidas" / "candidatos" / f"{vid}_G08.csv",
                         usecols=["track_id", "k_ini", "personas_aparecen_o_desaparecen"])
        d = d.merge(pj, on=["track_id", "k_ini"], how="left")
        filas.append(d[~d.bus & ~d.falsa])
    d = pd.concat(filas, ignore_index=True)
    d["pasajeros"] = d.personas_aparecen_o_desaparecen.fillna(0) > 0
    d["mov"] = (d.transito >= CP["theta_transito"]) & ~d.detenido_delante
    d["larga"] = d.mov & (d.dur_s >= duracion_minima(d))
    d = d.merge(forzadas(d[d.larga]), on=["video_id", "track_id", "k_ini"], how="left")
    for col in ["cola", "giro_cruce", "interseccion"]:
        d[col] = d[col].fillna(False).astype(bool)
    d["vecinos_parados"] = (d.vecinos.fillna(0) > 0) & (d.vecinos_en_marcha < M["vecinos_min_en_marcha"])
    d["carril"] = d.carril.fillna(0).astype(int)
    d["regla"] = d.larga & (d.carril > 0) & ~d.cola & ~d.giro_cruce & ~d.interseccion & ~d.vecinos_parados
    ilu = pd.read_csv(c.R["dataset"] / "iluminacion.csv")
    per = []
    for r in d.itertuples():
        g = ilu[ilu.video_id == r.video_id]
        b = int(r.k_ini / c.fps(r.video_id) / 60 // c.P["dataset"]["bloque_min"])
        per.append(g.periodo.iloc[int(np.argmin(np.abs(g.bloque.to_numpy() - b)))])
    d["periodo"] = per
    return d


def duracion_minima(d):
    return float(np.exp(il.otsu(np.log(d.dur_s[d.mov].to_numpy()))))


def forzadas(x):
    """Condiciones de cada detención larga (se guardan en resultados/m21/forzadas.csv).
    Carriles trazados a mano sobre la imagen (rutas.carriles, polígonos carril_1 a carril_4, de izquierda a
    derecha; el tránsito avanza hacia abajo). El carril de un vehículo es el polígono que contiene su punto de apoyo.
      cola         otro vehículo del mismo carril, entre 0,3 y `carril_largo_altos` altos más adelante, casi quieto
                   en la primera mitad de la detención;
      vecinos      vehículos de los carriles adyacentes a menos de `vecinos_altos` altos;
      giro_cruce   el rumbo cambia `giro_grados` o más junto al crucero (cede el paso para girar);
      interseccion el punto de apoyo está más allá del borde inferior del crucero (manda la preferencia de paso)."""
    ruta = c.salida("m21") / "forzadas.csv"
    if ruta.exists():
        f = pd.read_csv(ruta)
        if "carril" in f and len(f.merge(x[["video_id", "track_id", "k_ini"]])) == len(x):
            return f
    sys.path.insert(0, str(c.ETQ / "src"))
    import candidatos as K
    import cv2
    pr = K.configuracion.cargar(); z = K.zonas.cargar(pr["rutas"]["zonas"]); cr = z[CP["zona_crucero"]]
    carriles = [p for _, p in sorted(K.zonas.cargar(c.R["carriles"]).items())]
    W, H = pr["video"]["resolucion"]
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [cr.astype(np.int32)], 1)
    cols = np.where(m.any(0))[0]; borde = np.full(W, -1.0)
    for cc in cols:
        borde[cc] = np.where(m[:, cc])[0].max()
    borde[:cols.min()] = borde[cols.min()]; borde[cols.max() + 1:] = borde[cols.max()]
    mapa = np.zeros((H, W), np.uint8)            # número de carril de cada píxel (0: fuera de los carriles)
    for i, p in enumerate(carriles, 1):
        cv2.fillPoly(mapa, [p.astype(np.int32)], i)
    carril = lambda xs, ys: mapa[np.clip(np.round(ys).astype(int), 0, H - 1), np.clip(np.round(xs).astype(int), 0, W - 1)]
    largo = M["carril_largo_altos"]
    filas = []
    for vid, g in x.groupby("video_id"):
        fps = c.fps(vid)
        tr = c.leer_csv(vid, "trayectorias_vehiculos.csv.gz")
        tr = pd.concat([K.suavizar(t) for _, t in tr.groupby("track_id")], ignore_index=True).sort_values(["track_id", "frame"])
        pista = {t: q.reset_index(drop=True) for t, q in tr.groupby("track_id")}
        trf = tr.sort_values("frame").reset_index(drop=True); fr = trf.frame.to_numpy()
        for r in g.itertuples():
            a = pista[r.track_id]
            pre = a[(a.frame >= r.k_ini - M["rumbo_s"] * fps) & (a.frame < r.k_ini)]
            post = a[(a.frame > r.k_fin) & (a.frame <= r.k_fin + M["rumbo_s"] * fps)]
            rumbo = lambda q: np.array([q.sx.iloc[-1] - q.sx.iloc[0], q.sy.iloc[-1] - q.sy.iloc[0]])
            giro = np.nan
            if len(pre) >= 5 and len(post) >= 5 and min(np.linalg.norm(rumbo(pre)), np.linalg.norm(rumbo(post))) > 1e-6:
                u, v = rumbo(pre), rumbo(post)
                giro = float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
            lr = int(carril(np.array([r.x]), np.array([r.y]))[0])
            cola, vec = False, []
            for kk in np.linspace(r.k_ini, (r.k_ini + r.k_fin) / 2, 4):
                w = trf.iloc[np.searchsorted(fr, kk - 3):np.searchsorted(fr, kk + 3, "right")]
                w = w[w.track_id != r.track_id]
                if not len(w) or lr == 0:
                    continue
                lw = carril(w.sx.to_numpy(), w.sy.to_numpy())
                dist = np.hypot(w.sx - r.x, w.sy - r.y).to_numpy() / r.h
                ade = (w.sy - r.y).to_numpy() / r.h
                dl = w[(lw == lr) & (ade > 0.3) & (dist <= largo)]
                cola |= bool(len(dl) and (dl.groupby("track_id").vel.median() < CP["umbral_vel"]).any())
                v = w[(lw > 0) & (np.abs(lw.astype(int) - lr) == 1) & (dist <= M["vecinos_altos"])]
                vec += (v.groupby("track_id").vel.median() >= CP["umbral_vel"]).tolist()
            cruce = r.prof_mediana > -M["cruce_px"] if r.prof_mediana == r.prof_mediana else False
            filas.append({"video_id": vid, "track_id": r.track_id, "k_ini": r.k_ini, "carril": lr, "cola": cola,
                          "giro": giro, "giro_cruce": bool(giro >= M["giro_grados"] and cruce),
                          "interseccion": bool(r.y > borde[int(np.clip(round(r.x), 0, W - 1))]),
                          "vecinos": len(vec), "vecinos_en_marcha": float(np.mean(vec)) if vec else np.nan})
    f = pd.DataFrame(filas)
    f.to_csv(ruta, index=False)
    return f


def solapa(df, r):
    return (df.video_id == r.video_id) & (df.track_id == r.track_id) & (df.k_ini <= r.k_fin) & (df.k_fin >= r.k_ini)


def armar_muestra(d):
    a = d[d.regla][["video_id", "track_id", "k_ini", "k_fin"]].assign(fuente="regla")   # todas las señaladas
    an = c.etiquetas("M21")[["video_id", "track_id", "k_ini", "k_fin"]].assign(fuente="anotacion")
    # una anotación se revisa sobre la detención que más se le superpone (si no hay, su propio intervalo)
    for i, r in an.iterrows():
        m = d[solapa(d, r)]
        if len(m):
            s = m.iloc[int(np.argmax(np.minimum(m.k_fin, r.k_fin) - np.maximum(m.k_ini, r.k_ini)))]
            an.loc[i, ["k_ini", "k_fin"]] = [s.k_ini, s.k_fin]
    x = pd.concat([a, an], ignore_index=True)
    # un mismo caso que sale de las dos fuentes se revisa una vez
    x = x.groupby(["video_id", "track_id", "k_ini", "k_fin"], as_index=False).fuente.agg("+".join)
    return x   # el orden aleatorio y los nombres neutros los pone exportar_clips


def respuestas(ruta):
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    return pd.read_excel(ruta, dtype=str)


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción k/n."""
    if n == 0:
        return None
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    radio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centro - radio), 4), round(float(centro + radio), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", action="store_true")
    ap.add_argument("--respuestas", default=str(c.R["verificacion_m21"] / "revision_m21.csv"))
    a = ap.parse_args()
    out = c.salida("m21")
    d = detenciones()
    if a.muestra:
        x = armar_muestra(d)
        ent = c.R["verificacion_m21"]
        sys.path.insert(0, str(c.ETQ / "src"))
        import exportar_clips as ec
        planes = [{"video_id": r.video_id, "track_id": int(r.track_id), "a": int(r.k_ini), "b": int(r.k_fin),
                   "origen": {"caso"}, "infracciones": set(), "fila": i} for i, r in enumerate(x.itertuples())]
        vidas = {}
        for vid in x.video_id.unique():
            t = c.leer_csv(vid, "trayectorias_vehiculos.csv.gz", usecols=["frame", "track_id"])
            vidas.update({(vid, k): (int(g.frame.min()), int(g.frame.max())) for k, g in t.groupby("track_id")})
        ec.escribir_clips(planes, vidas, ent, ec.configuracion.cargar(), c.IND, random.Random(c.P["validacion"]["semilla"]))
        x["caso"] = ""
        for pl in planes:
            x.loc[pl["fila"], ["caso", "c0", "c1"]] = [pl["clip"], pl["c0"], pl["c1"]]
        x = x.sort_values("caso").reset_index(drop=True)
        x.to_csv(out / "clave.csv", index=False)
        f = x[["caso"]].assign(m21="", sube_o_baja_alguien="", nota="")
        f.to_csv(ent / "revision_m21.csv", index=False, sep=";", encoding="utf-8-sig")
        print(f"{len(x)} casos ({(x.fuente.str.contains('regla')).sum()} de la regla, "
              f"{(x.fuente.str.contains('anotacion')).sum()} anotados) -> {ent}")
        return

    clave = pd.read_csv(out / "clave.csv")
    r = respuestas(a.respuestas)
    r["m21"] = r.m21.astype(str).str.strip().str.lower().str[:2]
    x = clave.merge(r[["caso", "m21"]], on="caso", how="left")
    faltan = x.m21.isna() | ~x.m21.isin(["si", "sí", "no", "du"])
    if faltan.any():
        raise SystemExit(f"Casos sin respuesta válida (si, no o dudoso): {list(x.caso[faltan])}")
    x["si"] = x.m21.isin(["si", "sí"])
    x["decidido"] = x.m21 != "du"
    x["regla"] = [bool((d.regla & solapa(d, r_)).any()) for r_ in x.itertuples()]
    x = x.merge(d[["video_id", "track_id", "k_ini", "k_fin", "periodo", "transito"]], how="left",
                on=["video_id", "track_id", "k_ini", "k_fin"])
    x["vehiculo"] = x.video_id + "_" + x.track_id.astype(str)
    x.to_csv(out / "revision_cruzada.csv", index=False)

    res = {"regla": "detención estricta, no bus, fuera de F, tránsito >= θ, duración >= T, fuera de la intersección, sin cola en su carril, con los carriles adyacentes en marcha y sin giro junto al crucero",
           "excluidas_por_cola": int((d.larga & d.cola).sum()),
           "excluidas_por_interseccion": int((d.larga & ~d.cola & d.interseccion).sum()),
           "excluidas_por_vecinos_detenidos": int((d.larga & ~d.cola & ~d.interseccion & d.vecinos_parados).sum()),
           "excluidas_por_giro": int((d.larga & ~d.cola & ~d.interseccion & ~d.vecinos_parados & d.giro_cruce).sum()),
           "theta": CP["theta_transito"], "T_s": round(duracion_minima(d), 2),
           "senaladas_con_pasajeros": int((d.regla & d.pasajeros).sum()), "detenciones": int(len(d)), "senaladas_por_la_regla": int(d.regla.sum()),
           "senaladas_por_hora": round(d.regla.sum() / c.horas_video(d.video_id), 1),
           "dudosos": int((~x.decidido).sum()), "por_periodo": {}}
    # precisión: detenciones señaladas, con la revisión visual como referencia
    for nombre, sel in (("global", np.ones(len(x), bool)), ("dia", (x.periodo == "dia").to_numpy()),
                        ("noche", (x.periodo == "noche").to_numpy())):
        p = x[sel & x.fuente.str.contains("regla") & x.decidido]
        res["por_periodo"][nombre] = {"senaladas_decididas": int(len(p)), "confirmadas": int(p.si.sum()),
                                      "precision": float(p.si.mean()) if len(p) else None,
                                      "ic95_precision": wilson(int(p.si.sum()), len(p))}
    # recall: eventos anotados, con la anotación como referencia
    an = c.etiquetas("M21")
    encontrados = [bool((d.regla & solapa(d, r_)).any()) for r_ in an.itertuples()]
    k, n = int(sum(encontrados)), len(an)
    res["recall"] = {"eventos_anotados": n, "encontrados": k, "recall": k / n if n else None, "ic95_recall": wilson(k, n)}
    # detenciones confirmadas que coinciden con un evento anotado de la infracción M21 o de la infracción G08
    conf = x[x.fuente.str.contains("regla") & x.si]
    g08 = c.etiquetas("G08")
    res["confirmadas_con_anotacion_m21"] = int(sum(bool(solapa(an, r_).any()) for r_ in conf.itertuples()))
    res["confirmadas_con_evento_g08"] = int(sum(bool(solapa(g08, r_).any()) and not bool(solapa(an, r_).any())
                                                for r_ in conf.itertuples()))
    res["confirmadas_con_pasajeros"] = int(sum(bool((d.pasajeros & solapa(d, r_)).any()) for r_ in conf.itertuples()))
    c.guardar_json(res, out / "metricas_m21.json")
    filas = [{"periodo": k, **{kk: vv for kk, vv in v.items()}} for k, v in res["por_periodo"].items()]
    txt = ["# Evaluación de la regla M21", "",
           f"Regla: {res['regla']} (θ = {res['theta']}, T = {res['T_s']} s). Señala {res['senaladas_por_la_regla']} de "
           f"{res['detenciones']} detenciones ({res['senaladas_por_hora']} por hora).", "",
           "Precisión: detenciones señaladas por la regla, con la revisión visual como referencia. Recall: eventos "
           "anotados que coinciden con una detención señalada, con la anotación como referencia. Intervalos de Wilson.", "",
           c.markdown(pd.DataFrame(filas)), "",
           f"Recall: {res['recall']['encontrados']} de {res['recall']['eventos_anotados']} eventos anotados "
           f"= {res['recall']['recall']:.2f} (IC 95 %: {res['recall']['ic95_recall']}).",
           f"Casos sin decisión (fuera de la precisión): {res['dudosos']}.",
           f"Confirmadas: {res['confirmadas_con_anotacion_m21']} con una anotación de M21, "
           f"{res['confirmadas_con_evento_g08']} con un evento de G08 y {res['confirmadas_con_pasajeros']} con personas que suben o bajan."]
    (out / "metricas_m21.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
