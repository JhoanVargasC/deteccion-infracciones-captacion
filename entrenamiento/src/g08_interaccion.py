"""Matriz de interacción de G08: variables por instante y su resumen por ventana.
Ninguna variable usa la posición del automóvil en la imagen.

Por instante f de un automóvil con caja b y altura h (J_f = personas próximas: >= 50 % de su caja dentro de
b ampliada 0,5 h):
  1 |J_f|                    2 de J_f a la izquierda del centro    3 a la derecha
  4 de J_f en moto/bicicleta (>= 30 % de su caja sobre una moto o bicicleta en >= 30 % de su trayectoria)
  5 de J_f con los pies en el crucero               6 velocidad de la persona próxima más lenta (NaN si J_f vacío)
  7 personas cuya caja interseca el recorte
  8-9   apariciones y desapariciones asignadas al automóvil en f-2..f+2
  10-11 las mismas sin motociclistas ni ciclistas
  12-13 las mismas de trayectorias >= 1 s que empiezan o terminan a >= 15 px del borde del área
  14 velocidad suavizada     15 1[vel < 0,02]     16 vel(f) - vel(f - 1 s)
  17 segundos continuos con vel < 0,02 hasta f      18 fracción de los demás automóviles del mismo segundo con vel < 0,02
  19-20 apariciones y desapariciones netas: sin las que forman un fragmento, es decir, una desaparición
        seguida de una aparición junto al mismo automóvil (o una aparición precedida de una desaparición) en
        menos de `fragmento_persona_s`. En la vereda y el crucero el seguidor pierde peatones y los vuelve a
        crear con otro número, lo que imita una subida.
El primer y el último cuadro de cada trayectoria de persona se asignan al automóvil más cercano a sus pies
(distancia del punto a la caja), si la persona es próxima a él en ese cuadro.
Resumen por ventana (Tabla D.2; 22 columnas, porque la variable 1 genera tres y las demás una; las dos
últimas son las variables netas 19-20): 1 -> máximo, media y cambio (media de los dos últimos instantes
válidos menos la de los dos primeros); 2-5, 7, 17 -> máximo; 6 -> mínimo; 8-13 -> suma; 14-16, 18 -> media.

Entradas: instantes.csv.gz y ventanas.csv.gz (g08_ventanas.py), trayectorias y detecciones del procesamiento.
Salida (rutas.rasgos/g08): interaccion_instantes.npy (float32, fila = id_instante),
interaccion_ventanas.npy (float32, fila = w_id) e interaccion_columnas.json
Uso: python src/g08_interaccion.py
"""
import cv2
import numpy as np
import pandas as pd

import comun as c

G = c.P["g08"]
VARS = ["n_prox", "n_prox_izq", "n_prox_der", "n_prox_rodado", "n_prox_crucero", "vel_persona_min", "n_interseca_recorte",
        "apariciones", "desapariciones", "apariciones_sin_rodado", "desapariciones_sin_rodado",
        "apariciones_filtradas", "desapariciones_filtradas",
        "vel", "detenido", "dvel_1s", "t_detenido_s", "frac_otros_detenidos",
        "apariciones_netas", "desapariciones_netas"]
COLUMNAS = ["n_prox_max", "n_prox_media", "n_prox_cambio", "n_prox_izq_max", "n_prox_der_max", "n_prox_rodado_max",
            "n_prox_crucero_max", "vel_persona_min_min", "n_interseca_recorte_max",
            "apariciones_suma", "desapariciones_suma", "apariciones_sin_rodado_suma", "desapariciones_sin_rodado_suma",
            "apariciones_filtradas_suma", "desapariciones_filtradas_suma",
            "vel_media", "detenido_media", "dvel_1s_media", "t_detenido_s_max", "frac_otros_detenidos_media",
            "apariciones_netas_suma", "desapariciones_netas_suma"]


def velocidad(t):
    """Misma definición que el procesamiento (recortar.con_velocidad): paso del centro por cuadro en altos."""
    t = t.sort_values(["track_id", "frame"]).reset_index(drop=True)
    cx, cy, h = (t.x1 + t.x2) / 2, (t.y1 + t.y2) / 2, (t.y2 - t.y1).clip(lower=1)
    paso = np.hypot(cx.diff(), cy.diff()) / t.frame.diff() / h
    paso[t.track_id != t.track_id.shift()] = np.nan
    r = c.PP["recortes"]
    t["vel"] = paso.groupby(t.track_id).transform(
        lambda s: s.rolling(r["vel_ventana_filas"], center=True, min_periods=r["vel_min_filas"]).median())
    return t


def ioa(a, b):
    """Fracción de cada caja de `a` (n×4) cubierta por cada caja de `b` (m×4): n×m."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return ix * iy / np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)[:, None]


def ampliada(caja):
    m = G["margen_proximidad_altos"] * (caja[..., 3] - caja[..., 1])
    return np.stack([caja[..., 0] - m, caja[..., 1] - m, caja[..., 2] + m, caja[..., 3] + m], -1)


def rodados(per, vid):
    """Personas que van en moto o bicicleta (variable 4)."""
    det = c.leer_csv(vid, "detecciones.csv.gz", usecols=["frame", "clase", "conf", "x1", "y1", "x2", "y2"])
    det = det[det.clase.isin(G["clases_rodado"]) & (det.conf >= G["conf_rodado"])]
    rod = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in det.groupby("frame")}
    sobre = np.zeros(len(per), bool)
    cajas = per[["x1", "y1", "x2", "y2"]].to_numpy()
    for i, f in enumerate(per.frame.to_numpy()):
        b = rod.get(f)
        if b is not None:
            sobre[i] = (ioa(cajas[i:i + 1], b) >= G["ioa_rodado"]).any()
    frac = pd.Series(sobre).groupby(per.track_id.to_numpy()).mean()
    return set(frac.index[frac >= G["frac_rodado"]])


def instantes_video(vid, ins, ventana_s):
    fps = c.fps(vid)
    per = velocidad(c.leer_csv(vid, "trayectorias_personas.csv.gz"))
    veh = c.leer_csv(vid, "trayectorias_vehiculos.csv.gz")
    veh = veh[veh.track_id.isin(ins.track_id.unique())]          # solo los automóviles del conjunto
    moto = rodados(per, vid)
    crucero = c.ZONAS[c.PE["candidatos"]["zona_crucero"]]
    area = c.ZONAS[c.PP["zonas"]["area"]]
    per["rodado"] = per.track_id.isin(moto)
    per["en_crucero"] = c._zonas.dentro((per.x1 + per.x2) / 2, per.y2, crucero)
    pf = {f: g for f, g in per.groupby("frame")}
    n = len(ins)
    X = np.zeros((n, len(VARS)), np.float32)
    X[:, VARS.index("vel_persona_min")] = np.nan
    caja = ins[["x1", "y1", "x2", "y2"]].to_numpy()
    rec = ins[["cx1", "cy1", "cx2", "cy2"]].to_numpy()
    ex = ampliada(caja)
    for i, f in enumerate(ins.frame.to_numpy()):
        g = pf.get(f)
        if g is None:
            continue
        cp = g[["x1", "y1", "x2", "y2"]].to_numpy()
        X[i, 6] = ((np.minimum(cp[:, 2], rec[i, 2]) > np.maximum(cp[:, 0], rec[i, 0])) &
                   (np.minimum(cp[:, 3], rec[i, 3]) > np.maximum(cp[:, 1], rec[i, 1]))).sum()
        j = ioa(cp, ex[i:i + 1])[:, 0] >= G["ioa_proximidad"]
        if not j.any():
            continue
        gj = g[j]
        cx = (gj.x1.to_numpy() + gj.x2.to_numpy()) / 2
        centro = (caja[i, 0] + caja[i, 2]) / 2
        X[i, 0], X[i, 1], X[i, 2] = j.sum(), (cx < centro).sum(), (cx >= centro).sum()
        X[i, 3], X[i, 4] = gj.rodado.sum(), gj.en_crucero.sum()
        v = gj.vel.to_numpy()
        X[i, 5] = np.nanmin(v) if np.isfinite(v).any() else np.nan

    # extremos de trayectoria de persona -> automóvil más cercano a los pies, si la persona le es próxima
    g = per.groupby("track_id")
    ext = pd.concat([g.head(1).assign(tipo="aparece"), g.tail(1).assign(tipo="desaparece")], ignore_index=True)
    dur = (g.frame.max() - g.frame.min() + 1) / fps
    ext["dur_s"] = ext.track_id.map(dur)
    ext["borde_px"] = [abs(cv2.pointPolygonTest(area, (float((a + b) / 2), float(y)), True))
                       for a, b, y in zip(ext.x1, ext.x2, ext.y2)]
    vf = {f: x for f, x in veh.groupby("frame")}
    asig = []
    for r in ext.itertuples():
        x = vf.get(r.frame)
        if x is None:
            asig.append(-1)
            continue
        b = x[["x1", "y1", "x2", "y2"]].to_numpy()
        px, py = (r.x1 + r.x2) / 2, r.y2
        d = np.hypot(np.clip(px, b[:, 0], b[:, 2]) - px, np.clip(py, b[:, 1], b[:, 3]) - py)
        k = int(np.argmin(d))
        prox = ioa(np.array([[r.x1, r.y1, r.x2, r.y2]]), ampliada(b[k:k + 1]))[0, 0] >= G["ioa_proximidad"]
        asig.append(int(x.track_id.iloc[k]) if prox else -1)
    ext["auto"] = asig
    ext = ext[ext.auto >= 0]
    ext["filtrada"] = (ext.dur_s >= G["min_duracion_persona_s"]) & (ext.borde_px >= G["min_borde_area_px"])
    # fragmentos: desaparición y aparición del extremo opuesto junto al mismo automóvil en menos de T
    T = round(G["fragmento_persona_s"] * fps)
    ext["fragmento"] = False
    for t, e in ext.groupby("auto"):
        ap_ = np.sort(e.frame[e.tipo == "aparece"].to_numpy())
        de_ = np.sort(e.frame[e.tipo == "desaparece"].to_numpy())
        frag = []
        for f_, tp in zip(e.frame.to_numpy(), e.tipo.to_numpy()):
            if tp == "desaparece":
                j = np.searchsorted(ap_, f_, "right")
                frag.append(j < len(ap_) and ap_[j] - f_ <= T)
            else:
                j = np.searchsorted(de_, f_, "left") - 1
                frag.append(j >= 0 and f_ - de_[j] <= T)
        ext.loc[e.index, "fragmento"] = frag
    w = G["ventana_extremos_cuadros"]
    pos = {t: np.nonzero(ins.track_id.to_numpy() == t)[0] for t in ins.track_id.unique()}
    fr = ins.frame.to_numpy()
    for t, filas in pos.items():
        e = ext[ext.auto == t]
        if e.empty:
            continue
        for col, sel in ((7, e.tipo == "aparece"), (8, e.tipo == "desaparece"),
                         (9, (e.tipo == "aparece") & ~e.rodado), (10, (e.tipo == "desaparece") & ~e.rodado),
                         (11, (e.tipo == "aparece") & e.filtrada), (12, (e.tipo == "desaparece") & e.filtrada),
                         (18, (e.tipo == "aparece") & ~e.fragmento), (19, (e.tipo == "desaparece") & ~e.fragmento)):
            fe = np.sort(e.frame.to_numpy()[sel.to_numpy()])
            if len(fe):
                X[filas, col] = np.searchsorted(fe, fr[filas] + w, "right") - np.searchsorted(fe, fr[filas] - w, "left")

    # cinemática del automóvil (los instantes están cada 5 cuadros desde su primer cuadro en el área)
    umbral = G["umbral_detenido"]
    vel = ins.vel.to_numpy(float)
    X[:, 13] = vel
    X[:, 14] = vel < umbral                      # NaN cuenta como en marcha
    clave = pd.Series(vel, index=pd.MultiIndex.from_arrays([ins.track_id.to_numpy(), fr]))
    atras = pd.MultiIndex.from_arrays([ins.track_id.to_numpy(), fr - int(round(G["dvel_s"] * fps))])
    X[:, 15] = vel - clave.reindex(atras).to_numpy()
    paso_s = c.PP["recortes"]["cada_k_cuadros"] / fps
    det = pd.Series(vel < umbral).astype(int)
    corrida = det.groupby([ins.track_id.to_numpy(), (det != det.shift()).cumsum().to_numpy()]).cumsum()
    X[:, 16] = corrida.to_numpy() * paso_s
    seg = (fr / fps).astype(int)
    s = pd.DataFrame({"seg": seg, "d": (vel < umbral).astype(float)}).groupby("seg").d.agg(["sum", "count"])
    sm, ct = s["sum"].reindex(seg).to_numpy(), s["count"].reindex(seg).to_numpy()
    X[:, 17] = np.where(ct > 1, (sm - X[:, 14]) / np.maximum(ct - 1, 1), np.nan)
    return X, {"video_id": vid, "personas_en_rodado": len(moto), "extremos_asignados": len(ext)}


def resumen_ventanas(Xi, w):
    """20 columnas por ventana."""
    W = np.full((len(w), len(COLUMNAS)), np.nan, np.float32)
    for k, (i0, i1) in enumerate(zip(w.i0.to_numpy(), w.i1.to_numpy())):
        x = Xi[i0:i1]
        n = x[:, 0]
        W[k, 0], W[k, 1] = n.max(), n.mean()
        W[k, 2] = n[-2:].mean() - n[:2].mean()
        W[k, 3:7] = x[:, 1:5].max(0)
        v = x[:, 5]
        W[k, 7] = np.nanmin(v) if np.isfinite(v).any() else np.nan
        W[k, 8] = x[:, 6].max()
        W[k, 9:15] = x[:, 7:13].sum(0)
        with np.errstate(all="ignore"):
            W[k, 15:18] = np.nanmean(x[:, 13:16], 0)
            W[k, 19] = np.nanmean(x[:, 17])
        W[k, 18] = x[:, 16].max()
        W[k, 20:22] = x[:, 18:20].sum(0)
    return W


def main():
    d = c.R["rasgos"] / "g08"
    ins = pd.read_csv(d / "instantes.csv.gz", usecols=["id_instante", "video_id", "track_id", "frame", "x1", "y1", "x2", "y2",
                                                      "cx1", "cy1", "cx2", "cy2", "vel"])
    w = pd.read_csv(d / "ventanas.csv.gz", usecols=["w_id", "i0", "i1"])
    Xi = np.full((len(ins), len(VARS)), np.nan, np.float32)
    diag = []
    for vid, g in ins.groupby("video_id", sort=False):
        X, dg = instantes_video(vid, g.reset_index(drop=True), None)
        Xi[g.index.to_numpy()] = X
        diag.append(dg)
        print(dg, flush=True)
    np.save(d / "interaccion_instantes.npy", Xi)
    W = resumen_ventanas(Xi, w)
    np.save(d / "interaccion_ventanas.npy", W)
    c.guardar_json({"variables_instante": VARS, "columnas_ventana": COLUMNAS, "diagnostico": diag,
                    "nan_por_columna": {col: int(np.isnan(W[:, k]).sum()) for k, col in enumerate(COLUMNAS)}},
                   d / "interaccion_columnas.json")
    print(f"interacción: {Xi.shape} instantes, {W.shape} ventanas -> {d}")


if __name__ == "__main__":
    main()
