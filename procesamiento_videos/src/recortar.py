"""Etapa 3: recortes, manifiesto, tabla de tracklets y video anotado.

Un tracklet vehicular entra al corpus si su punto de apoyo en el piso (centro de la base de
la caja) pasa al menos `min_segundos_en_area` dentro del área. Solo se usan sus cuadros
dentro del área. De cada tracklet se guarda un recorte cada K cuadros, contados desde su
primer cuadro en el área, con la caja expandida `margen_altos` altos por lado y sin dibujos.
El muestreo es uniforme y no depende de nada que delate la etiqueta.

Salida en la carpeta del video:
  recortes/<track_id>/<video_id>_t<track_id>_f<frame>.jpg
  <video_id>_manifiesto.csv     una fila por recorte
  <video_id>_tracklets.csv      una fila por tracklet: la unidad de etiquetado
  <video_id>_video_anotado.mp4  el área con cajas, el número de cada tracklet y la hora
  <video_id>_recortes.json      parámetros, conteos y tiempos
El video y los recortes salen de una sola lectura secuencial, así que sus índices de cuadro
coinciden con los de la detección.

Entrada: trayectorias de la etapa 2 y el video original.
Uso: se invoca desde correr.py (python src/correr.py --videos <video_id> --etapas recortar).
"""
import json
import time

import cv2
import numpy as np
import pandas as pd

import configuracion
import video
import zonas

NOMBRE_CLASE = {2: "auto", 5: "bus", 0: "persona"}


def con_velocidad(t, cfg):
    """Velocidad de cada fila: paso del centro por cuadro, en altos de la caja, con mediana
    móvil centrada sobre las filas del tracklet. Se calcula antes de recortar al área."""
    t = t.sort_values(["track_id", "frame"]).reset_index(drop=True)
    cx, cy, h = (t.x1 + t.x2) / 2, (t.y1 + t.y2) / 2, (t.y2 - t.y1).clip(lower=1)
    paso = np.hypot(cx.diff(), cy.diff()) / t.frame.diff() / h
    paso[t.track_id != t.track_id.shift()] = np.nan
    t["vel"] = paso.groupby(t.track_id).transform(
        lambda s: s.rolling(cfg["vel_ventana_filas"], center=True, min_periods=cfg["vel_min_filas"]).median())
    return t


def vehiculos_en_area(v, poligono, min_cuadros):
    v = v.copy()
    v["en_area"] = zonas.dentro((v.x1 + v.x2) / 2, v.y2, poligono)
    n = v[v.en_area].groupby("track_id").size()
    validos = set(n.index[n >= min_cuadros])
    descartados = int(n.size - len(validos))
    return v[v.track_id.isin(validos) & v.en_area].drop(columns="en_area").sort_values(["track_id", "frame"]), descartados


def tabla_tracklets(v, reloj, vid, cfg):
    g = v.groupby("track_id")
    alto = v.y2 - v.y1
    tl = pd.DataFrame({"n_cuadros_area": g.size(), "frame_ini": g.frame.min(), "frame_fin": g.frame.max(),
                       "pct_bus": g.clase.apply(lambda c: round(100 * (c == 5).mean(), 1)),
                       "pct_detenido": g.vel.apply(lambda s: round(100 * (s < cfg["umbral_detenido"]).mean(), 1)),
                       "alto_mediano_px": alto.groupby(v.track_id).median().round(1)})
    tl["clase"] = np.where(tl.pct_bus >= cfg["pct_bus_minimo"], "bus", "auto")
    tl["duracion_s"] = ((tl.frame_fin - tl.frame_ini + 1) / reloj.fps).round(2)
    tl["hora_ini"] = [reloj.hora(f) for f in tl.frame_ini]
    tl["hora_fin"] = [reloj.hora(f) for f in tl.frame_fin]
    tl["instante_ini"] = [reloj.instante(f).isoformat(timespec="milliseconds") for f in tl.frame_ini]
    tl["instante_fin"] = [reloj.instante(f).isoformat(timespec="milliseconds") for f in tl.frame_fin]
    ini, fin = g.head(1).set_index("track_id"), g.tail(1).set_index("track_id")
    for c in ("x1", "y1", "x2", "y2"):   # cajas inicial y final: sirven para enlazar fragmentos de un mismo vehículo
        tl[f"ini_{c}"], tl[f"fin_{c}"] = ini[c].round(1), fin[c].round(1)
    tl.insert(0, "video_id", vid)
    return tl.reset_index()


def plan_recortes(v, per, cfg, ancho, alto):
    v = v.copy()
    v["k"] = v.frame - v.groupby("track_id").frame.transform("min")
    s = v[v.k % cfg["cada_k_cuadros"] == 0].copy()
    d = cfg["margen_altos"] * (s.y2 - s.y1)
    s["cx1"] = (s.x1 - d).clip(lower=0).round().astype(int)
    s["cy1"] = (s.y1 - d).clip(lower=0).round().astype(int)
    s["cx2"] = (s.x2 + d).clip(upper=ancho).round().astype(int)
    s["cy2"] = (s.y2 + d).clip(upper=alto).round().astype(int)
    vf = {f: g[["track_id", "x1", "y1", "x2", "y2"]].values for f, g in v.groupby("frame")}
    pf = {f: g[["x1", "y1", "x2", "y2"]].values for f, g in per.groupby("frame")}
    otro, n_per = [], []
    for r in s.itertuples():   # auditoría: otro vehículo del área o personas enteras dentro del recorte
        o = vf[r.frame]
        o = o[o[:, 0] != r.track_id]
        ix = np.clip(np.minimum(o[:, 3], r.cx2) - np.maximum(o[:, 1], r.cx1), 0, None)
        iy = np.clip(np.minimum(o[:, 4], r.cy2) - np.maximum(o[:, 2], r.cy1), 0, None)
        otro.append(bool((ix * iy / ((o[:, 3] - o[:, 1]) * (o[:, 4] - o[:, 2])) >= cfg["umbral_otro_vehiculo"]).any()))
        q = pf.get(r.frame)
        n_per.append(0 if q is None else int(((q[:, 0] >= r.cx1) & (q[:, 1] >= r.cy1) &
                                              (q[:, 2] <= r.cx2) & (q[:, 3] <= r.cy2)).sum()))
    s["otro_vehiculo_en_recorte"], s["personas_en_recorte"] = otro, n_per
    return s


class Dibujante:
    """Dibuja un cuadro del video anotado: el rectángulo del área, lo exterior al polígono
    atenuado, cajas por clase, el número de cada tracklet, personas sin número, una leyenda
    y abajo el video, el cuadro y la hora."""

    def __init__(self, cfg, poligono, rect, vid, reloj):
        self.cfg, self.rect, self.vid, self.reloj = cfg, rect, vid, reloj
        self.dentro = zonas.mascara(poligono) if cfg["atenuar_fuera_del_area"] else None
        self.col = {c: tuple(cfg["colores_bgr"][n]) for c, n in NOMBRE_CLASE.items()}
        self.fuente = cv2.FONT_HERSHEY_SIMPLEX

    def __call__(self, cuadro, k, veh, per, clase_de):
        c, (zx, zy, zw, zh), g = self.cfg, self.rect, self.cfg["grosor_caja"]
        src = cuadro[zy:zy + zh, zx:zx + zw]
        if self.dentro is not None:
            im = cv2.convertScaleAbs(src, alpha=c["factor_atenuacion"])
            cv2.copyTo(src, self.dentro, im)
        else:
            im = src.copy()
        for x1, y1, x2, y2 in per:
            cv2.rectangle(im, (int(x1 - zx), int(y1 - zy)), (int(x2 - zx), int(y2 - zy)), self.col[0], g)
        e, m = c["escala_texto_id"], c["relleno_etiqueta"]
        for tid, x1, y1, x2, y2 in veh:
            col = self.col[clase_de[int(tid)]]
            a, b = (int(x1 - zx), int(y1 - zy)), (int(x2 - zx), int(y2 - zy))
            cv2.rectangle(im, a, b, col, g)
            txt = f"#{int(tid)}"
            (tw, th), _ = cv2.getTextSize(txt, self.fuente, e, g)
            ty = max(0, a[1] - th - m)
            cv2.rectangle(im, (a[0], ty), (a[0] + tw + m, ty + th + m), col, -1)
            cv2.putText(im, txt, (a[0] + m // 2, ty + th + m // 4), self.fuente, e, (0, 0, 0), g, cv2.LINE_AA)
        # leyenda arriba a la derecha, con el alto de línea derivado del tamaño del texto
        (lw, lh), _ = cv2.getTextSize("persona", self.fuente, e, g)
        paso, x0 = lh + 2 * m, zw - lw - lh - 3 * m
        capa = im.copy()
        cv2.rectangle(capa, (x0 - m, m), (zw - m, m + paso * len(self.col) + m), (0, 0, 0), -1)
        im = cv2.addWeighted(capa, c["opacidad_leyenda"], im, 1 - c["opacidad_leyenda"], 0)
        for j, cl in enumerate(self.col):
            y = 2 * m + paso * j
            cv2.rectangle(im, (x0, y), (x0 + lh, y + lh), self.col[cl], -1)
            cv2.putText(im, NOMBRE_CLASE[cl], (x0 + lh + m, y + lh), self.fuente, e, (255, 255, 255), g, cv2.LINE_AA)
        hud = f"{self.vid} f{k:06d} {self.reloj.hora(k)}"
        eh = c["escala_texto_hud"]
        cv2.putText(im, hud, (2 * m, zh - 2 * m), self.fuente, eh, (0, 0, 0), c["grosor_borde_hud"], cv2.LINE_AA)
        cv2.putText(im, hud, (2 * m, zh - 2 * m), self.fuente, eh, (255, 255, 255), g, cv2.LINE_AA)
        return im


def correr(p, fila, carpeta, log=print):
    cfg, cva, vid = p["recortes"], p["video_anotado"], fila["video_id"]
    t0 = time.time()
    reloj = video.Reloj(fila)
    meta_seg = json.loads((carpeta / f"{vid}_seguimiento.json").read_text(encoding="utf-8"))
    n = int(meta_seg["cuadros"])
    poligono, rect = zonas.area(p)
    ancho, alto = p["video"]["resolucion"]
    v = pd.read_csv(carpeta / f"{vid}_trayectorias_vehiculos.csv.gz")
    per = pd.read_csv(carpeta / f"{vid}_trayectorias_personas.csv.gz")
    v = con_velocidad(v, cfg)
    min_cuadros = round(cfg["min_segundos_en_area"] * reloj.fps)
    v, descartados = vehiculos_en_area(v, poligono, min_cuadros)
    tl = tabla_tracklets(v, reloj, vid, cfg)
    plan = plan_recortes(v, per, cfg, ancho, alto)
    plan["ruta_recorte"] = [f"recortes/{t:05d}/{vid}_t{t:05d}_f{f:06d}.jpg" for t, f in zip(plan.track_id, plan.frame)]
    clase_de = {int(t): (5 if c == "bus" else 2) for t, c in zip(tl.track_id, tl.clase)}

    for d in {carpeta / r.rsplit("/", 1)[0] for r in plan.ruta_recorte}:
        d.mkdir(parents=True, exist_ok=True)
    por_cuadro = {f: g[["cy1", "cy2", "cx1", "cx2", "ruta_recorte"]].values for f, g in plan.groupby("frame")}
    vf = {f: g[["track_id", "x1", "y1", "x2", "y2"]].values for f, g in v.groupby("frame")}
    pf = {f: g[["x1", "y1", "x2", "y2"]].values for f, g in per.groupby("frame")}
    escritor, dibujar = None, None
    ruta_video = carpeta / f"{vid}_video_anotado.mp4"
    if cva["generar"]:
        dibujar = Dibujante(cva, poligono, rect, vid, reloj)
        escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*cva["codec"]), reloj.fps, (rect[2], rect[3]))
        if not escritor.isOpened():
            raise SystemExit(f"No se pudo abrir el escritor de video con el códec {cva['codec']}")
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, cfg["calidad_jpeg"]]
    t = {"lectura": 0.0, "recortes": 0.0, "video": 0.0}
    cada, leidos = p["video"]["registrar_cada_cuadros"], 0
    t1 = time.perf_counter()
    for k, cuadro in video.cuadros(fila["ruta"], 0, n):
        t2 = time.perf_counter()
        for cy1, cy2, cx1, cx2, ruta in por_cuadro.get(k, ()):
            if not cv2.imwrite(str(carpeta / ruta), cuadro[cy1:cy2, cx1:cx2], jpeg):
                raise SystemExit(f"No se pudo escribir {ruta}")
        t3 = time.perf_counter()
        if escritor is not None:
            escritor.write(dibujar(cuadro, k, vf.get(k, ()), pf.get(k, ()), clase_de))
        t4 = time.perf_counter()
        t["lectura"] += t2 - t1
        t["recortes"] += t3 - t2
        t["video"] += t4 - t3
        t1, leidos = t4, k + 1
        if leidos % cada == 0:
            log(f"{vid} recortes: cuadro {leidos}/{n} ({100 * leidos / n:.1f} %)")
    if escritor is not None:
        escritor.release()
    if leidos < n:
        raise SystemExit(f"{vid}: el video terminó en el cuadro {leidos} y se esperaban {n}")

    tl["n_recortes"] = plan.groupby("track_id").size().reindex(tl.track_id).fillna(0).astype(int).values
    man = plan.assign(video_id=vid, hora=[reloj.hora(f) for f in plan.frame],
                      instante=[reloj.instante(f).isoformat(timespec="milliseconds") for f in plan.frame])
    man["clase"] = man.clase.map(NOMBRE_CLASE)
    man["detenido"] = man.vel < cfg["umbral_detenido"]
    cols = ["video_id", "track_id", "frame", "hora", "instante", "clase", "conf", "x1", "y1", "x2", "y2",
            "cx1", "cy1", "cx2", "cy2", "ruta_recorte", "vel", "detenido", "otro_vehiculo_en_recorte",
            "personas_en_recorte"]
    man[cols].round({"vel": 4}).to_csv(carpeta / f"{vid}_manifiesto.csv", index=False)
    tl.to_csv(carpeta / f"{vid}_tracklets.csv", index=False)
    bytes_rec = sum(f.stat().st_size for f in (carpeta / "recortes").rglob("*.jpg"))
    meta = {"firma": configuracion.firma(cfg, cva, meta_seg["firma"], fila["hora_cuadro0"], fila["fps"]),
            "estado": "completo", "parametros": {"recortes": cfg, "video_anotado": cva},
            "hora_cuadro0": fila["hora_cuadro0"], "fps": float(fila["fps"]), "min_cuadros_en_area": min_cuadros,
            "cuadros": n, "tracklets": len(tl), "tracklets_descartados_menos_de_min": descartados,
            "recortes": len(plan), "gb_recortes": round(bytes_rec / 1e9, 3),
            "gb_video_anotado": round(ruta_video.stat().st_size / 1e9, 3) if escritor is not None else 0,
            "ms_por_cuadro": {kk: round(1000 * x / max(1, n), 1) for kk, x in t.items()},
            "minutos": round((time.time() - t0) / 60, 1)}
    (carpeta / f"{vid}_recortes.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{vid} recortes: {meta['tracklets']} tracklets, {meta['recortes']} recortes ({meta['gb_recortes']} GB), "
        f"video {meta['gb_video_anotado']} GB, {meta['minutos']} min")
