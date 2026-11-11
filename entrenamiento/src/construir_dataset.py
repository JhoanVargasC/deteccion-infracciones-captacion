"""Construcción del conjunto de datos (Etapa 2): una tabla por infracción, con una fila por unidad de análisis.

  G10  tracklet del vehículo
  G41  detención estricta del vehículo (sin cajas falsas); la evaluación agrupa las detenciones por vehículo
  G08  tracklet del automóvil
Una unidad recibe la etiqueta 1 si coincide en el tiempo con un caso anotado del mismo vehículo, y 0 en caso
contrario. Las ocho horas se revisaron completas, de modo que lo no anotado es negativo.

Partición de G08: tramos de 10 minutos; el tramo b va a la partición b mod 5. Dos tracklets contiguos de un
mismo vehículo reciben la partición del primero, y una detención hereda la partición de su tracklet.

Columnas comunes: video_id, track_id, k_ini, k_fin, instante_ini, etiqueta (0/1), confianza, vehiculo,
bloque, fold y periodo. Los recortes no se copian: el manifiesto apunta a la carpeta del procesamiento.

Entradas: etiquetas finales (etiquetado) y salidas por video (procesamiento_videos); iluminacion.csv.
Salida (rutas.dataset): g10.csv, g41.csv, g08.csv, tracklets.csv, manifiesto_recortes.csv.gz, resumen.csv
Uso: python src/construir_dataset.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import comun as c

D = c.P["dataset"]


def enlazar_fragmentos(tl):
    """Etiqueta de vehículo: componentes conexas de los enlaces fin de A -> inicio de B."""
    padre = {}

    def raiz(x):
        while padre.get(x, x) != x:
            padre[x] = padre.get(padre[x], padre[x])
            x = padre[x]
        return x

    n_enl = 0
    for vid, g in tl.groupby("video_id"):
        g = g.sort_values("k_ini")
        fin = g[["track_id", "k_fin", "fin_x1", "fin_x2", "fin_y1", "fin_y2"]].to_numpy()
        for r in g.itertuples():
            cand = fin[(fin[:, 1] < r.k_ini) & (r.k_ini - fin[:, 1] <= D["fragmento_max_cuadros"]) & (fin[:, 0] != r.track_id)]
            if not len(cand):
                continue
            h = np.maximum(cand[:, 5] - cand[:, 4], 1.0)
            d = np.hypot((cand[:, 2] + cand[:, 3]) / 2 - (r.ini_x1 + r.ini_x2) / 2, cand[:, 5] - r.ini_y2) / h
            ok = d <= D["fragmento_max_altos"]
            if ok.any():
                a = (vid, int(cand[ok][np.argmin(d[ok]), 0]))
                b = (vid, int(r.track_id))
                padre[raiz(b)] = raiz(a)
                n_enl += 1
    tl["vehiculo"] = [f"{v}:{raiz((v, int(t)))[1]}" for v, t in zip(tl.video_id, tl.track_id)]
    return tl, n_enl


def solapa(u, et):
    """Para cada unidad (video_id, track_id, k_ini, k_fin): índice de la etiqueta que se superpone o -1."""
    g = {k: v for k, v in et.groupby(["video_id", "track_id"])}
    out = []
    for r in u.itertuples():
        e = g.get((r.video_id, r.track_id))
        if e is None:
            out.append(-1)
            continue
        m = e[(e.k_ini <= r.k_fin) & (e.k_fin >= r.k_ini)]
        out.append(m.index[0] if len(m) else -1)
    return np.array(out)


def marcar(u, et, inf):
    i = solapa(u, et)
    u["etiqueta"] = (i >= 0).astype(int)
    u["confianza"] = np.where(i >= 0, et.confianza.reindex(i).values, "")
    return u


def main():
    out = c.R["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    vids = c.videos()
    et = {inf: c.etiquetas(inf) for inf in ("G10", "G41", "G08")}
    inicio_sesion = {s: datetime.fromisoformat(f"{g.fecha.iloc[0]}T{g.hora_cuadro0.iloc[0]}") for s, g in c.IND.groupby("sesion")}
    tl, det, man = [], [], []
    for vid in vids:
        f = c.IND.loc[vid]
        t = c.leer_csv(vid, "tracklets.csv").rename(columns={"frame_ini": "k_ini", "frame_fin": "k_fin"})
        t["sesion"] = f.sesion
        tl.append(t)
        d = pd.read_csv(c.ETQ / "salidas" / "candidatos" / f"{vid}_detenciones.csv")
        det.append(d[~d.falsa].assign(video_id=vid))
        m = c.leer_csv(vid, "manifiesto.csv")
        m["ruta_recorte"] = str(c.carpeta_video(vid)) + "/" + m.ruta_recorte
        man.append(m)
    tl, det, man = pd.concat(tl, ignore_index=True), pd.concat(det, ignore_index=True), pd.concat(man, ignore_index=True)

    # particiones por vehículo: el bloque del primer fragmento
    tl, n_enl = enlazar_fragmentos(tl)
    seg = [(datetime.fromisoformat(i) - inicio_sesion[s]).total_seconds() for i, s in zip(tl.instante_ini, tl.sesion)]
    tl["bloque_propio"] = (np.array(seg) // (60 * D["bloque_min"])).astype(int)
    primero = tl.sort_values("instante_ini").groupby("vehiculo").bloque_propio.first()
    tl["bloque"] = tl.vehiculo.map(primero).astype(int)
    tl["fold"] = tl.bloque % D["n_folds"]
    cambia = int((tl.bloque_propio % D["n_folds"] != tl.fold).sum())
    tl = tl.drop(columns="bloque_propio")
    clave = ["video_id", "track_id"]
    det = det.merge(tl[clave + ["vehiculo", "bloque", "fold", "sesion", "clase"]], on=clave, how="left", validate="many_to_one")
    assert det.fold.notna().all(), "hay detenciones sin tracklet"
    t0 = {v: datetime.fromisoformat(f"{c.IND.loc[v, 'fecha']}T{c.IND.loc[v, 'hora_cuadro0']}") for v in vids}
    det["instante_ini"] = [(t0[v] + pd.Timedelta(seconds=k / c.fps(v))).isoformat() for v, k in zip(det.video_id, det.k_ini)]

    # período de iluminación: el del bloque en que empieza cada unidad (iluminacion.csv)
    il = pd.read_csv(out / "iluminacion.csv")

    def periodo(u):
        seg = np.array([(datetime.fromisoformat(i) - inicio_sesion[s]).total_seconds() for i, s in zip(u.instante_ini, u.sesion)])
        b = (seg // (60 * D["bloque_min"])).astype(int)
        res = []
        for v, bb in zip(u.video_id, b):
            g = il[il.video_id == v]
            res.append(g.periodo.iloc[int(np.argmin(np.abs(g.bloque.to_numpy() - bb)))])   # bloque medido más cercano
        u["periodo"] = res
        return u

    tl, det = periodo(tl), periodo(det)
    tablas = {"G10": marcar(tl.copy(), et["G10"], "G10"),
              "G41": marcar(det.copy(), et["G41"], "G41"),
              "G08": marcar(tl[tl.clase == "auto"].copy(), et["G08"], "G08")}
    # hora sin eventos de captación: fuera de la tabla de G08 (dataset.excluir_videos)
    for inf, vs in (D.get("excluir_videos") or {}).items():
        tablas[inf] = tablas[inf][~tablas[inf].video_id.isin(vs)]
    res = []
    for inf, u in tablas.items():
        u.to_csv(out / f"{inf.lower()}.csv", index=False)
        for vid, g in u.groupby("video_id"):
            res.append({"infraccion": inf, "video_id": vid, "unidades": len(g), "positivos": int(g.etiqueta.sum())})
        print(f"{inf}: {len(u)} unidades, {int(u.etiqueta.sum())} positivas, por fold {u.groupby('fold').etiqueta.sum().to_dict()}")
    tl.to_csv(out / "tracklets.csv", index=False)
    man = man.merge(tl[clave + ["vehiculo", "fold"]], on=clave, how="left", validate="many_to_one")
    man.to_csv(out / "manifiesto_recortes.csv.gz", index=False)
    pd.DataFrame(res).to_csv(out / "resumen.csv", index=False)
    print(f"Fragmentos enlazados: {n_enl}; tracklets que cambian de partición por seguir a su primer fragmento: {cambia}")
    print(f"Manifiesto: {len(man)} recortes (no copiados) -> {out}")


if __name__ == "__main__":
    main()
