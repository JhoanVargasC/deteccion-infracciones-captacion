"""Validaciones del conjunto de datos antes de entrenar o evaluar.

Cada comprobación es "error" (detiene el pipeline: sale con código 1) o "aviso" (solo se informa).
  1. Cobertura: los videos del conjunto son exactamente los revisados completos.
  2. Unicidad de las unidades y coherencia de sus cuadros (k_ini <= k_fin, dentro del video).
  3. Correspondencia: cada etiqueta final encuentra al menos una unidad; las que no, se listan con el motivo.
  4. Fuga temporal: cada vehículo (fragmentos enlazados) está en una sola partición, en todas las tablas, y
     cada detención G41 está en la partición de su tracklet.
  5. Particiones: todas tienen al menos `min_positivos_por_fold` positivos por infracción; se informa la
     composición (positivos, día y noche, confianza) de cada una.
  6. Recortes: cada automóvil G08 tiene recortes; una muestra al azar existe, se lee y contiene la caja del
     vehículo.
Entradas: tablas y manifiesto de rutas.dataset.
Salida: rutas.resultados/validacion/validacion.json y validacion.md
Uso: python src/validar_dataset.py
"""
import sys

import cv2
import numpy as np
import pandas as pd

import comun as c

V = c.P["validacion"]


def main():
    ds = c.R["dataset"]
    t = {inf: pd.read_csv(ds / f"{inf.lower()}.csv") for inf in ("G10", "G41", "G08")}
    tl = pd.read_csv(ds / "tracklets.csv")
    man = pd.read_csv(ds / "manifiesto_recortes.csv.gz", usecols=["video_id", "track_id", "frame", "clase", "x1", "y1", "x2", "y2",
                                                                 "cx1", "cy1", "cx2", "cy2", "ruta_recorte", "fold"])
    vids = c.videos()
    res = []

    def chequeo(nombre, ok, nivel, detalle):
        res.append({"chequeo": nombre, "ok": bool(ok), "nivel": nivel, "detalle": detalle})
        print(f"[{'ok' if ok else nivel.upper()}] {nombre}: {detalle}")

    # 1. cobertura
    en_ds = sorted(set().union(*[set(u.video_id) for u in t.values()]))
    chequeo("cobertura", en_ds == sorted(vids), "error", f"{len(en_ds)} videos en el conjunto, {len(vids)} revisados completos")

    # 2. unicidad y cuadros
    cuadros = c.IND.cuadros.to_dict()
    for inf, u in t.items():
        clave = ["video_id", "track_id"] + (["k_ini"] if inf == "G41" else [])
        dup = int(u.duplicated(clave).sum())
        malos = int(((u.k_ini > u.k_fin) | (u.k_ini < 0) | (u.k_fin >= u.video_id.map(cuadros))).sum())
        chequeo(f"{inf} unidades únicas y cuadros válidos", dup == 0 and malos == 0, "error",
                f"{len(u)} unidades, {dup} duplicadas, {malos} con cuadros fuera del video")

    # 3. correspondencia etiqueta -> unidad
    sin_unidad = {}
    for inf, u in t.items():
        et = c.etiquetas(inf)
        et = et[et.video_id.isin(vids)].reset_index(drop=True)
        g = {k: v for k, v in u.groupby(["video_id", "track_id"])}
        faltan = []
        for i, r in et.iterrows():
            x = g.get((r.video_id, r.track_id))
            if x is None or not ((x.k_ini <= r.k_fin) & (x.k_fin >= r.k_ini)).any():
                if not ((tl.video_id == r.video_id) & (tl.track_id == r.track_id)).any():
                    # el vehículo existe en las trayectorias pero no llegó a 1 s en el área: no forma parte
                    # de la población de unidades (procesamiento, recortes.min_segundos_en_area)
                    tv = c.leer_csv(r.video_id, "trayectorias_vehiculos.csv.gz", usecols=["track_id", "frame"])
                    n = int((tv.track_id == r.track_id).sum())
                    corto = 0 < n < c.PP["recortes"]["min_segundos_en_area"] * c.fps(r.video_id)
                    motivo = f"fuera del corpus ({n} cuadros, menos de 1 s)" if corto else "sin tracklet"
                else:
                    motivo = ("el tracklet no es automóvil" if inf == "G08" else
                              "sin detención estricta que se superponga" if inf == "G41" else "sin unidad")
                faltan.append({"video_id": r.video_id, "track_id": int(r.track_id), "k_ini": int(r.k_ini), "k_fin": int(r.k_fin),
                               "motivo": motivo})
        sin_unidad[inf] = faltan
        # Un vehículo que estuvo menos de 1 s en el área o sin detención estricta no tiene unidad: se informa.
        # Cualquier otro motivo es un error.
        graves = [f for f in faltan if not (f["motivo"].startswith("fuera del corpus") or f["motivo"].startswith("sin detención"))]
        chequeo(f"{inf} etiquetas con unidad", not graves, "error",
                f"{len(et) - len(faltan)}/{len(et)} etiquetas encuentran unidad; sin unidad: "
                f"{pd.Series([f['motivo'] for f in faltan]).value_counts().to_dict() if faltan else 0}")

    # 4. fuga temporal
    pf = tl.groupby("vehiculo").fold.nunique()
    chequeo("un vehículo, una partición", (pf == 1).all(), "error",
            f"{int((pf > 1).sum())} vehículos en más de una partición ({tl.vehiculo.nunique()} vehículos, {len(tl)} tracklets)")
    for inf, u in t.items():
        m = u.merge(tl[["video_id", "track_id", "fold"]], on=["video_id", "track_id"], how="left", suffixes=("", "_tl"))
        chequeo(f"{inf} partición = partición del tracklet", (m.fold == m.fold_tl).all(), "error",
                f"{int((m.fold != m.fold_tl).sum())} unidades difieren")
    mf = man.merge(tl[["video_id", "track_id", "fold"]], on=["video_id", "track_id"], how="left", suffixes=("", "_tl"))
    chequeo("recortes en la partición de su tracklet", (mf.fold == mf.fold_tl).all(), "error",
            f"{int((mf.fold != mf.fold_tl).sum())} recortes difieren")

    # 5. particiones
    comp = []
    noche = {v: c.IND.loc[v, "sesion"].endswith("tarde") for v in vids}
    for inf, u in t.items():
        for f, g in u[u.fold >= 0].groupby("fold"):
            comp.append({"infraccion": inf, "fold": int(f), "unidades": len(g), "positivos": int(g.etiqueta.sum()),
                         "positivos_noche": int(g.etiqueta[g.periodo == "noche"].sum()),
                         "positivos_confianza_alta": int((g.etiqueta.astype(bool) & (g.confianza == "alta")).sum())})
    comp = pd.DataFrame(comp)
    minimo = comp.groupby("infraccion").positivos.min()
    chequeo("positivos por partición", (minimo >= V["min_positivos_por_fold"]).all(), "error",
            f"mínimo por infracción: {minimo.to_dict()}")

    # 6. recortes
    autos = t["G08"][["video_id", "track_id"]]
    n_rec = man[man.clase == "auto"].groupby(["video_id", "track_id"]).size()
    sin_rec = int((~pd.MultiIndex.from_frame(autos).isin(n_rec.index)).sum())
    chequeo("automóviles G08 con recortes", sin_rec == 0, "error", f"{sin_rec} de {len(autos)} sin recortes")
    nr = tl.set_index(["video_id", "track_id"]).n_recortes
    dif = int((man.groupby(["video_id", "track_id"]).size().reindex(nr.index).fillna(0) != nr).sum())
    chequeo("recortes del manifiesto = n_recortes del tracklet", dif == 0, "error", f"{dif} tracklets difieren")
    rng = np.random.default_rng(V["semilla"])
    mu = man.iloc[rng.choice(len(man), min(V["muestra_recortes"], len(man)), replace=False)]
    fallos, fuera = [], 0
    for r in mu.itertuples():
        im = cv2.imread(r.ruta_recorte)
        if im is None:
            fallos.append(r.ruta_recorte)
            continue
        h, w = im.shape[:2]
        if abs(w - (r.cx2 - r.cx1)) > 1 or abs(h - (r.cy2 - r.cy1)) > 1 or r.x1 < r.cx1 - 1 or r.y1 < r.cy1 - 1 \
                or r.x2 > r.cx2 + 1 or r.y2 > r.cy2 + 1:
            fuera += 1
    chequeo("muestra de recortes legibles", not fallos, "error", f"{len(mu) - len(fallos)}/{len(mu)} se leen")
    chequeo("caja dentro del recorte y tamaño coherente", fuera == 0, "error", f"{fuera} de {len(mu) - len(fallos)} incoherentes")

    chequeo("M21 fuera del conjunto", not (ds / "m21.csv").exists(), "error",
            "M21 no entra al conjunto de datos: se evalúa aparte con evaluar_m21.py")

    errores = [r for r in res if not r["ok"] and r["nivel"] == "error"]
    out = c.salida("validacion")
    c.guardar_json({"chequeos": res, "sin_unidad": sin_unidad, "particiones": comp.to_dict("records")}, out / "validacion.json")
    lineas = ["# Validación del conjunto de datos", "",
              f"Resultado: **{'APROBADO' if not errores else 'CON ERRORES'}** ({len(errores)} errores, "
              f"{sum(1 for r in res if not r['ok'] and r['nivel'] == 'aviso')} avisos).", "",
              "| Chequeo | Resultado | Detalle |", "|---|---|---|"]
    lineas += [f"| {r['chequeo']} | {'ok' if r['ok'] else r['nivel']} | {r['detalle']} |" for r in res]
    lineas += ["", "## Composición de las particiones", "", c.markdown(comp)]
    for inf, f in sin_unidad.items():
        if f:
            lineas += ["", f"## {inf}: etiquetas sin unidad", "", c.markdown(pd.DataFrame(f))]
    (out / "validacion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\n{'APROBADO' if not errores else 'CON ERRORES'} -> {out}")
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
