"""Instantes de muestreo, ventanas y su etiqueta, para G08.

Instantes: los recortes de automóvil del manifiesto (uno cada 5 cuadros, ~3 por segundo) de los automóviles
del conjunto G08, ordenados por video, tracklet y cuadro.
Recortes contaminados: un recorte de un automóvil sin evento, tomado durante un evento, que contiene al
menos la mitad de la caja del automóvil del evento o la caja completa de una persona próxima a él. Se
marcan (columnas contamina_*) y no se borran: solo se excluyen del entrenamiento (g08_entrenar.py).
Ventanas: 6 instantes consecutivos; cada una empieza 3 instantes (1 s) después de la anterior. Si la última
no coincide con el final, se añade una que termina en el último instante. Un tracklet con menos de 6
instantes forma una sola ventana. La regla es la misma para todos los automóviles.
Etiqueta de la ventana: 1 si algún instante cae en un evento anotado del automóvil; -1 (excluida) si no,
pero alguno cae en el margen de 1 s alrededor del evento; 0 en otro caso.

Entradas: g08.csv y manifiesto_recortes.csv.gz (rutas.dataset), trayectorias del procesamiento y etiquetas G08.
Salida (rutas.rasgos/g08): instantes.csv.gz (fila = id del instante, el mismo índice que los rasgos),
ventanas.csv.gz, eventos.csv, contaminados.csv y ventanas.json
Uso: python src/g08_ventanas.py
"""
import numpy as np
import pandas as pd

import comun as c

G = c.P["g08"]


def proximas(per_f, caja, h):
    """Máscara de personas próximas: >= 50 % de su caja dentro de la caja ampliada 0,5 h."""
    m = G["margen_proximidad_altos"] * h
    ex = (caja[0] - m, caja[1] - m, caja[2] + m, caja[3] + m)
    ix = np.clip(np.minimum(per_f.x2, ex[2]) - np.maximum(per_f.x1, ex[0]), 0, None)
    iy = np.clip(np.minimum(per_f.y2, ex[3]) - np.maximum(per_f.y1, ex[1]), 0, None)
    area = np.maximum((per_f.x2 - per_f.x1) * (per_f.y2 - per_f.y1), 1e-9)
    return (ix * iy / area) >= G["ioa_proximidad"]


def contaminados(vid, ins, ev):
    """Índices (de `ins`) de recortes de automóviles sin evento que muestran un evento vecino."""
    if ev.empty:
        return []
    tr = c.leer_csv(vid, "trayectorias_vehiculos.csv.gz")
    per = c.leer_csv(vid, "trayectorias_personas.csv.gz")
    sin_evento = ~ins.track_id.isin(ev.track_id)
    en_ev = np.zeros(len(per), bool)
    for e in ev.itertuples():
        en_ev |= per.frame.between(e.k_ini, e.k_fin).to_numpy()
    per_f = {f: g for f, g in per[en_ev].groupby("frame")}
    vacio = per.iloc[0:0]
    fuera = []
    for e in ev.itertuples():
        caja_ev = tr[(tr.track_id == e.track_id) & tr.frame.between(e.k_ini, e.k_fin)].set_index("frame")
        cand = ins[sin_evento & ins.frame.between(e.k_ini, e.k_fin)]
        for r in cand.itertuples():
            if r.frame not in caja_ev.index:
                continue
            b = caja_ev.loc[r.frame]
            ix = max(0.0, min(b.x2, r.cx2) - max(b.x1, r.cx1))
            iy = max(0.0, min(b.y2, r.cy2) - max(b.y1, r.cy1))
            if ix * iy >= G["contaminado_frac_caja"] * (b.x2 - b.x1) * (b.y2 - b.y1):
                fuera.append((r.Index, e.track_id, "caja del automóvil del evento"))
                continue
            pf = per_f.get(r.frame, vacio)
            pf = pf[proximas(pf, (b.x1, b.y1, b.x2, b.y2), b.y2 - b.y1)]
            dentro = (pf.x1 >= r.cx1) & (pf.y1 >= r.cy1) & (pf.x2 <= r.cx2) & (pf.y2 <= r.cy2)
            if dentro.any():
                fuera.append((r.Index, e.track_id, "persona próxima al automóvil del evento"))
    return fuera


def ventanas(ins):
    L, paso = G["instantes_por_ventana"], G["paso_instantes"]
    clave = (ins.video_id + ":" + ins.track_id.astype(str)).to_numpy()
    cortes = np.r_[0, np.nonzero(clave[1:] != clave[:-1])[0] + 1, len(ins)]
    en_ev, en_mg = ins.en_evento.to_numpy(bool), ins.en_margen.to_numpy(bool)
    filas = []
    for s, e in zip(cortes[:-1], cortes[1:]):
        n = e - s
        inicios = list(range(s, e - L + 1, paso)) if n > L else [s]
        if n > L and inicios[-1] != e - L:
            inicios.append(e - L)
        for i0 in inicios:
            i1 = min(i0 + L, e)
            filas.append((s, i0, i1, 1 if en_ev[i0:i1].any() else (-1 if en_mg[i0:i1].any() else 0)))
    w = pd.DataFrame(filas, columns=["s", "i0", "i1", "etiqueta_ventana"])
    for col in ("video_id", "track_id", "vehiculo", "fold", "etiqueta"):
        w[col] = ins[col].to_numpy()[w.s]
    w["k_ini"], w["k_fin"] = ins.frame.to_numpy()[w.i0], ins.frame.to_numpy()[w.i1 - 1]
    w.insert(0, "w_id", np.arange(len(w)))
    return w.drop(columns="s")


def main():
    out = c.R["rasgos"] / "g08"
    out.mkdir(parents=True, exist_ok=True)
    ds = c.R["dataset"]
    autos = pd.read_csv(ds / "g08.csv")[["video_id", "track_id", "vehiculo", "fold", "etiqueta", "duracion_s", "pct_detenido"]]
    ev = c.etiquetas("G08")
    ev = ev[ev.video_id.isin(autos.video_id.unique())].reset_index(drop=True)
    man = pd.read_csv(ds / "manifiesto_recortes.csv.gz", usecols=[
        "video_id", "track_id", "frame", "clase", "x1", "y1", "x2", "y2", "cx1", "cy1", "cx2", "cy2", "ruta_recorte", "vel",
        "detenido", "personas_en_recorte"])
    man = man[man.clase == "auto"].drop(columns="clase")
    ins = man.merge(autos[["video_id", "track_id", "vehiculo", "fold", "etiqueta"]], on=["video_id", "track_id"])   # solo automóviles del conjunto
    ins = ins.sort_values(["video_id", "track_id", "frame"]).reset_index(drop=True)

    fuera = []
    for vid, g in ins.groupby("video_id"):
        x = contaminados(vid, g, ev[ev.video_id == vid])
        fuera += x
        print(f"{vid}: {len(x)} recortes contaminados", flush=True)
    cont = pd.DataFrame(fuera, columns=["fila", "track_evento", "motivo"])
    cont = cont.drop_duplicates("fila")
    n_autos_cont = int(ins.loc[cont.fila, ["video_id", "track_id"]].drop_duplicates().shape[0])
    cont.join(ins[["video_id", "track_id", "frame"]], on="fila").to_csv(out / "contaminados.csv", index=False)
    # Los recortes contaminados no se borran: se marcan con el automóvil del evento que los contamina.
    # g08_entrenar.py excluye del entrenamiento las ventanas que los contienen, y solo cuando el evento pertenece
    # a una partición de entrenamiento. En la prueba se ven todos, como en la operación real: quitarlos usaría
    # las etiquetas para limpiar la prueba y la haría más fácil, es decir, una fuga de información.
    ins["contamina_video"], ins["contamina_track"] = "", -1
    ins.loc[cont.fila, "contamina_video"] = ins.loc[cont.fila, "video_id"].to_numpy()
    ins.loc[cont.fila, "contamina_track"] = cont.track_evento.to_numpy()

    ins["en_evento"] = False
    ins["en_margen"] = False
    for e in ev.itertuples():
        m = (ins.video_id == e.video_id) & (ins.track_id == e.track_id)
        mg = round(G["margen_evento_s"] * c.fps(e.video_id))
        ins.loc[m & ins.frame.between(e.k_ini, e.k_fin), "en_evento"] = True
        ins.loc[m & ins.frame.between(e.k_ini - mg, e.k_fin + mg), "en_margen"] = True
    ins.insert(0, "id_instante", np.arange(len(ins)))
    w = ventanas(ins)
    sin_ventana_pos = sorted(set(map(tuple, ev[["video_id", "track_id"]].values)) -
                             set(map(tuple, w[w.etiqueta_ventana == 1][["video_id", "track_id"]].values)))
    ins.to_csv(out / "instantes.csv.gz", index=False)
    w.to_csv(out / "ventanas.csv.gz", index=False)
    ev.to_csv(out / "eventos.csv", index=False)
    r = {"instantes": len(ins), "recortes_contaminados_marcados": len(cont),
         "automoviles_con_contaminados": n_autos_cont,
         "ventanas": len(w), "positivas": int((w.etiqueta_ventana == 1).sum()), "excluidas": int((w.etiqueta_ventana == -1).sum()),
         "negativas": int((w.etiqueta_ventana == 0).sum()), "eventos": len(ev),
         "eventos_sin_ventana_positiva": [list(map(str, x)) for x in sin_ventana_pos],
         "positivas_por_fold": w[w.etiqueta_ventana == 1].groupby("fold").size().to_dict(),
         "autos_con_evento_por_fold": autos[autos.etiqueta == 1].groupby("fold").size().to_dict()}
    r["negativas_por_positiva"] = round(r["negativas"] / max(1, r["positivas"]), 1)
    c.guardar_json(r, out / "ventanas.json")
    print(r)


if __name__ == "__main__":
    main()
