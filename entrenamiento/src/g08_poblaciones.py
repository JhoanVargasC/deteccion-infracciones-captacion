"""Roles de los automóviles de G08 (Etapa 4).

Usa solo la geometría, el tiempo y la etiqueta del automóvil (nunca el puntaje del modelo).
  persona_junto  candidato: automóvil con una detención estricta durante la cual empieza o termina, junto a
                 él, el tracklet de una persona (selección de candidatos).
  parecido       automóvil sin evento con personas próximas, o detenido a menos de 1 altura y 10 minutos de
                 un automóvil con evento (grupo de negativas «parecidos» del entrenamiento).
  falsa          tracklet del conjunto F de cajas falsas (etiquetado/src/candidatos.py): no es un automóvil y
                 no se puntúa.
Posición de un automóvil: mediana del punto de contacto en sus instantes detenidos (en sus instantes del
evento, si tiene evento); su altura, la mediana de la altura de esas cajas.

Entradas: g08.csv (rutas.dataset), instantes.csv.gz e interaccion_instantes.npy (rutas.rasgos/g08).
Salida (rutas.rasgos/g08): autos.csv (un automóvil por fila, con fold, etiqueta y roles) y autos.json
Uso: python src/g08_poblaciones.py
"""
import numpy as np
import pandas as pd

import comun as c

G = c.P["g08"]


def main():
    d = c.R["rasgos"] / "g08"
    autos = pd.read_csv(c.R["dataset"] / "g08.csv")[["video_id", "track_id", "vehiculo", "fold", "etiqueta", "duracion_s",
                                                    "pct_detenido", "k_ini", "k_fin"]]
    ins = pd.read_csv(d / "instantes.csv.gz", usecols=["video_id", "track_id", "frame", "x1", "x2", "y1", "y2", "en_evento"])
    Xi = np.load(d / "interaccion_instantes.npy", mmap_mode="r")
    import json
    cols = json.loads((d / "interaccion_columnas.json").read_text(encoding="utf-8"))["variables_instante"]
    det = Xi[:, cols.index("detenido")] > 0
    ext = (Xi[:, cols.index("apariciones")] + Xi[:, cols.index("desapariciones")]) > 0
    ins["px"], ins["h"] = (ins.x1 + ins.x2) / 2, ins.y2 - ins.y1
    ins["det"], ins["ext_det"] = det, det & ext
    clave = ["video_id", "track_id"]
    con_ev = autos.etiqueta == 1
    # Posición: instantes del evento (con evento) o instantes detenidos (sin evento).
    usar = np.where(ins.merge(autos[clave + ["etiqueta"]], on=clave, how="left").etiqueta.to_numpy() == 1, ins.en_evento, ins.det)
    pos = ins[usar].groupby(clave).agg(px=("px", "median"), py=("y2", "median"), h=("h", "median"),
                                       k_med=("frame", "median")).reset_index()
    autos = autos.merge(pos, on=clave, how="left")
    autos["persona_junto"] = autos.set_index(clave).index.isin(ins[ins.ext_det].set_index(clave).index.unique())
    # Cajas falsas: el conjunto F calculado en etiquetado/src/candidatos.py.
    det = pd.concat([pd.read_csv(c.ETQ / "salidas" / "candidatos" / f"{v}_detenciones.csv").assign(video_id=v)
                     for v in autos.video_id.unique()])
    F = det[det.falsa][clave].drop_duplicates()
    autos["falsa"] = autos.set_index(clave).index.isin(F.set_index(clave).index)
    assert not (autos.falsa & (autos.etiqueta == 1)).any(), "una caja falsa tiene un evento G08: revisar"
    ev = c.etiquetas("G08")
    ev = ev.merge(autos[clave + ["px", "py", "h"]], on=clave, how="left")

    mismo_lugar = np.zeros(len(autos), bool)
    for e in ev.itertuples():
        f = c.fps(e.video_id)
        cand = (autos.video_id == e.video_id) & ~con_ev & autos.px.notna() & ~autos.falsa
        dist = np.hypot(autos.px - e.px, autos.py - e.py) / e.h
        dt = (autos.k_med - (e.k_ini + e.k_fin) / 2).abs() / f / 60
        cerca = cand & (dist <= G["parecidos_radio_altos"]) & (dt <= G["parecidos_max_min"])
        mismo_lugar |= cerca.to_numpy()
    autos["parecido"] = ~con_ev & ~autos.falsa & (autos.persona_junto | mismo_lugar)
    autos.to_csv(d / "autos.csv", index=False)
    r = {"autos": len(autos), "falsas_excluidas": int(autos.falsa.sum()),
         "con_evento": int(con_ev.sum()), "parecidos": int(autos.parecido.sum()),
         "candidatos": int(autos.persona_junto.sum()), "parecidos_por_lugar": int(mismo_lugar.sum())}
    c.guardar_json(r, d / "autos.json")
    print(r)


if __name__ == "__main__":
    main()
