"""Exporta clips para que un validador externo (un oficial de tránsito) confirme las etiquetas
sin conocer su origen.

Entrada: salidas/<anotador>/ (etiquetas y candidatos G10 sin anotar) y las trayectorias.

Cada clip sale del video original, sin cajas ni números de seguimiento ni zonas. Solo el
vehículo a evaluar va marcado con un rectángulo de color, y un contador muestra los segundos
del clip. Los clips mezclan tres orígenes, en orden aleatorio y con nombres neutros:
  - etiqueta: un vehículo anotado (G08, G10, G41 o M21; las etiquetas cercanas de un mismo
    vehículo se juntan en un clip);
  - candidato: un vehículo que pisa el carril exclusivo y no fue anotado como G10;
  - control: un vehículo parecido sin ninguna etiqueta (detenido antes del crucero, en el
    carril vecino al exclusivo, o detenido sin evento).
Así el validador no puede suponer que todos los clips muestran una infracción.

Salidas:
  <carpeta de entrega>/clips/clip_NNN.mp4     lo que ve el validador
  <carpeta de entrega>/validacion.csv         base de la hoja que llena el validador
  salidas/<anotador>/clave_clips.csv          qué es cada clip (no se entrega al validador)

Uso: python src/exportar_clips.py --anotador anotador_a
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((RAIZ / P["rutas"]["procesamiento"]).resolve() / "src"))
import configuracion, indice, video, zonas  # noqa: E402

C = P["clips_validacion"]


def velocidad(t):
    cx, cy, h = (t.x1 + t.x2) / 2, (t.y1 + t.y2) / 2, (t.y2 - t.y1).clip(lower=1)
    v = np.hypot(cx.diff(), cy.diff()) / t.frame.diff() / h
    return v.rolling(C["vel_ventana"], center=True, min_periods=C["vel_min"]).median()


def escribir_clips(planes, vidas, entrega, p, ind, rng):
    """Pasos 4 y 5: ventana de cada clip (márgenes, duración acotada, controles con duración sorteada),
    orden aleatorio, nombres neutros y escritura con una lectura secuencial por video. Modifica `planes`
    (agrega clip, c0, c1). `vidas`: (video_id, track_id) -> (primer, último cuadro) del vehículo."""
    # 4) ventanas de los clips: margen antes y después, duración acotada. Los controles toman una
    #    duración sorteada de las de las etiquetas, para que el largo del clip no delate su origen.
    durs = [pl["b"] - pl["a"] for pl in planes if "control" not in pl["origen"]]
    for pl in planes:
        if pl["origen"] == {"control"}:
            pl["b"] = pl["a"] + rng.choice(durs)
    for pl in planes:
        f = ind.loc[pl["video_id"]]
        fps, n = float(f.fps), int(f.cuadros)
        v0, v1 = vidas[(pl["video_id"], pl["track_id"])]   # el clip debe mostrar al vehículo marcado
        if pl["b"] < v0 or pl["a"] > v1:
            pl["a"], pl["b"] = v0, v1
        pl["a"], pl["b"] = max(pl["a"], v0), min(pl["b"], v1)
        a = pl["a"] - round(C["margen_antes_s"] * fps)
        b = min(pl["b"], pl["a"] + round(C["max_evento_s"] * fps)) + round(C["margen_despues_s"] * fps)
        b = max(b, a + round(C["min_clip_s"] * fps))
        pl["c0"], pl["c1"] = max(0, a), min(n - 1, b)
    rng.shuffle(planes)
    ancho = len(str(len(planes)))
    for i, pl in enumerate(planes, 1):
        pl["clip"] = f"clip_{i:0{ancho}d}"
    (entrega / "clips").mkdir(parents=True, exist_ok=True)

    # 5) una lectura secuencial por video, escribiendo todos los clips que la cubren
    for vid in sorted({pl["video_id"] for pl in planes}):
        f = ind.loc[vid]
        fps = float(f.fps)
        tr = pd.read_csv(p["rutas"]["salidas"] / f.sesion / vid / f"{vid}_trayectorias_vehiculos.csv.gz")
        mios = sorted([pl for pl in planes if pl["video_id"] == vid], key=lambda x: x["c0"])
        cajas = {pl["clip"]: tr[tr.track_id == pl["track_id"]].set_index("frame")[["x1", "y1", "x2", "y2"]] for pl in mios}
        activos, j, ultimo = {}, 0, max(pl["c1"] for pl in mios)
        w, h = C["resolucion"]
        esc = w / p["video"]["resolucion"][0]
        for k, cuadro in video.cuadros(f.ruta, mios[0]["c0"], ultimo + 1):
            while j < len(mios) and mios[j]["c0"] <= k:
                pl = mios[j]
                activos[pl["clip"]] = (pl, cv2.VideoWriter(str(entrega / "clips" / f"{pl['clip']}.mp4"),
                                                           cv2.VideoWriter_fourcc(*C["codec"]), fps, (w, h)), {"caja": None, "edad": 0})
                j += 1
            if not activos:
                continue
            base = cv2.resize(cuadro, (w, h), interpolation=cv2.INTER_AREA)
            for nombre in list(activos):
                pl, wr, st = activos[nombre]
                im = base.copy()
                cj = cajas[nombre]
                if k in cj.index:
                    st["caja"], st["edad"] = cj.loc[k].values * esc, 0
                else:
                    st["edad"] += 1
                if st["caja"] is not None and st["edad"] <= round(C["mantener_caja_s"] * fps):
                    x1, y1, x2, y2 = st["caja"].astype(int)
                    m = C["margen_caja_px"]
                    cv2.rectangle(im, (x1 - m, y1 - m), (x2 + m, y2 + m), tuple(C["color_bgr"]), C["grosor"])
                    # recuadro ampliado del vehículo marcado, arriba a la derecha (igual en todos los clips)
                    lado = max(C["zoom_min_px"], int(C["zoom_factor_caja"] * max(x2 - x1, y2 - y1)))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    a0, b0 = int(np.clip(cx - lado // 2, 0, w - lado)), int(np.clip(cy - lado // 2, 0, h - lado))
                    z = cv2.resize(im[b0:b0 + lado, a0:a0 + lado], (C["zoom_px"], C["zoom_px"]), interpolation=cv2.INTER_CUBIC)
                    zx, zy = w - C["zoom_px"] - C["margen_texto_px"], C["margen_texto_px"]
                    im[zy:zy + C["zoom_px"], zx:zx + C["zoom_px"]] = z
                    cv2.rectangle(im, (zx, zy), (zx + C["zoom_px"], zy + C["zoom_px"]), tuple(C["color_bgr"]), C["grosor"])
                txt = f"{nombre}   t = {(k - pl['c0']) / fps:5.1f} s"
                org = (C["margen_texto_px"], h - C["margen_texto_px"])
                cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, C["escala_texto"], (0, 0, 0), C["grosor"] + 2, cv2.LINE_AA)
                cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, C["escala_texto"], (255, 255, 255), C["grosor"] - 1, cv2.LINE_AA)
                wr.write(im)
                if k >= pl["c1"]:
                    wr.release()
                    del activos[nombre]
        for pl, wr, _ in activos.values():
            wr.release()
        print(f"{vid}: {len(mios)} clips")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotador", required=True)
    args = ap.parse_args()
    p = configuracion.cargar()
    ind = indice.leer(p)
    z = zonas.cargar(p["rutas"]["zonas"])
    carril, cruce = z[C["zona_carril"]], z[C["zona_crucero"]]
    sal = RAIZ / P["rutas"]["salidas"] / args.anotador
    etq = {inf: pd.read_csv(sal / f"etiquetas_{inf.lower()}.csv") for inf in ("G08", "G10", "G41", "M21")}
    cand = pd.read_csv(sal / "g10_candidatos_sin_anotar.csv")
    rng = random.Random(C["semilla"])
    planes, vidas = [], {}
    for vid in sorted(set().union(*[set(e.video_id) for e in etq.values()])):
        f = ind.loc[vid]
        fps = float(f.fps)
        tr = pd.read_csv(p["rutas"]["salidas"] / f.sesion / vid / f"{vid}_trayectorias_vehiculos.csv.gz")
        tr = tr.sort_values(["track_id", "frame"])
        tr["vel"] = tr.groupby("track_id", group_keys=False).apply(velocidad)
        tr["px"], tr["py"] = (tr.x1 + tr.x2) / 2, tr.y2
        tr["en_carril"] = zonas.dentro(tr.px, tr.py, carril)
        por_id = dict(tuple(tr.groupby("track_id")))
        vidas.update({(vid, tid): (int(t.frame.min()), int(t.frame.max())) for tid, t in por_id.items()})
        # 1) intervalos por vehículo, de las etiquetas y de los candidatos
        filas = []
        for inf, e in etq.items():
            for r in e[(e.video_id == vid) & (e.estado == "ok")].itertuples():
                t = por_id.get(int(r.track_id))
                if t is None:
                    continue
                if pd.notna(r.k_ini):
                    a, b = int(r.k_ini), int(r.k_fin)
                else:   # G10 sin horas: el tramo del vehículo dentro del carril (o toda su vida)
                    x = t[t.en_carril] if t.en_carril.any() else t
                    a, b = int(x.frame.min()), int(x.frame.max())
                filas.append((int(r.track_id), a, b, "etiqueta", inf))
        for r in cand[cand.video_id == vid].itertuples():
            filas.append((int(r.track_id), int(r.k_entrada), int(r.k_salida), "candidato", "G10?"))
        # 2) controles: vehículos sin ninguna fila, con situaciones parecidas
        usados = {x[0] for x in filas}
        libres = [t for tid, t in por_id.items() if tid not in usados and len(t) >= C["min_cuadros_control"]]
        rng.shuffle(libres)
        n_ctrl = round(C["fraccion_controles"] / (1 - C["fraccion_controles"]) * len({(x[0]) for x in filas}))
        tipos = {"detenido_antes_del_crucero": [], "junto_al_carril": [], "detenido_sin_evento": []}
        dist = lambda pol, t: np.array([cv2.pointPolygonTest(pol, (float(x), float(y)), True)  # noqa: E731
                                        for x, y in zip(t.px, t.py)])   # > 0 dentro, < 0 fuera (px)
        for t in libres:
            q = t[t.vel < C["umbral_detenido"]]
            if t.en_carril.any():
                continue
            if len(q) >= C["min_cuadros_detenido"]:
                dc = dist(cruce, q)
                if (dc < 0).all() and (-dc).min() <= C["distancia_crucero_px"]:
                    tipos["detenido_antes_del_crucero"].append((t, int(q.frame.min()), int(q.frame.max())))
                else:
                    tipos["detenido_sin_evento"].append((t, int(q.frame.min()), int(q.frame.max())))
            elif (-dist(carril, t)).min() <= C["distancia_carril_px"]:
                tipos["junto_al_carril"].append((t, int(t.frame.min()), int(t.frame.max())))
        k = 0
        while k < n_ctrl and any(tipos.values()):
            for nombre, lista in tipos.items():
                if lista and k < n_ctrl:
                    t, a, b = lista.pop()
                    filas.append((int(t.track_id.iloc[0]), a, b, "control", nombre))
                    k += 1
        # 3) un clip por vehículo y tramo: se juntan los intervalos cercanos del mismo vehículo
        df = pd.DataFrame(filas, columns=["track_id", "a", "b", "origen", "infraccion"]).sort_values(["track_id", "a"])
        junta = round(C["juntar_si_separacion_s"] * fps)
        for tid, g in df.groupby("track_id"):
            grupo = None
            for r in g.itertuples():
                if grupo and r.a <= grupo["b"] + junta:
                    grupo["b"] = max(grupo["b"], r.b)
                    grupo["origen"].add(r.origen)
                    grupo["infracciones"].add(r.infraccion)
                    continue
                if grupo:
                    planes.append(grupo)
                grupo = {"video_id": vid, "track_id": tid, "a": r.a, "b": r.b, "origen": {r.origen}, "infracciones": {r.infraccion}}
            planes.append(grupo)
    entrega = Path(P["rutas"]["entrega_validacion"])
    escribir_clips(planes, vidas, entrega, p, ind, rng)

    # 6) clave (no se entrega) y base de la hoja del validador
    clave = pd.DataFrame([{"clip": pl["clip"], "video_id": pl["video_id"], "track_id": pl["track_id"],
                           "origen": "+".join(sorted(pl["origen"])), "infracciones_anotadas": "+".join(sorted(pl["infracciones"])),
                           "cuadro_inicio_clip": pl["c0"], "cuadro_fin_clip": pl["c1"],
                           "hora_inicio_clip": video.Reloj(ind.loc[pl["video_id"]]).hora(pl["c0"]),
                           "duracion_s": round((pl["c1"] - pl["c0"] + 1) / float(ind.loc[pl["video_id"]].fps), 1)}
                          for pl in planes]).sort_values("clip")
    clave.to_csv(sal / "clave_clips.csv", index=False)
    clave[["clip", "duracion_s"]].assign(archivo=lambda x: "clips/" + x["clip"] + ".mp4").to_csv(entrega / "validacion.csv", index=False)
    print(clave.origen.value_counts().to_string())


if __name__ == "__main__":
    main()
