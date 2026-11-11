"""Etapa 2: seguimiento. De las detecciones guardadas a trayectorias, en CPU y sin leer el video.

Por cuadro y por familia (vehículos: auto y bus; personas):
  1. Supresión de duplicados por IoU.
  2. Descarte de personas que son ocupantes de un vehículo (vistas por el parabrisas).
  3. TrackTrack de Ultralytics, alimentado con las detecciones (img=None: sin compensación de
     movimiento, porque la cámara es fija). Se llama en todos los cuadros, también en los
     vacíos, para que las pistas perdidas envejezcan con el tiempo real.
Cada fila conserva la caja y la confianza de la detección cruda asociada, no la del filtro
de Kalman. track_id es local al video: 1, 2, 3... en orden de primera aparición. La clave
única entre videos es (video_id, track_id).

Salida: <video_id>_trayectorias_vehiculos.csv.gz, <video_id>_trayectorias_personas.csv.gz,
        <video_id>_seguimiento.json
Entrada: <video_id>_detecciones.csv.gz y su .json.

Uso: se invoca desde correr.py (python src/correr.py --videos <video_id> --etapas seguir).
"""
import json
import time

import numpy as np
import pandas as pd
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

import configuracion


def iou_matriz(a, b):
    """IoU entre cajas xyxy, (N,4) x (M,4) -> (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def suprimir_duplicados(cajas, conf, umbral):
    """Supresión voraz de no máximos; devuelve los índices conservados."""
    if len(cajas) < 2:
        return np.arange(len(cajas))
    iou = iou_matriz(cajas, cajas)
    suprimida, conservar = np.zeros(len(cajas), bool), []
    for i in np.argsort(-conf):
        if not suprimida[i]:
            conservar.append(i)
            suprimida |= iou[i] >= umbral
    return np.array(conservar, int)


def ocupantes(per, veh, ioa_min, margen_pie):
    """True para cada persona contenida en un vehículo y con los pies por encima de su base."""
    if len(per) == 0 or len(veh) == 0:
        return np.zeros(len(per), bool)
    ix = np.clip(np.minimum(per[:, None, 2], veh[None, :, 2]) - np.maximum(per[:, None, 0], veh[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(per[:, None, 3], veh[None, :, 3]) - np.maximum(per[:, None, 1], veh[None, :, 1]), 0, None)
    area_p = ((per[:, 2] - per[:, 0]) * (per[:, 3] - per[:, 1]))[:, None]
    ioa = ix * iy / np.maximum(area_p, 1e-9)
    pies_arriba = per[:, None, 3] < veh[None, :, 3] - margen_pie * (veh[None, :, 3] - veh[None, :, 1])
    return ((ioa >= ioa_min) & pies_arriba).any(1)


def preparar(det, n, cfg):
    """Detecciones -> {familia: lista por cuadro de arreglos (k,6) x1,y1,x2,y2,conf,clase}."""
    det = det.sort_values("frame", kind="stable")
    arr = det[["x1", "y1", "x2", "y2", "conf", "clase"]].values.astype(np.float32)
    inicios = np.searchsorted(det.frame.values, np.arange(n + 1))
    salida = {"vehiculos": [None] * n, "personas": [None] * n}
    cont = {"duplicados_vehiculo": 0, "duplicados_persona": 0, "personas_ocupantes": 0}
    oc = cfg["ocupantes"]
    for k in range(n):
        a = arr[inicios[k]:inicios[k + 1]]
        v = a[np.isin(a[:, 5], cfg["clases_vehiculo"])]
        q = a[np.isin(a[:, 5], cfg["clases_persona"])]
        kv = suprimir_duplicados(v[:, :4], v[:, 4], cfg["nms_iou"])
        kq = suprimir_duplicados(q[:, :4], q[:, 4], cfg["nms_iou"])
        cont["duplicados_vehiculo"] += len(v) - len(kv)
        cont["duplicados_persona"] += len(q) - len(kq)
        v, q = v[kv], q[kq]
        ref = v[v[:, 4] >= oc["conf_vehiculo"], :4]
        es_ocupante = ocupantes(q[:, :4], ref, oc["ioa_min"], oc["margen_pie"])
        cont["personas_ocupantes"] += int(es_ocupante.sum())
        salida["vehiculos"][k], salida["personas"][k] = v, q[~es_ocupante]
    return salida, cont


def crear_tracker(cfg):
    args = IterableSimpleNamespace(**YAML.load(check_yaml(cfg["tracker"])))
    for clave, valor in cfg["tracker_ajustes"].items():
        setattr(args, clave, valor)
    return TRACKER_MAP[args.tracker_type](args=args)


def rastrear(por_cuadro, cfg, forma):
    tracker = crear_tracker(cfg)
    filas = []
    for k, a in enumerate(por_cuadro):
        for t in tracker.update(Boxes(a, forma), img=None):
            x1, y1, x2, y2, conf, cls = a[int(t[7])]
            filas.append((k, int(t[4]), int(cls), float(conf), x1, y1, x2, y2))
    tr = pd.DataFrame(filas, columns=["frame", "id_tracker", "clase", "conf", "x1", "y1", "x2", "y2"])
    # El contador interno de Ultralytics se comparte entre instancias y deja huecos: se renumera.
    orden = tr.groupby("id_tracker").frame.min().sort_values(kind="stable").index
    tr.insert(1, "track_id", tr.id_tracker.map({v: i + 1 for i, v in enumerate(orden)}).astype(np.int64))
    return tr.drop(columns="id_tracker")


def correr(p, fila, carpeta, log=print):
    cfg, vid = p["seguimiento"], fila["video_id"]
    meta_det = json.loads((carpeta / f"{vid}_detecciones.json").read_text(encoding="utf-8"))
    if meta_det["estado"] != "completo":
        raise SystemExit(f"{vid}: la detección no está completa ({meta_det['estado']})")
    det = pd.read_csv(carpeta / f"{vid}_detecciones.csv.gz")
    n = int(meta_det["cuadros_objetivo"])
    forma = (p["video"]["resolucion"][1], p["video"]["resolucion"][0])   # (alto, ancho)
    t0 = time.perf_counter()
    por_familia, cont = preparar(det, n, cfg)
    res = {fam: rastrear(por_familia[fam], cfg, forma) for fam in ("vehiculos", "personas")}
    for fam, tr in res.items():
        tmp = carpeta / f"{vid}_trayectorias_{fam}.tmp.csv.gz"
        tr.round({"conf": 3, "x1": 1, "y1": 1, "x2": 1, "y2": 1}).to_csv(tmp, index=False)
        tmp.replace(carpeta / f"{vid}_trayectorias_{fam}.csv.gz")
    meta = {"firma": configuracion.firma(cfg, meta_det["firma"]), "estado": "completo", "parametros": cfg,
            "cuadros": n, **cont, "tracklets_vehiculos": int(res["vehiculos"].track_id.nunique()),
            "tracklets_personas": int(res["personas"].track_id.nunique()),
            "minutos": round((time.perf_counter() - t0) / 60, 1)}
    (carpeta / f"{vid}_seguimiento.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{vid} seguimiento: {meta['tracklets_vehiculos']} trayectorias de vehículo, "
        f"{meta['tracklets_personas']} de persona, {meta['minutos']} min")
