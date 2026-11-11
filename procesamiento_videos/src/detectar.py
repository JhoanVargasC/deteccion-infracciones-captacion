"""Etapa 1: detección. Guarda todas las detecciones de cada cuadro, con reanudación.

El detector ve solo el rectángulo envolvente del área de reconocimiento, con lo exterior al
polígono pintado de gris. Las cajas se guardan en coordenadas del cuadro completo.

Salida en la carpeta del video:
  <video_id>_detecciones.csv.gz   frame, clase, conf, x1, y1, x2, y2
  <video_id>_detecciones.json     parámetros, estado y tiempos

Reanudación: si la corrida se corta, la siguiente sigue desde el último cuadro guardado
completo. Si el archivo existente se hizo con otros parámetros, se detiene en vez de mezclar.

Uso: se invoca desde correr.py (python src/correr.py --videos <video_id> --etapas detectar).
"""
import gzip
import json
import time
import zlib

import cv2
import numpy as np

import configuracion
import video
import zonas

CABECERA = "frame,clase,conf,x1,y1,x2,y2\n"
CAMPOS = CABECERA.count(",") + 1


def ultimo_cuadro_guardado(ruta):
    """Último cuadro completo del CSV comprimido; repara el archivo si quedó truncado."""
    if not ruta.exists():
        return -1
    lineas, truncado = [], False
    try:
        with gzip.open(ruta, "rt") as f:
            next(f, None)
            for ln in f:
                if ln.endswith("\n") and ln.count(",") == CAMPOS - 1:
                    lineas.append(ln)
                else:
                    truncado = True
                    break
    except (EOFError, OSError, zlib.error):
        truncado = True
    if not lineas:
        ruta.unlink()
        return -1
    ultimo = int(lineas[-1].split(",", 1)[0])
    if truncado:   # el último cuadro pudo quedar a medias: se descarta y se reescribe el archivo
        lineas = [ln for ln in lineas if int(ln.split(",", 1)[0]) < ultimo]
        ultimo -= 1
        tmp = ruta.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            f.write(CABECERA)
            f.writelines(lineas)
        tmp.replace(ruta)
    return ultimo


def correr(p, fila, carpeta, modelo, disp, limite=None, log=print):
    cfg = p["deteccion"]
    poligono, (zx, zy, zw, zh) = zonas.area(p)
    dentro = zonas.mascara(poligono)
    fondo = np.full((zh, zw, 3), cfg["relleno_mascara"], np.uint8)
    n = min(int(fila["cuadros"]), limite) if limite else int(fila["cuadros"])
    vid = fila["video_id"]
    ruta, ruta_meta = carpeta / f"{vid}_detecciones.csv.gz", carpeta / f"{vid}_detecciones.json"
    huella = configuracion.firma(cfg, poligono.tolist(), fila["archivo"], p["rutas"]["modelo"].name)

    if ruta_meta.exists():
        previa = json.loads(ruta_meta.read_text(encoding="utf-8"))
        if previa.get("firma") == huella and previa.get("estado") == "completo" and (not limite or previa.get("cuadros_objetivo") == n):
            log(f"{vid} detección: ya estaba completa")
            return
    desde = ultimo_cuadro_guardado(ruta) + 1
    if desde > 0:
        previa = json.loads(ruta_meta.read_text(encoding="utf-8")).get("firma") if ruta_meta.exists() else None
        if previa != huella:
            raise SystemExit(f"{ruta.name} se hizo con otros parámetros. Muévelo o bórralo antes de seguir.")
    if desde >= n:
        log(f"{vid} detección: ya estaba hecha hasta el cuadro {desde - 1}")
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        meta.update({"estado": "completo", "cuadros_objetivo": n, "ultimo_cuadro": n - 1})
        ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    log(f"{vid} detección: cuadros {desde} a {n - 1}" + (" (reanuda)" if desde else ""))
    carpeta.mkdir(parents=True, exist_ok=True)
    meta = {"firma": huella, "parametros": cfg, "video": fila["archivo"], "modelo": p["rutas"]["modelo"].name,
            "rectangulo_xywh": [zx, zy, zw, zh], "estado": "en_curso", "reanudado_desde": desde}
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    t = {"decodificacion": 0.0, "deteccion": 0.0, "escritura": 0.0}
    salida = gzip.open(ruta, "at" if desde else "wt")
    if not desde:
        salida.write(CABECERA)
    bufer, ultimo, n_det, t_ini, fin_del_video = [], desde - 1, 0, time.time(), False
    cada = p["video"]["registrar_cada_cuadros"]
    desplaz = np.array([zx, zy, zx, zy])
    try:
        t0 = time.perf_counter()
        for k, cuadro in video.cuadros(fila["ruta"], desde, n):
            if cfg["enmascarar_fuera_del_area"]:
                entrada = fondo.copy()
                cv2.copyTo(cuadro[zy:zy + zh, zx:zx + zw], dentro, entrada)
            else:
                entrada = np.ascontiguousarray(cuadro[zy:zy + zh, zx:zx + zw])
            t1 = time.perf_counter()
            r = modelo.predict(entrada, device=disp, imgsz=cfg["imgsz"], conf=cfg["conf_guardado"],
                               classes=cfg["clases"], nms=cfg["nms_ultralytics"], verbose=False)[0]
            cajas = r.boxes.xyxy.cpu().numpy() + desplaz
            for (x1, y1, x2, y2), c, cf in zip(cajas, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                bufer.append(f"{k},{int(c)},{cf:.3f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}\n")
            t2 = time.perf_counter()
            ultimo = k            # el cuadro entra al búfer completo; se escribe por bloques
            if (k + 1) % cfg["bloque_escritura"] == 0:
                salida.writelines(bufer)
                salida.flush()
                n_det, bufer = n_det + len(bufer), []
            t3 = time.perf_counter()
            t["decodificacion"] += t1 - t0
            t["deteccion"] += t2 - t1
            t["escritura"] += t3 - t2
            t0 = t3
            if (k + 1) % cada == 0:
                hechos = k + 1 - desde
                vel = hechos / (time.time() - t_ini)
                log(f"{vid} detección: cuadro {k + 1}/{n} ({100 * (k + 1) / n:.1f} %), {vel:.1f} cuadros/s, "
                    f"faltan {(n - k - 1) / vel / 60:.0f} min")
        # El generador termina en el cuadro n o al acabarse el video. Si se acabó antes, el
        # conteo del índice era mayor que el real: cuenta lo leído de verdad.
        fin_del_video = not limite and ultimo + 1 < n
    finally:
        salida.writelines(bufer)
        salida.close()
        n_det += len(bufer)
        hechos = max(1, ultimo + 1 - desde)
        if fin_del_video:
            log(f"{vid} detección: el video terminó en el cuadro {ultimo} y el índice decía {n} cuadros")
            n = ultimo + 1
        meta.update({"estado": "completo" if ultimo + 1 >= n else f"parcial: cuadros 0 a {ultimo}",
                     "cuadros_objetivo": n, "ultimo_cuadro": ultimo, "detecciones_esta_corrida": n_det,
                     "ms_por_cuadro": {kk: round(1000 * v / hechos, 1) for kk, v in t.items()},
                     "minutos_esta_corrida": round((time.time() - t_ini) / 60, 1)})
        ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"{vid} detección: {meta['estado']} en {meta['minutos_esta_corrida']} min, {json.dumps(meta['ms_por_cuadro'])}")
