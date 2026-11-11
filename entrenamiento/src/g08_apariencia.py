"""Matriz de apariencia de G08: un descriptor de 1536 valores por instante.

Cada recorte (caja ampliada 0,7 alturas, ya guardado por el procesamiento) se lleva a 224 × 224 y pasa
por DINOv2 ViT-S/14 congelado (timm). De la última capa se guardan cuatro vectores de 384: el token de
clase, la media de los parches con centro dentro de la caja del automóvil, y la media de los parches
exteriores a la izquierda y a la derecha de su centro. Si una región queda vacía (la caja toca el borde del
recorte) se usa la media de todos los parches.
Se guarda en float16; al final se compara con float32 en una muestra (coseno mínimo).
Corre en la GPU con torch-directml y se puede reanudar (el progreso se guarda cada CADA_GUARDAR lotes).

Entradas: instantes.csv.gz (g08_ventanas.py) y los recortes del procesamiento.
Salida (rutas.rasgos/g08): apariencia.npy (float16, N × 4 × 384, fila = id_instante) y apariencia.json.
Reutilización: si existen apariencia_previa.npy e instantes_previos.csv.gz (una extracción anterior y su
índice), las filas cuyo recorte ya se procesó se copian, emparejadas por la ruta del recorte, y solo se
extraen las nuevas. Los rasgos de un recorte no dependen de los demás, así que el resultado es idéntico al
de extraer todo.
Uso: python src/g08_apariencia.py [--limite N] [--reanudar] [--codificador NOMBRE] [--nombre ARCHIVO]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

import comun as c

G = c.P["g08"]
VECTORES = ["cls", "caja", "izquierda", "derecha"]
CADA_GUARDAR = 50


def cargar(ruta, caja, lado):
    im = cv2.imread(ruta)
    if im is None:
        raise FileNotFoundError(ruta)
    h, w = im.shape[:2]
    im = cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA if max(h, w) > lado else cv2.INTER_LINEAR)
    x1, y1, x2, y2 = caja
    return im[:, :, ::-1], np.array([x1 * lado / w, y1 * lado / h, x2 * lado / w, y2 * lado / h], np.float32)


def pesos_regiones(cajas, grid, parche):
    """(B, 3, P): pesos de promedio de los parches dentro, a la izquierda y a la derecha (Apéndice D)."""
    ctr = (np.arange(grid) + 0.5) * parche
    cx, cy = np.meshgrid(ctr, ctr)
    cx, cy = cx.ravel()[None], cy.ravel()[None]
    b = cajas[:, :, None]
    dentro = (cx >= b[:, 0]) & (cx <= b[:, 2]) & (cy >= b[:, 1]) & (cy <= b[:, 3])
    centro = (b[:, 0] + b[:, 2]) / 2
    m = np.stack([dentro, ~dentro & (cx < centro), ~dentro & (cx >= centro)], 1).astype(np.float32)
    vacias = m.sum(2) == 0
    m[vacias] = 1.0
    return m / m.sum(2, keepdims=True), int(vacias.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo los primeros N instantes (prueba)")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--codificador", default=None, help="otro codificador de timm (p. ej. vit_base_patch14_dinov2.lvd142m)")
    ap.add_argument("--nombre", default="apariencia", help="nombre del archivo de rasgos (apariencia_b14, etc.)")
    a = ap.parse_args()
    import torch
    import timm
    import compat_directml   # noqa: F401  (parche de DirectML del procesamiento; da `dispositivo`)

    d = c.R["rasgos"] / "g08"
    ins = pd.read_csv(d / "instantes.csv.gz", usecols=["id_instante", "ruta_recorte", "x1", "y1", "x2", "y2", "cx1", "cy1"])
    if a.limite:
        ins = ins.head(a.limite)
    N, lado = len(ins), G["lado"]
    dev = compat_directml.dispositivo(G["dispositivo"])
    cod = a.codificador or G["codificador"]
    modelo = timm.create_model(cod, pretrained=True, num_classes=0, img_size=lado).eval().to(dev)
    cfg = modelo.pretrained_cfg
    media, desv = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
    parche = modelo.patch_embed.patch_size[0]
    grid, pref, D = lado // parche, modelo.num_prefix_tokens, modelo.num_features
    nombre = a.nombre + ("_prueba" if a.limite else "")
    ruta, prog = d / f"{nombre}.npy", d / f"{nombre}_progreso.json"
    if a.reanudar and ruta.exists() and prog.exists():
        arr = np.load(ruta, mmap_mode="r+")
        inicio = json.loads(prog.read_text())["hasta"]
        assert arr.shape == (N, len(VECTORES), D)
    else:
        arr = np.lib.format.open_memmap(ruta, mode="w+", dtype=np.float16, shape=(N, len(VECTORES), D))
        inicio = 0
    rutas = ins.ruta_recorte.tolist()
    pendientes = np.arange(N)
    prev_npy, prev_ins = d / f"{a.nombre}_previa.npy", d / "instantes_previos.csv.gz"
    if prev_npy.exists() and prev_ins.exists() and not a.limite and inicio == 0:
        ant = np.load(prev_npy, mmap_mode="r")
        fila_ant = pd.Series(np.arange(len(ant)), index=pd.read_csv(prev_ins, usecols=["ruta_recorte"]).ruta_recorte)
        f = fila_ant.reindex(rutas).to_numpy()
        hay = ~np.isnan(f)
        orden = np.argsort(f[hay])
        destino, origen = np.nonzero(hay)[0][orden], f[hay][orden].astype(int)
        for k in range(0, len(destino), 50000):
            arr[destino[k:k + 50000]] = ant[origen[k:k + 50000]]
        arr.flush()
        pendientes = np.nonzero(~hay)[0]
        print(f"reutilizados {int(hay.sum())} de {N}; faltan {len(pendientes)}", flush=True)
    cajas = np.c_[ins.x1 - ins.cx1, ins.y1 - ins.cy1, ins.x2 - ins.cx1, ins.y2 - ins.cy1].astype(np.float32)
    B = G["lote"]

    def lote(i0):
        ii = list(pendientes[i0:i0 + B])
        return ii, list(pool.map(lambda i: cargar(rutas[i], cajas[i], lado), ii))

    def codificar(datos):
        x = (np.stack([q[0] for q in datos]).astype(np.float32) / 255.0 - media) / desv
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
        m, v = pesos_regiones(np.stack([q[1] for q in datos]), grid, parche)
        tok = modelo.forward_features(x.to(dev))
        res = torch.cat([tok[:, :1], torch.bmm(torch.from_numpy(m).to(dev), tok[:, pref:])], 1).cpu().numpy()
        return res, v

    t0, hechos, vacias = time.time(), 0, 0
    with ThreadPoolExecutor(G["hilos_lectura"]) as pool, torch.no_grad():
        pendientes = pendientes[pendientes >= inicio]
        P_ = len(pendientes)
        fut = pool.submit(lote, 0) if P_ else None
        for k, i0 in enumerate(range(0, P_, B)):
            ii, datos = fut.result()
            if i0 + B < P_:
                fut = pool.submit(lote, i0 + B)
            res, v = codificar(datos)
            if not np.isfinite(res).all():
                raise RuntimeError(f"rasgos no finitos en el lote que empieza en {i0}")
            arr[ii] = res.astype(np.float16)
            hechos, vacias = hechos + len(ii), vacias + v
            if k % CADA_GUARDAR == 0:
                arr.flush()
                prog.write_text(json.dumps({"hasta": int(ii[-1]) + 1}))
                el = time.time() - t0
                print(f"{i0 + len(ii)}/{P_}  {1000 * el / hechos:.1f} ms/recorte  faltan {(P_ - i0 - len(ii)) * el / hechos / 60:.0f} min",
                      flush=True)
        arr.flush()
        prog.write_text(json.dumps({"hasta": N}))
        muestra = list(range(0, min(N, 32)))
        ref, _ = codificar([cargar(rutas[i], cajas[i], lado) for i in muestra])
    g = np.asarray(arr[muestra], np.float32)
    cos = (ref * g).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(g, axis=-1))
    meta = {"codificador": cod, "lado": lado, "parche": parche, "grid": grid, "dim": D, "vectores": VECTORES,
            "instantes": N, "dtype": "float16", "regiones_vacias_reemplazadas": vacias,
            "ms_por_recorte": round(1000 * (time.time() - t0) / max(1, hechos), 2), "minutos": round((time.time() - t0) / 60, 1),
            "coseno_min_float16_vs_float32": round(float(cos.min()), 5)}
    c.guardar_json(meta, d / f"{nombre}.json")
    print(meta)


if __name__ == "__main__":
    main()
