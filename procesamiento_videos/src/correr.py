"""Orquestador: corre las etapas sobre uno o varios videos del índice, con registro.

Entrada: datos/indice_videos.csv (ver indice.py) y config/parametros.yaml.
Salida: una carpeta por video con los archivos de cada etapa.

Uso: python src/correr.py --sesion 2026-04-06_manana
     python src/correr.py --videos 2026-04-06_tarde_h3 --limite-segundos 180 --salida prueba
     python src/correr.py --videos 2026-04-01_manana_h2 --etapas seguir recortar

Cada video va a <salidas>/<sesion>/<video_id>/. Al relanzar, la detección reanuda donde
quedó y las etapas ya completas con los mismos parámetros se saltan. El registro va a
registros/AAAAMMDD-HHMMSS_<etapas>_<videos>.log y cada corrida suma una línea al cuaderno.
"""
import compat_directml  # noqa: F401  (debe importarse antes que Ultralytics)

import argparse
import ctypes
import sys
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import configuracion
import detectar
import indice
import recortar
import seguir

ETAPAS = ("detectar", "seguir", "recortar")
ES_CONTINUO, ES_SISTEMA_REQUERIDO = 0x80000000, 0x00000001   # banderas de la API de Windows


def registro(p, etiqueta):
    p["rutas"]["registros"].mkdir(parents=True, exist_ok=True)
    ruta = p["rutas"]["registros"] / f"{datetime.now():%Y%m%d-%H%M%S}_{etiqueta}.log"
    log = logging.getLogger("correr")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(ruta, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log, ruta


def completa(carpeta, vid, etapa):
    """True si la etapa ya terminó en esta carpeta."""
    nombre = {"seguir": "seguimiento", "recortar": "recortes"}[etapa]
    ruta = carpeta / f"{vid}_{nombre}.json"
    return ruta.exists() and json.loads(ruta.read_text(encoding="utf-8")).get("estado") == "completo"


def esperar_deteccion(p, fila, carpeta, log):
    """Espera a que otro proceso termine la detección del video (modo en paralelo)."""
    a, vid = p["almacenamiento"], fila["video_id"]
    ruta = carpeta / f"{vid}_detecciones.json"
    limite = time.time() + a["espera_max_horas"] * 3600
    avisado = False
    while time.time() < limite:
        if ruta.exists():
            try:
                m = json.loads(ruta.read_text(encoding="utf-8"))
                if m.get("estado") == "completo":
                    return
            except json.JSONDecodeError:
                pass   # el otro proceso está escribiendo el archivo
        if not avisado:
            log(f"{vid}: esperando que termine su detección")
            avisado = True
        time.sleep(a["espera_segundos"])
    raise SystemExit(f"{vid}: la detección no terminó en {a['espera_max_horas']} h")


def libre_gb(ruta):
    ruta = Path(ruta)
    while not ruta.exists():
        ruta = ruta.parent
    return shutil.disk_usage(ruta).free / 1e9


def mover_video(p, fila, carpeta, log):
    """Mueve el video anotado a la salida secundaria si cabe; si no, lo deja donde está."""
    a, vid = p["almacenamiento"], fila["video_id"]
    origen = carpeta / f"{vid}_video_anotado.mp4"
    if not (a["mover_video_anotado"] and origen.exists()):
        return
    gb = origen.stat().st_size / 1e9
    destino = p["rutas"]["salidas_secundaria"] / fila["sesion"] / vid / origen.name
    if gb >= a["max_archivo_gb_secundario"]:
        log(f"{vid}: el video anotado pesa {gb:.2f} GB y no cabe en FAT32; se queda en {origen.parent}")
        return
    if not Path(p["rutas"]["salidas_secundaria"].anchor).exists() or libre_gb(destino) - gb < a["libre_minimo_gb_secundario"]:
        log(f"{vid}: sin espacio o sin unidad secundaria; el video anotado se queda en {origen.parent}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".parcial")
    shutil.copyfile(origen, tmp)
    if tmp.stat().st_size != origen.stat().st_size:
        tmp.unlink()
        raise SystemExit(f"{vid}: la copia del video anotado quedó incompleta")
    tmp.replace(destino)
    origen.unlink()
    ruta_meta = carpeta / f"{vid}_recortes.json"
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    meta["ubicacion_video_anotado"] = str(destino)
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{vid}: video anotado ({gb:.2f} GB) movido a {destino}")


def anotar_cuaderno(p, linea):
    ruta = p["rutas"]["cuaderno"]
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("# Cuaderno de corridas\n\n| Inicio | Fin | Etapas | Videos | Límite | Salida | Resultado | Registro |\n"
                        "|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with ruta.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--videos", nargs="+")
    grupo.add_argument("--sesion")
    ap.add_argument("--etapas", nargs="+", choices=ETAPAS, default=list(ETAPAS))
    ap.add_argument("--limite-segundos", type=float, default=None, help="solo los primeros N s de cada video (pruebas)")
    ap.add_argument("--salida", default=None, help="carpeta base distinta de salidas/ (pruebas)")
    ap.add_argument("--forzar", action="store_true", help="rehace seguimiento y recortes aunque estén completos")
    ap.add_argument("--esperar", action="store_true",
                    help="sin la etapa detectar: espera a que otro proceso termine la detección de cada video")
    a = ap.parse_args()

    p = configuracion.cargar(a.config)
    if sys.platform == "win32":   # que Windows no suspenda el equipo mientras este proceso corre
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUO | ES_SISTEMA_REQUERIDO)
    ind = indice.leer(p)
    ids = a.videos or list(ind[ind.sesion == a.sesion].video_id)
    faltan = [v for v in ids if v not in ind.index]
    if not ids or faltan:
        raise SystemExit(f"Videos que no están en el índice: {faltan or a.sesion}")
    base = Path(a.salida) if a.salida else p["rutas"]["salidas"]
    base = base if base.is_absolute() else configuracion.RAIZ / base
    etiqueta = "-".join(a.etapas) + "_" + (a.sesion or (ids[0] if len(ids) == 1 else f"{ids[0]}_y{len(ids) - 1}"))
    log, ruta_log = registro(p, etiqueta)
    inicio = datetime.now()
    log.info(f"Inicio | etapas {a.etapas} | videos {ids} | límite {a.limite_segundos} s | salida {base}")
    log.info(f"Parámetros: {configuracion.PARAMETROS if not a.config else a.config} "
             f"(firma {configuracion.firma(p['deteccion'], p['seguimiento'], p['recortes'], p['video_anotado'])})")

    modelo = disp = None
    resultado = "completo"
    try:
        for vid in ids:
            fila = ind.loc[vid].to_dict()
            if str(fila.get("hora_cuadro0", "nan")) == "nan":
                raise SystemExit(f"{vid}: falta la hora de inicio medida en el índice (datos/lecturas_reloj.csv)")
            carpeta = base / fila["sesion"] / vid
            limite = round(a.limite_segundos * float(fila["fps"])) if a.limite_segundos else None
            if "detectar" in a.etapas:
                if modelo is None:
                    from ultralytics import YOLO
                    modelo = YOLO(str(p["rutas"]["modelo"]))
                    disp = compat_directml.dispositivo(p["deteccion"]["dispositivo"])
                detectar.correr(p, fila, carpeta, modelo, disp, limite, log.info)
            if a.esperar and "detectar" not in a.etapas:
                esperar_deteccion(p, fila, carpeta, log.info)
            if "seguir" in a.etapas:
                if a.forzar or "detectar" in a.etapas or not completa(carpeta, vid, "seguir"):
                    seguir.correr(p, fila, carpeta, log.info)
                else:
                    log.info(f"{vid} seguimiento: ya estaba completo")
            if "recortar" in a.etapas:
                if a.forzar or "seguir" in a.etapas or not completa(carpeta, vid, "recortar"):
                    al, libre = p["almacenamiento"], libre_gb(carpeta)
                    horas = float(fila["cuadros"]) / float(fila["fps"]) / 3600
                    pedido = max(al["libre_minimo_gb_absoluto"], al["libre_minimo_gb_principal"] * horas)
                    if libre < pedido:
                        raise SystemExit(f"{vid}: quedan {libre:.1f} GB libres y se piden {pedido:.1f} GB; se detienen los recortes")
                    recortar.correr(p, fila, carpeta, log.info)
                    if not a.salida:
                        mover_video(p, fila, carpeta, log.info)
                else:
                    log.info(f"{vid} recortes: ya estaban completos")
    except BaseException as e:
        resultado = f"interrumpido: {type(e).__name__}: {e}"
        log.exception(resultado)
        raise
    finally:
        fin = datetime.now()
        log.info(f"Fin | {resultado} | {(fin - inicio).total_seconds() / 60:.1f} min")
        anotar_cuaderno(p, f"| {inicio:%Y-%m-%d %H:%M} | {fin:%Y-%m-%d %H:%M} | {' '.join(a.etapas)} | {' '.join(ids)} | "
                           f"{a.limite_segundos or 'completo'} | {base.relative_to(configuracion.RAIZ) if base.is_relative_to(configuracion.RAIZ) else base} | "
                           f"{resultado} | {ruta_log.name} |")


if __name__ == "__main__":
    main()
