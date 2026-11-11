"""Carga config/parametros.yaml y resuelve sus rutas relativas a la carpeta del módulo.

Entrada: config/parametros.yaml. Salida: diccionario de parámetros y huella (firma) de un conjunto de valores.
Uso: import configuracion; p = configuracion.cargar()
"""
import hashlib
import json
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
PARAMETROS = RAIZ / "config" / "parametros.yaml"


def cargar(ruta=None):
    ruta = Path(ruta) if ruta else PARAMETROS
    p = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    p["rutas"] = {k: Path(v) if Path(v).is_absolute() else RAIZ / v for k, v in p["rutas"].items()}
    return p


def firma(*partes):
    """Huella corta de un conjunto de parámetros, para saber si una salida se hizo con ellos."""
    texto = json.dumps(partes, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:12]
