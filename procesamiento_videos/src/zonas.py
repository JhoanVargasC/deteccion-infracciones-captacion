"""Zonas de la escena: carga desde zonas.json, rectángulo envolvente, máscara y pertenencia de puntos.

Uso: import zonas; poligono, rect = zonas.area(p)
"""
import json

import cv2
import numpy as np


def cargar(ruta):
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {z["label"]: np.array(z["points"], dtype=np.int32) for z in datos["zones"]}


def area(p):
    """Polígono del área de reconocimiento y su rectángulo (x, y, ancho, alto), comprobado
    contra el valor esperado de parametros.yaml."""
    poligono = cargar(p["rutas"]["zonas"])[p["zonas"]["area"]]
    rect = tuple(int(v) for v in cv2.boundingRect(poligono))
    if list(rect) != p["zonas"]["rectangulo_esperado"]:
        raise SystemExit(f"El rectángulo del área es {rect} y se esperaba {p['zonas']['rectangulo_esperado']}: "
                         "zonas.json cambió.")
    return poligono, rect


def mascara(poligono):
    """uint8 del tamaño del rectángulo envolvente: 255 dentro del polígono, 0 fuera."""
    x, y, w, h = cv2.boundingRect(poligono)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poligono - np.array([x, y])], 255)
    return m


def dentro(xs, ys, poligono):
    """True para cada punto dentro del polígono o sobre su borde."""
    return np.array([cv2.pointPolygonTest(poligono, (float(x), float(y)), False) >= 0
                     for x, y in zip(xs, ys)], dtype=bool)
