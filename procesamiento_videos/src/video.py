"""Lectura secuencial de video y hora absoluta de cada cuadro.

Los cuadros se leen siempre en orden desde el 0. En estos videos HEVC la búsqueda por índice
de OpenCV no es exacta (entrega el cuadro pedido o hasta 11 antes), y los índices de cuadro
deben coincidir entre todas las etapas.

Uso: import video; for k, cuadro in video.cuadros(ruta): ...
"""
from datetime import datetime, timedelta

import cv2


def abrir(ruta):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {ruta}")
    return cap


def cuadros(ruta, desde=0, hasta=None):
    """Genera (índice, cuadro) en orden. Antes de `desde` solo avanza sin decodificar la imagen."""
    cap = abrir(ruta)
    try:
        k = 0
        while hasta is None or k < hasta:
            if not cap.grab():
                return
            if k >= desde:
                ok, cuadro = cap.retrieve()
                if not ok:
                    return
                yield k, cuadro
            k += 1
    finally:
        cap.release()


class Reloj:
    """Hora absoluta de un cuadro: inicio del cuadro 0 (medido) más índice / fps."""

    def __init__(self, fila):
        self.inicio = datetime.fromisoformat(f"{fila['fecha']}T{fila['hora_cuadro0']}")
        self.fps = float(fila["fps"])

    def instante(self, k):
        return self.inicio + timedelta(seconds=k / self.fps)

    def hora(self, k):
        return self.instante(k).strftime("%H:%M:%S")

    def segundos_del_dia(self, k):
        t = self.instante(k)
        return round(t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6, 2)
