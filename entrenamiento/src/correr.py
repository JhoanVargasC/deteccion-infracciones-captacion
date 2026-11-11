"""Ejecuta en orden los pasos del pipeline de entrenamiento y evaluación. Se detiene en el primer paso
que falla; en particular, si la validación del conjunto de datos encuentra un error.

Pasos (cada uno usa lo que dejan los anteriores):
  iluminacion   iluminacion.py         período de iluminación de cada tramo de 10 minutos (brillo del cielo, Otsu)
  dataset       construir_dataset.py   conjunto de datos y particiones
  validar       validar_dataset.py     comprobaciones del conjunto; detiene todo si hay un error
  reglas        evaluar_reglas.py      evaluación de G10 y G41
  ventanas      g08_ventanas.py        instantes de muestreo y ventanas de G08
  interaccion   g08_interaccion.py     vectores de interacción
  poblaciones   g08_poblaciones.py     candidatos y automóviles parecidos
  particiones   particiones.py         composición de las particiones y de los conjuntos train y test
  apariencia    g08_apariencia.py      vectores de apariencia con DINOv2 (GPU; reanudable)
  entrenar      g08_entrenar.py        validación cruzada en train y modelo final
  evaluar       g08_evaluar.py         peso de la fusión, umbral de alerta y medidas de G08

Se ejecutan aparte: evaluar_m21.py (regla de M21 y su revisión visual) y comparar_configuraciones.py
(selección de la configuración de G08).
La salida de cada paso queda en rutas.registros/<paso>.log.
Uso: python src/correr.py [--pasos validar reglas ...] [--desde entrenar] [--reanudar]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import comun as c

PASOS = {"iluminacion": "iluminacion.py", "dataset": "construir_dataset.py", "validar": "validar_dataset.py",
         "reglas": "evaluar_reglas.py", "ventanas": "g08_ventanas.py", "interaccion": "g08_interaccion.py",
         "poblaciones": "g08_poblaciones.py", "particiones": "particiones.py", "apariencia": "g08_apariencia.py",
         "entrenar": "g08_entrenar.py", "evaluar": "g08_evaluar.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", nargs="*", choices=list(PASOS))
    ap.add_argument("--desde", choices=list(PASOS))
    ap.add_argument("--reanudar", action="store_true", help="la extracción de apariencia continúa donde quedó")
    a = ap.parse_args()
    pasos = a.pasos or list(PASOS)[list(PASOS).index(a.desde) if a.desde else 0:]
    reg = c.R["registros"]
    reg.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for p in pasos:
        extra = (["--reanudar"] if p == "apariencia" and a.reanudar else []) + (["--sin-netas"] if p == "entrenar" else [])
        cmd = [sys.executable, "-W", "ignore", str(src / PASOS[p])] + extra
        t0 = time.time()
        print(f"== {p} ...", flush=True)
        with open(reg / f"{p}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=src.parent, env={**__import__("os").environ,
                                                                                            "PYTHONIOENCODING": "utf-8"})
        print(f"   {p}: {'ok' if r.returncode == 0 else 'FALLÓ'} en {time.time() - t0:.0f} s (registros/{p}.log)", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
