"""Crea el Excel que llena el validador externo a partir de validacion.csv (lo escribe exportar_clips.py).

Entrada: <carpeta de entrega>/validacion.csv. Salida: <carpeta de entrega>/validacion_infracciones.xlsx
(se borra validacion.csv). La carpeta es rutas.entrega_validacion o el primer argumento.
Requiere openpyxl, que no usan los demás scripts.
Uso: python src/crear_excel_validacion.py [carpeta_de_entrega]
"""
import csv
import sys
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
# carpeta de entrega: la de parametros.yaml, o la que se pase como argumento (p. ej. la de muestra_validacion)
ENTREGA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(P["rutas"]["entrega_validacion"])

INFRACCIONES = [  # texto oficial: tabla de infracciones de SUTRAN, TUO del Reglamento Nacional de Tránsito (DS 016-2009-MTC)
    ("G08", "No utilizar el carril derecho para recoger o dejar pasajeros o carga."),
    ("G10", "Incumplir las disposiciones sobre el uso de las vías de tránsito rápido y/o de acceso restringido "
            "(en esta vía: circular por el carril exclusivo del Corredor Rojo sin ser bus del corredor)."),
    ("G41", "Estacionar o detener el vehículo sobre la línea demarcatoria de intersección, dentro de éstas o en el "
            "crucero peatonal (paso peatonal)."),
    ("M21", "Estacionar interrumpiendo totalmente el tránsito."),
]
INSTRUCCIONES = [
    "Cada clip muestra un vehículo marcado con un rectángulo magenta.",
    "Indique si ESE vehículo comete alguna de las infracciones de la tabla y, si es así, en qué segundo del clip "
    "empieza y termina (el contador está abajo a la izquierda del video).",
    "Muchos clips NO muestran ninguna infracción: responder «No» es tan importante como responder «Sí».",
    "Si el vehículo marcado comete más de una infracción, anótelas TODAS: use las columnas «Infracción 2» y, si hubiera "
    "una tercera, escríbala en Observaciones.",
    "Si no se puede decidir (vehículo tapado, poca luz, el clip no muestra lo necesario), elija «No se puede determinar» "
    "y explique en Observaciones.",
    "Los clips están en orden aleatorio. Si no alcanza el tiempo para todos, avance en orden: cualquier tramo desde el "
    "inicio es una muestra al azar.",
    "Si no conoce el segundo exacto, deje los segundos en blanco y responda solo la infracción.",
]
COLS = ["Clip", "Archivo", "Duración (s)", "¿El vehículo marcado comete una infracción?", "Infracción 1",
        "Segundo inicio 1", "Segundo fin 1", "Infracción 2", "Segundo inicio 2", "Segundo fin 2", "Observaciones"]


def main():
    wb = Workbook()
    ins = wb.active
    ins.title = "Instrucciones"
    ins["A1"] = "Validación de infracciones de tránsito: instrucciones"
    ins["A1"].font = Font(bold=True, size=14)
    for i, t in enumerate(INSTRUCCIONES, 3):
        ins.cell(i, 1, f"{i - 2}. {t}").alignment = Alignment(wrap_text=True, vertical="top")
    f0 = len(INSTRUCCIONES) + 4
    ins.cell(f0, 1, "Código").font = ins.cell(f0, 2, "Conducta (texto oficial)").font = Font(bold=True)
    for i, (c, t) in enumerate(INFRACCIONES, f0 + 1):
        ins.cell(i, 1, c)
        ins.cell(i, 2, t).alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width = 110
    ins.column_dimensions["B"].width = 90

    ws = wb.create_sheet("Validación")
    ws.append(COLS)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    with (ENTREGA / "validacion.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ws.append([r["clip"], r["archivo"], float(r["duracion_s"])])
    n = ws.max_row
    si_no = DataValidation(type="list", formula1='"Sí,No,No se puede determinar"', allow_blank=True)
    inf = DataValidation(type="list", formula1='"G08,G10,G41,M21,Otra"', allow_blank=True)
    seg = DataValidation(type="decimal", operator="between", formula1="0", formula2="300", allow_blank=True)
    for dv in (si_no, inf, seg):
        ws.add_data_validation(dv)
    si_no.add(f"D2:D{n}")
    inf.add(f"E2:E{n}")
    inf.add(f"H2:H{n}")
    for col in "FGIJ":
        seg.add(f"{col}2:{col}{n}")
    for col, w in zip("ABCDEFGHIJK", (11, 22, 10, 26, 12, 10, 10, 12, 10, 10, 50)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B2"
    wb.active = 1
    salida = ENTREGA / "validacion_infracciones.xlsx"
    wb.save(salida)
    (ENTREGA / "validacion.csv").unlink()
    print(f"{salida}: {n - 1} clips")


if __name__ == "__main__":
    main()
