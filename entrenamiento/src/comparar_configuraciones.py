"""Selección de la configuración de G08: compara dos variantes con una regla fijada de antemano.

Se calcula la AP de los candidatos de train con las predicciones out-of-fold de ambas variantes. La variante
más compleja se adopta solo si mejora la AP y el intervalo de confianza al 95 % de la diferencia, con
bootstrap pareado por vehículo, excluye el cero. En caso contrario, se conserva la más simple.
Entradas: rutas.resultados/<base>/pred_autos.csv y rutas.resultados/<variante>/pred_autos.csv
Salida: rutas.resultados/comparaciones/<base>_vs_<variante>.json
Uso: python src/comparar_configuraciones.py --base g08_v1 --variante g08_v2
"""
import argparse

import numpy as np
import pandas as pd

import comun as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variante", required=True)
    a = ap.parse_args()
    cl = ["video_id", "track_id"]
    # puntajes con el peso de fusión estimado (g08_evaluar.py) si existen; si no, los del entrenamiento
    def lee(n):
        d = c.R["resultados"] / n
        ruta = d / "pred_autos_fusion.csv" if (d / "pred_autos_fusion.csv").exists() else d / "pred_autos.csv"
        return pd.read_csv(ruta).query("fold >= 0")
    b, v = lee(a.base), lee(a.variante)
    x = b[cl + ["vehiculo", "etiqueta", "persona_junto", "S"]].merge(v[cl + ["S"]], on=cl, suffixes=("_b", "_v"))
    assert len(x) == len(b) == len(v), "las dos configuraciones no tienen los mismos automóviles"
    x = x[x.persona_junto.astype(bool)].reset_index(drop=True)
    y = x.etiqueta.to_numpy(bool)
    sb, sv = x.S_b.to_numpy(), x.S_v.to_numpy()
    dif = c.average_precision(y, sv) - c.average_precision(y, sb)
    nb, semilla = c.P["reglas"]["bootstrap"], c.P["validacion"]["semilla"]
    ic = c.bootstrap(y, lambda i: c.average_precision(y[i], sv[i]) - c.average_precision(y[i], sb[i]), nb, semilla,
                     x.vehiculo.to_numpy())
    adopta = bool(dif > 0 and ic[0] is not None and ic[0] > 0)
    res = {"base": a.base, "variante": a.variante, "poblacion": "candidatos de train (out-of-fold)",
           "automoviles": int(len(x)), "eventos": int(y.sum()),
           "ap_base": round(c.average_precision(y, sb), 4), "ap_variante": round(c.average_precision(y, sv), 4),
           "diferencia": round(float(dif), 4), "ic95_diferencia": ic,
           "decision": a.variante if adopta else a.base,
           "regla": "adoptar la variante solo si la diferencia es > 0 y su IC95 pareado por vehículo excluye el cero"}
    c.guardar_json(res, c.salida("comparaciones") / f"{a.base}_vs_{a.variante}.json")
    print(res)


if __name__ == "__main__":
    main()
