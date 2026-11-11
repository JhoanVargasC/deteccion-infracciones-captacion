"""Junta las etiquetas válidas de todas las anotadores en una tabla por infracción.

Entrada: salidas/<anotador>/etiquetas_<infraccion>.csv (de validar_anotaciones.py), para las
anotadores de `consolidado.anotadores` en parametros.yaml. Una anotador puede limitarse a una
lista de videos (p. ej. anotador_a2).
Salida en salidas/consolidado/:
  etiquetas_<infraccion>.csv   solo filas `ok`, con la columna `anotador`
  cobertura.csv                por video, infracción y anotador: filas válidas y vehículos distintos
Uso: python src/consolidar.py
"""
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
SAL = RAIZ / P["rutas"]["salidas"]


def main():
    out = SAL / "consolidado"
    out.mkdir(parents=True, exist_ok=True)
    cob = []
    for inf in P["validacion"]["infracciones"]:
        partes = []
        for nombre, vids in P["consolidado"]["anotadores"].items():
            e = pd.read_csv(SAL / nombre / f"etiquetas_{inf.lower()}.csv")
            e = e[e.estado == "ok"].assign(anotador=nombre.rstrip("0123456789"))
            if vids != "todos":
                e = e[e.video_id.isin(vids)]
            partes.append(e)
        if not partes:
            continue
        e = pd.concat(partes, ignore_index=True)
        dup = e.duplicated(["video_id", "track_id", "hora_inicio", "hora_fin"], keep=False)
        if dup.any():
            print(f"{inf}: {dup.sum()} filas repetidas entre anotadores (mismo video, ID e intervalo)")
        e.sort_values(["video_id", "track_id"]).to_csv(out / f"etiquetas_{inf.lower()}.csv", index=False)
        for (vid, an), g in e.groupby(["video_id", "anotador"]):
            cob.append({"video_id": vid, "infraccion": inf, "anotador": an, "filas_ok": len(g),
                        "vehiculos": g.track_id.nunique()})
    c = pd.DataFrame(cob)
    c.to_csv(out / "cobertura.csv", index=False)
    print(c.pivot_table(index="video_id", columns="infraccion", values="vehiculos", fill_value=0).astype(int).to_string())


if __name__ == "__main__":
    main()
