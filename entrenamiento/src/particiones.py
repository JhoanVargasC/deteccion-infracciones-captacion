"""Composición de las particiones de G08 y de los conjuntos train y test (Etapa 2).

Cada video se corta en tramos de 10 minutos y el tramo b va a la partición b mod 5 (construir_dataset.py).
Las particiones de `g08.particiones_test` (P3 y P4) forman el conjunto test, y las demás (P0 a P2), el
conjunto train. La elección de P3 y P4 sigue dos criterios: contener entre el 20 y el 30 % de los eventos y
mantener una proporción de eventos nocturnos cercana a la del total. Este script informa esos valores.

Entradas: g08.csv (rutas.dataset) y autos.csv (g08_poblaciones.py), para excluir las cajas falsas.
Salida (rutas.dataset): particiones.csv y particiones.md
Uso: python src/particiones.py
"""
import pandas as pd

import comun as c


def resumen(u):
    con = u[u.etiqueta == 1]
    return {"automoviles": int(len(u)), "con_evento": int(len(con)),
            "de_dia": int((con.periodo == "dia").sum()), "de_noche": int((con.periodo == "noche").sum())}


def main():
    u = pd.read_csv(c.R["dataset"] / "g08.csv")
    falsa = pd.read_csv(c.R["rasgos"] / "g08" / "autos.csv", usecols=["video_id", "track_id", "falsa"])
    u = u.merge(falsa, on=["video_id", "track_id"], how="left")
    u = u[~u.falsa.fillna(False).astype(bool)]   # las cajas falsas no son automóviles
    test = c.P["g08"]["particiones_test"]
    filas = [{"grupo": f"P{k}", **resumen(g)} for k, g in u.groupby("fold")]
    filas += [{"grupo": "train", **resumen(u[~u.fold.isin(test)])},
              {"grupo": "test", **resumen(u[u.fold.isin(test)])},
              {"grupo": "total", **resumen(u)}]
    t = pd.DataFrame(filas)
    t.to_csv(c.R["dataset"] / "particiones.csv", index=False)
    tot, te = t.set_index("grupo").loc["total"], t.set_index("grupo").loc["test"]
    texto = ["# Particiones de G08", "", c.markdown(t), "",
             f"- Eventos en test: {te.con_evento / tot.con_evento:.1%} (criterio: entre 20 y 30 %).",
             f"- Eventos nocturnos: {te.de_noche / max(te.con_evento, 1):.1%} en test y "
             f"{tot.de_noche / max(tot.con_evento, 1):.1%} en el total."]
    (c.R["dataset"] / "particiones.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
