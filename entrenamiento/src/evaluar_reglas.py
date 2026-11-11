"""Evaluación de las reglas de G10 y G41 (Etapas 4 y 5).

Las reglas no se ajustan con etiquetas: se aplican y se comparan con las anotaciones de referencia.
  G10(T) = 1[fotogramas con el punto de contacto en el carril exclusivo >= 7] · 1[T no es bus], por tracklet.
  G41(d) = 1[prof_d > 0], por detención estricta (g41.csv ya excluye las cajas falsas). El vehículo se
           clasifica como infractor cuando al menos una de sus detenciones cumple la regla.
Medidas: precisión, recall y F1, en total y por período de iluminación, con intervalos de confianza al 95 %
por bootstrap percentil que remuestrea vehículos completos.

Entradas: g10.csv y g41.csv (construir_dataset.py).
Salida (rutas.resultados/reglas): g10_decisiones.csv, g41_vehiculo_decisiones.csv, metricas_reglas.json y .md
Uso: python src/evaluar_reglas.py
"""
import cv2
import numpy as np
import pandas as pd

import comun as c
import zonas

RG = c.P["reglas"]


def cuadros_en_carril(vids):
    """Fotogramas con el punto de contacto (centro del borde inferior de la caja) en el carril exclusivo."""
    carril = c.ZONAS[c.PE["candidatos"]["zona_carril"]]
    x0, y0, w, h = (int(v) for v in cv2.boundingRect(carril))
    out = []
    for vid in vids:
        tr = c.leer_csv(vid, "trayectorias_vehiculos.csv.gz", usecols=["frame", "track_id", "x1", "x2", "y2"])
        px, py = (tr.x1 + tr.x2) / 2, tr.y2
        tr = tr[(px >= x0) & (px <= x0 + w) & (py >= y0) & (py <= y0 + h)]
        tr = tr[zonas.dentro((tr.x1 + tr.x2) / 2, tr.y2, carril)]
        out.append(tr.groupby("track_id").size().rename("cuadros_carril").reset_index().assign(video_id=vid))
    return pd.concat(out, ignore_index=True)


def metricas(u, b, semilla):
    y, d = u.etiqueta.to_numpy(bool), u.decision.to_numpy(bool)
    m = c.prf(y, d)
    for k in ("precision", "recall", "f1"):
        m[f"ic95_{k}"] = c.bootstrap(y, lambda i, k=k: c.prf(y[i], d[i])[k], b, semilla, u.vehiculo.to_numpy())
    m["n"], m["n_pos"] = int(len(y)), int(y.sum())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def evaluar(u, b, semilla):
    return {"total": metricas(u, b, semilla),
            "dia": metricas(u[u.periodo == "dia"], b, semilla),
            "noche": metricas(u[u.periodo == "noche"], b, semilla)}


def main():
    ds = c.R["dataset"]
    out = c.salida("reglas")
    b, semilla = RG["bootstrap"], c.P["validacion"]["semilla"]

    g10 = pd.read_csv(ds / "g10.csv")
    g10 = g10.merge(cuadros_en_carril(sorted(g10.video_id.unique())), on=["video_id", "track_id"], how="left")
    g10["cuadros_carril"] = g10.cuadros_carril.fillna(0).astype(int)
    g10["decision"] = (g10.cuadros_carril >= RG["G10"]["min_cuadros_carril"]) & (g10.clase != "bus")

    det = pd.read_csv(ds / "g41.csv")
    det["decision"] = det.prof_mediana > RG["G41"]["umbral_profundidad"]
    g41 = det.groupby(["video_id", "track_id"]).agg(
        vehiculo=("vehiculo", "first"), periodo=("periodo", "first"), clase=("clase", "first"),
        etiqueta=("etiqueta", "max"), decision=("decision", "max"), prof_max=("prof_mediana", "max"),
        n_detenciones=("k_ini", "size")).reset_index()

    res = {"G10": evaluar(g10, b, semilla), "G41": evaluar(g41, b, semilla)}
    g10[["video_id", "track_id", "vehiculo", "k_ini", "k_fin", "clase", "periodo", "cuadros_carril", "decision", "etiqueta"]] \
        .to_csv(out / "g10_decisiones.csv", index=False)
    g41.to_csv(out / "g41_vehiculo_decisiones.csv", index=False)
    c.guardar_json(res, out / "metricas_reglas.json")

    filas = [{"infraccion": inf, "poblacion": p, "n": m["n"], "positivos": m["n_pos"], "vp": m["vp"], "fp": m["fp"],
              "fn": m["fn"], "precision": m["precision"], "ic95_precision": m["ic95_precision"], "recall": m["recall"],
              "ic95_recall": m["ic95_recall"], "f1": m["f1"]} for inf, r in res.items() for p, m in r.items()]
    texto = ["# Evaluación de las reglas de G10 y G41", "",
             "G10 por tracklet; G41 por vehículo (al menos una detención sobre el crucero).", "",
             c.markdown(pd.DataFrame(filas))]
    (out / "metricas_reglas.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
