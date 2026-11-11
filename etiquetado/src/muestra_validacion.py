"""Muestra aleatoria para la validación externa (oficial de tránsito), con el tamaño mínimo que exige
un plan de muestreo de aceptación por atributos.

Entrada: salidas/final, salidas/<anotador>, salidas/candidatos y parámetros de `muestra_validacion`.

Población: todas las etiquetas vigentes, es decir las finales de las horas revisadas completas
(salidas/final) más las anotaciones válidas de las horas de `muestra_validacion.horas_nuevas`.
Cada etiqueta (vehículo e intervalo) es una unidad; cada infracción es un estrato.

Tamaño por estrato: el menor n tal que, si la proporción de etiquetas erróneas fuera p0 (5 %),
la probabilidad de encontrar c errores o menos en la muestra sería a lo sumo α (5 %):
    P(X <= c | N, D = ceil(p0·N), n) <= α,  con X hipergeométrica (población finita, sin reemplazo).
Si el validador encuentra c errores o menos, se concluye con 95 % de confianza que el error del
estrato es menor que p0. Se usa c = 1: un plan con c = 0 rechaza por un único caso ambiguo o un
descuido del validador, y c = 1 cuesta pocos clips más (con N grande, n pasa de 59 a 93).
Asignación dentro del estrato: proporcional a cada video × origen de la etiqueta (restos mayores),
y selección aleatoria simple sin reemplazo dentro de cada celda (semilla fija).
Controles ciegos: por cada estrato, fraccion_controles/(1 - fraccion_controles) × n unidades SIN
etiqueta de esa infracción que la regla propone (candidatos sin anotar), en
proporción a los videos. Cumplen dos funciones: el validador no puede suponer que todo clip es una
infracción, y estiman cuántas infracciones faltan entre los candidatos (omisiones).
Los clips no muestran etiquetas, números ni zonas: solo el vehículo marcado (exportar_clips.py).

Salidas: etiquetado/salidas/validacion_v2/{diseno_muestral.csv, muestra.csv, clave_clips.csv} (NO se entregan)
         <entrega>/clips/*.mp4, validacion.csv (y el Excel con crear_excel_validacion.py)
Uso: python src/muestra_validacion.py [--sin-clips]
"""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import hypergeom

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((RAIZ / P["rutas"]["procesamiento"]).resolve() / "src"))
sys.path.insert(0, str(RAIZ / "src"))
import configuracion, indice, video  # noqa: E402
import exportar_clips  # noqa: E402

M = P["muestra_validacion"]
C = P["clips_validacion"]
SAL = RAIZ / P["rutas"]["salidas"]


def n_minimo(N, p0, alfa, c):
    D = max(1, math.ceil(p0 * N))
    for n in range(1, N + 1):
        if hypergeom.cdf(c, N, D, n) <= alfa:
            return n, D
    return N, D


def asignar(df, n, rng, celdas):
    """Asignación proporcional por celdas con restos mayores; muestreo aleatorio simple en cada una."""
    tam = df.groupby(celdas).size()
    cuota = tam / tam.sum() * n
    base = np.floor(cuota).astype(int)
    resto = (cuota - base).sort_values(ascending=False)
    base[resto.index[: n - base.sum()]] += 1
    partes = []
    for k, m in base.items():
        if m:
            g = df.set_index(celdas).loc[[k]].reset_index() if len(celdas) > 1 else df[df[celdas[0]] == k]
            partes.append(g.iloc[rng.choice(len(g), m, replace=False)])
    return pd.concat(partes, ignore_index=True), base


def poblacion(inf, p, ind):
    fin = pd.read_csv(SAL / "final" / f"etiquetas_{inf.lower()}.csv")
    fin = fin[fin.confianza != "sin_validar"] if inf != "M21" else fin
    nuevas = []
    for anot, vids in M["horas_nuevas"].items():
        e = pd.read_csv(SAL / anot / f"etiquetas_{inf.lower()}.csv")
        e = e[e.video_id.isin(vids) & (e.estado == "ok")].copy()
        if inf == "G10":   # G10 sin horas: el tramo del vehículo en el carril (candidatos de la regla)
            for vid in vids:
                cg = pd.read_csv(SAL / "candidatos" / f"{vid}_G10.csv").set_index("track_id")
                m = (e.video_id == vid) & e.k_ini.isna()
                e.loc[m, "k_ini"] = e.track_id[m].map(cg.k_ini)
                e.loc[m, "k_fin"] = e.track_id[m].map(cg.k_fin)
        e = e.dropna(subset=["k_ini", "k_fin"]).drop_duplicates(["video_id", "track_id", "k_ini"])
        nuevas.append(e.assign(origen=anot.rstrip("0123456789"), confianza="anotacion_sin_verificar")
                      [["video_id", "track_id", "k_ini", "k_fin", "origen", "confianza"]])
    x = pd.concat([fin[["video_id", "track_id", "k_ini", "k_fin", "origen", "confianza"]]] + nuevas, ignore_index=True)
    x[["track_id", "k_ini", "k_fin"]] = x[["track_id", "k_ini", "k_fin"]].astype(int)
    return x


def candidatos_sin_etiqueta(inf, pob, vids):
    """Unidades que la regla propone y que no tienen etiqueta de la infracción."""
    filas = []
    for vid in vids:
        if inf == "G10":
            c = pd.read_csv(SAL / "candidatos" / f"{vid}_G10.csv")
        elif inf == "G41":
            c = pd.read_csv(SAL / "candidatos" / f"{vid}_detenciones.csv")
            c = c[~c.falsa & (c.prof_mediana > -P["muestra_validacion"]["g41_control_borde_px"])]
        else:
            c = pd.read_csv(SAL / "candidatos" / f"{vid}_{inf}.csv")
            c = c[~c.falsa] if "falsa" in c else c
        etq = set(pob[pob.video_id == vid].track_id)
        c = c[~c.track_id.isin(etq)]
        filas.append(c[["track_id", "k_ini", "k_fin"]].assign(video_id=vid))
    return pd.concat(filas, ignore_index=True).drop_duplicates(["video_id", "track_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-clips", action="store_true", help="solo el diseño y la muestra")
    a = ap.parse_args()
    p = configuracion.cargar()
    ind = indice.leer(p)
    out = SAL / "validacion_v2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(M["semilla"])
    f = C["fraccion_controles"]
    diseno, muestra = [], []
    for inf in M["infracciones"]:
        pob = poblacion(inf, p, ind)
        N = len(pob)
        n, D = n_minimo(N, M["p0"], M["alfa"], M["c"])
        sel, cuotas = asignar(pob, n, rng, ["video_id", "origen"])
        vids = sorted(set(pob.video_id) | set(ind.index[ind.index.isin(M["videos"])]))
        cand = candidatos_sin_etiqueta(inf, pob, vids)
        n_c = min(len(cand), round(f / (1 - f) * n))
        ctrl, _ = asignar(cand, n_c, rng, ["video_id"]) if n_c else (cand.iloc[0:0], None)
        muestra += [sel.assign(infraccion=inf, tipo="etiqueta"), ctrl.assign(infraccion=inf, tipo="control", origen="candidato_sin_etiqueta")]
        diseno.append({"infraccion": inf, "N_etiquetas": N, "D_tolerable": D, "n_muestra": n, "c_aceptacion": M["c"],
                       "prob_aceptar_si_error_es_p0": round(float(hypergeom.cdf(M["c"], N, D, n)), 4),
                       "candidatos_sin_etiqueta": len(cand), "n_controles": n_c,
                       "por_celda": "; ".join(f"{k[0][5:]} {k[1]}: {v}" for k, v in cuotas.items() if v)})
        print(f"{inf}: N={N} D={D} n={n} controles={n_c}")
    d = pd.DataFrame(diseno)
    d.to_csv(out / "diseno_muestral.csv", index=False)
    ms = pd.concat(muestra, ignore_index=True)
    ms.to_csv(out / "muestra.csv", index=False)
    print(d.drop(columns="por_celda").to_string(index=False))
    if a.sin_clips:
        return

    # planes de clip: un clip por vehículo y tramo; tramos cercanos del mismo vehículo se juntan
    planes, vidas = [], {}
    for vid, g in ms.groupby("video_id"):
        fps = float(ind.loc[vid].fps)
        tr = pd.read_csv(p["rutas"]["salidas"] / ind.loc[vid].sesion / vid / f"{vid}_trayectorias_vehiculos.csv.gz",
                         usecols=["frame", "track_id"])
        v = tr.groupby("track_id").frame.agg(["min", "max"])
        vidas.update({(vid, int(t)): (int(r["min"]), int(r["max"])) for t, r in v.iterrows()})
        junta = round(C["juntar_si_separacion_s"] * fps)
        for tid, h in g.sort_values("k_ini").groupby("track_id"):
            grupo = None
            for r in h.itertuples():
                if grupo and r.k_ini <= grupo["b"] + junta:
                    grupo["b"] = max(grupo["b"], int(r.k_fin))
                    grupo["origen"].add(r.tipo)
                    grupo["infracciones"].add(r.infraccion)
                    continue
                if grupo:
                    planes.append(grupo)
                grupo = {"video_id": vid, "track_id": int(tid), "a": int(r.k_ini), "b": int(r.k_fin),
                         "origen": {r.tipo}, "infracciones": {r.infraccion}}
            planes.append(grupo)
    entrega = Path(M["entrega"])
    exportar_clips.escribir_clips(planes, vidas, entrega, p, ind, random.Random(M["semilla"]))
    clave = pd.DataFrame([{"clip": pl["clip"], "video_id": pl["video_id"], "track_id": pl["track_id"],
                           "tipo": "+".join(sorted(pl["origen"])), "infracciones": "+".join(sorted(pl["infracciones"])),
                           "cuadro_inicio_clip": pl["c0"], "cuadro_fin_clip": pl["c1"],
                           "hora_inicio_clip": video.Reloj(ind.loc[pl["video_id"]]).hora(pl["c0"]),
                           "duracion_s": round((pl["c1"] - pl["c0"] + 1) / float(ind.loc[pl["video_id"]].fps), 1)}
                          for pl in planes]).sort_values("clip")
    clave.to_csv(out / "clave_clips.csv", index=False)
    clave[["clip", "duracion_s"]].assign(archivo=lambda x: "clips/" + x["clip"] + ".mp4").to_csv(entrega / "validacion.csv", index=False)
    print(clave.tipo.value_counts().to_string(), f"\n{len(clave)} clips, {clave.duracion_s.sum() / 60:.0f} min -> {entrega}")


if __name__ == "__main__":
    main()
