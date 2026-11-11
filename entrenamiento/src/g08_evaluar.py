"""Evaluación del modelo de G08 (Etapas 4 y 5).

1. Peso de la fusión: se estima con las predicciones out-of-fold de train (g08_fusion.py). Cada partición de
   train se puntúa con el peso estimado en las otras dos, y el conjunto test, con el peso estimado en las tres.
2. Umbral de alerta τ: el puntaje con el que los candidatos de train producen 20 alertas por hora de video.
3. Medidas, en train (out-of-fold) y en test, sobre los candidatos y sobre todos los automóviles: AP (con su
   referencia, la proporción de automóviles con evento), AUROC y, con el umbral τ, precisión, recall y F1.
   Se reportan la fusión y cada rama por separado. Intervalos: bootstrap percentil por vehículo.
4. Verificaciones en train: si la ventana de mayor puntaje de cada automóvil con evento coincide con el evento
   ampliado 1 s por lado, frente a lo esperado al azar; y la correlación de Spearman entre el puntaje y la
   duración del tracklet en los automóviles sin evento.

Entradas: pred_autos.csv y pred_ventanas.csv.gz (g08_entrenar.py); eventos.csv (g08_ventanas.py).
Salida (rutas.resultados/<salida>): pred_autos_fusion.csv, pred_ventanas_fusion.csv.gz, metricas.json y .md
Uso: python src/g08_evaluar.py [--salida g08]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import comun as c
import g08_fusion as gf

G = c.P["g08"]


def horas(vids, folds):
    """Horas de video que cubren las particiones `folds` (fracción de sus tramos de 10 minutos)."""
    t = pd.read_csv(c.R["dataset"] / "tracklets.csv", usecols=["video_id", "bloque", "fold"])
    t = t[t.video_id.isin(set(vids))].drop_duplicates(["video_id", "bloque"])
    return c.horas_video(vids) * float(t.fold.isin(folds).mean())


def medidas(x, col, b, semilla):
    y, s, g = x.etiqueta.to_numpy(bool), x[col].to_numpy(float), x.vehiculo.to_numpy()
    r = {"n": int(len(y)), "con_evento": int(y.sum()), "referencia": round(float(y.mean()), 4),
         "ap": round(c.average_precision(y, s), 4), "ic95_ap": c.bootstrap(y, lambda i: c.average_precision(y[i], s[i]), b, semilla, g),
         "auroc": round(c.auroc(y, s), 4)}
    if col == "S":
        d = x.decision.to_numpy(bool)
        r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.prf(y, d).items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="g08")
    a = ap.parse_args()
    out = c.salida(a.salida)
    b, semilla = c.P["reglas"]["bootstrap"], c.P["validacion"]["semilla"]
    pa = pd.read_csv(out / "pred_autos.csv")
    pw = pd.read_csv(out / "pred_ventanas.csv.gz")

    # 1. peso de la fusión
    pa, pw, fusion = gf.aplicar(pw, pa, pa.persona_junto.astype(bool))
    pa.to_csv(out / "pred_autos_fusion.csv", index=False)
    pw.to_csv(out / "pred_ventanas_fusion.csv.gz", index=False)

    # 2. umbral de alerta con los candidatos de train (puntaje con el peso de todo train, el mismo que recibe test)
    train, test = pa[pa.fold >= 0].copy(), pa[pa.fold == -1].copy()
    h_train = horas(train.video_id, sorted(train.fold_original.unique()))
    h_test = horas(test.video_id, G["particiones_test"])
    cand_tr = train[train.persona_junto.astype(bool)]
    tau = c.umbral_alertas(cand_tr.S_alfa_total, h_train, G["alertas_por_hora"])
    train["decision"], test["decision"] = train.S_alfa_total >= tau, test.S >= tau

    # 3. medidas
    res = {"fusion": fusion, "umbral_alerta": round(tau, 4)}
    for nombre, x, h in (("train", train, h_train), ("test", test, h_test)):
        r = res[nombre] = {"horas": round(h, 2)}
        for pob, m in (("candidatos", x.persona_junto.astype(bool)), ("todos", np.ones(len(x), bool))):
            r[pob] = {rama: medidas(x[m], col, b, semilla)
                      for rama, col in (("fusion", "S"), ("interaccion", "S_interaccion"), ("apariencia", "S_apariencia"))}
        cand = x[x.persona_junto.astype(bool)]
        r["punto_operacion"] = c.punto_operacion(cand.etiqueta, cand.S_alfa_total if nombre == "train" else cand.S, tau, h)
    per = pd.read_csv(c.R["dataset"] / "g08.csv", usecols=["video_id", "track_id", "periodo"])
    tp = train.merge(per, on=["video_id", "track_id"])
    tp = tp[tp.persona_junto.astype(bool)]
    res["train"]["por_periodo"] = {p: medidas(g, "S", b, semilla) for p, g in tp.groupby("periodo")}

    # 4. verificaciones en train
    ev = pd.read_csv(c.R["rasgos"] / "g08" / "eventos.csv")
    loc = []
    for (vid, tid), g in pw[(pw.fold >= 0) & (pw.etiqueta_ventana >= -1)].groupby(["video_id", "track_id"]):
        e = ev[(ev.video_id == vid) & (ev.track_id == tid)]
        if e.empty:
            continue
        mg = round(G["margen_evento_s"] * c.fps(vid))
        toca = np.zeros(len(g), bool)
        for r_ in e.itertuples():
            toca |= ((g.k_ini <= r_.k_fin + mg) & (g.k_fin >= r_.k_ini - mg)).to_numpy()
        loc.append({"acierto": bool(toca[int(np.argmax(g.s_fusion.to_numpy()))]), "azar": float(toca.mean())})
    loc = pd.DataFrame(loc)
    res["localizacion"] = {"aciertos": int(loc.acierto.sum()), "automoviles": len(loc),
                           "proporcion": round(float(loc.acierto.mean()), 4), "esperado_al_azar": round(float(loc.azar.mean()), 4)}
    neg = train[train.etiqueta == 0]
    rho, p = spearmanr(neg.S, neg.duracion_s)
    res["duracion"] = {"spearman_rho": round(float(rho), 4), "p": float(p), "n": int(len(neg))}
    c.guardar_json(res, out / "metricas.json")

    filas = [{"conjunto": n, "poblacion": pob, "rama": rama, "n": v["n"], "con_evento": v["con_evento"],
              "ap": v["ap"], "ic95_ap": v["ic95_ap"], "referencia": v["referencia"], "auroc": v["auroc"]}
             for n in ("train", "test") for pob in ("candidatos", "todos") for rama, v in res[n][pob].items()]
    texto = ["# Evaluación del modelo de G08", "", c.markdown(pd.DataFrame(filas)), "",
             f"- Peso de la fusión: {fusion['alfa_train']} (por partición: {fusion['alfa_por_particion']}).",
             f"- Umbral de alerta τ = {res['umbral_alerta']}; train: {res['train']['punto_operacion']}; test: {res['test']['punto_operacion']}.",
             f"- Localización: {res['localizacion']['aciertos']} de {res['localizacion']['automoviles']} automóviles con evento "
             f"(al azar: {res['localizacion']['esperado_al_azar']:.1%}).",
             f"- Spearman entre el puntaje y la duración (sin evento): {res['duracion']['spearman_rho']} (n = {res['duracion']['n']})."]
    (out / "metricas.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
