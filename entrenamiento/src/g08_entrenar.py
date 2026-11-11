"""Entrenamiento del modelo de G08 (Etapa 4): validación cruzada en train y modelo final.

Las particiones de `g08.particiones_test` (P3 y P4) forman el conjunto test: no intervienen en el
entrenamiento. Con P0, P1 y P2 se realizan tres iteraciones: en cada una se entrena con dos particiones y se
puntúan los automóviles de la tercera (predicciones out-of-fold). Luego se entrena el modelo final con las tres
particiones y se puntúa el conjunto test.

  Rama de interacción: HistGradientBoostingClassifier (valores por defecto de scikit-learn) sobre el vector
    u_w de cada ventana. Puntaje s_i.
  Rama de apariencia: secuencia de seis vectores de apariencia (1536 valores) -> estandarización con los datos
    de entrenamiento -> proyección a 128 valores -> un bloque Transformer (4 cabezas, perceptrón de 256)
    -> promedio de los seis instantes -> puntaje s_a.
    En cada época recibe todas las ventanas con evento y 8 sin evento por cada una: 4 del mismo automóvil,
    2 de automóviles parecidos y 2 del resto. Desde la época 10, cada 5 épocas, las del resto se eligen entre
    el 5 % de ventanas sin evento con mayor puntaje (ejemplos difíciles). Entropía cruzada binaria con peso 8
    para la clase positiva; AdamW; 30 épocas; lotes de 128.
  Cada rama se entrena con 5 semillas; los puntajes se estandarizan con las ventanas de entrenamiento y se
  promedian. La fusión de las ramas, con el peso estimado, se aplica en g08_evaluar.py (g08_fusion.py).

Entradas: ventanas, automóviles, eventos y matrices de rasgos de rutas.rasgos/g08.
Salida (rutas.resultados/<salida>): pred_ventanas.csv.gz, pred_autos.csv y entrenamiento.json
Uso: python src/g08_entrenar.py --sin-netas [--apariencia apariencia] [--salida g08] [--reanudar]
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd

import comun as c

G = c.P["g08"]


def logit_hgb(semilla, Xtr, ytr, X):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=semilla).fit(Xtr, ytr)
    return m.decision_function(X)


# ---------------------------------------------------------------- rama de apariencia

def construir_modelo(torch, d_in):
    nn = torch.nn

    class Bloque(nn.Module):
        def __init__(self, d, h, ff, p):
            super().__init__()
            self.h = h
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
            self.drop = nn.Dropout(p)

        def forward(self, x, mask):
            B, T, d = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
            a = a.masked_fill(~mask[:, None, None, :], -1e4).softmax(-1)
            x = x + self.drop(self.o((a @ v).transpose(1, 2).reshape(B, T, d)))
            return x + self.drop(self.ff(self.n2(x)))

    class Modelo(nn.Module):
        def __init__(self):
            super().__init__()
            h = G["dim_modelo"]
            self.proy = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(G["dropout_proyeccion"]))
            self.pos = nn.Parameter(torch.zeros(1, G["instantes_por_ventana"], h))
            self.bloque = Bloque(h, G["cabezas"], G["dim_perceptron"], G["dropout_temporal"])
            self.cab = nn.Linear(h, 1)

        def forward(self, x, mask):
            e = self.bloque(self.proy(x) + self.pos[:, :x.shape[1]], mask)
            m = mask.float().unsqueeze(-1)
            return self.cab((e * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

    return Modelo()


class Apariencia:
    """Ensambla lotes (B, 6, 1536) desde la memmap de rasgos, estandarizados con la partición de entrenamiento."""

    def __init__(self, arr, w, torch):
        self.arr, self.torch = arr, torch
        L = G["instantes_por_ventana"]
        i0, i1 = w.i0.to_numpy(), w.i1.to_numpy()
        self.filas = i0[:, None] + np.arange(L)[None]
        self.mask = self.filas < i1[:, None]
        self.filas = np.where(self.mask, self.filas, i0[:, None])
        self.mu = self.sd = None

    def normalizar(self, filas_inst, rng, n=G["muestra_normalizacion"]):
        x = np.asarray(self.arr[np.sort(rng.choice(filas_inst, min(n, len(filas_inst)), replace=False))], np.float32)
        x = x.reshape(len(x), -1)
        self.mu, sd = x.mean(0), x.std(0)
        self.sd = np.where(sd < 1e-6, 1.0, sd)

    def lote(self, ids):
        f, m = self.filas[ids], self.mask[ids]
        x = np.asarray(self.arr[f.ravel()], np.float32).reshape(len(ids), f.shape[1], -1)
        x = (x - self.mu) / self.sd * m[..., None]
        return self.torch.from_numpy(x), self.torch.from_numpy(m)


def puntuar(modelo, datos, ids, torch, bloque=4096):
    modelo.eval()
    out = np.empty(len(ids), np.float32)
    with torch.no_grad():
        for k in range(0, len(ids), bloque):
            x, m = datos.lote(ids[k:k + bloque])
            out[k:k + bloque] = modelo(x, m).numpy()
    return out


def entrenar_apariencia(datos, w, roles, tr, semilla, epocas, torch, log):
    """Devuelve el modelo entrenado con las ventanas `tr` (índices de w)."""
    rng = np.random.default_rng(semilla)
    torch.manual_seed(semilla)
    y = w.etiqueta_ventana.to_numpy()
    pos = tr[y[tr] == 1]
    neg = tr[y[tr] == 0]
    rol = roles[neg]
    pools = {"mismo_auto": neg[rol == "mismo_auto"], "parecidos": neg[rol == "parecido"], "resto": neg[rol == "resto"]}
    comp = G["composicion_negativos"]
    modelo = construir_modelo(torch, datos.arr.shape[1] * datos.arr.shape[2])
    opt = torch.optim.AdamW(modelo.parameters(), lr=G["tasa_aprendizaje"], weight_decay=G["decaimiento"])
    pw = torch.tensor(float(G["peso_positivo"]))
    dificiles = None
    curva = []
    for ep in range(1, epocas + 1):
        if ep >= G["epoca_dificiles"] and (ep - G["epoca_dificiles"]) % G["cada_epocas_dificiles"] == 0:
            s = puntuar(modelo, datos, pools["resto"], torch)
            n = max(1, int(round(G["frac_dificiles"] * len(s))))
            dificiles = pools["resto"][np.argsort(-s)[:n]]
        sel = [pos]
        for nombre, k in comp.items():
            pool = dificiles if (nombre == "resto" and dificiles is not None) else pools[nombre]
            if len(pool) == 0:
                pool = pools["resto"]
            n = k * len(pos)
            sel.append(rng.choice(pool, n, replace=len(pool) < n))
        u = np.concatenate(sel)
        yy = np.r_[np.ones(len(pos)), np.zeros(len(u) - len(pos))].astype(np.float32)
        orden = rng.permutation(len(u))
        u, yy = u[orden], yy[orden]
        modelo.train()
        perdida = []
        for k in range(0, len(u), G["tam_lote"]):
            x, m = datos.lote(u[k:k + G["tam_lote"]])
            z = modelo(x, m)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.from_numpy(yy[k:k + G["tam_lote"]]), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            perdida.append(float(loss))
        curva.append(round(float(np.mean(perdida)), 4))
    log.update({"positivas": int(len(pos)), "pools": {k: int(len(v)) for k, v in pools.items()}, "curva_perdida": curva})
    return modelo


def parecidos_por_lugar(autos, ev, k):
    """Automóviles sin evento que se detuvieron a menos de 1 altura y 10 min de un evento de las particiones de
    ENTRENAMIENTO (fold != k). Así las etiquetas de la partición de prueba no deciden qué negativos se muestrean.
    Misma regla que g08_poblaciones.py (parecidos_radio_altos, parecidos_max_min)."""
    cl = ["video_id", "track_id"]
    e = ev.merge(autos[cl + ["px", "py", "h", "fold"]], on=cl, how="left")
    e = e[(e.fold != k) & (e.fold >= 0) & e.px.notna()]
    sin_ev = (autos.etiqueta == 0) & autos.px.notna() & ~autos.falsa.astype(bool)
    cerca = np.zeros(len(autos), bool)
    for r in e.itertuples():
        f = c.fps(r.video_id)
        dist = np.hypot(autos.px - r.px, autos.py - r.py) / r.h
        dt = (autos.k_med - (r.k_ini + r.k_fin) / 2).abs() / f / 60
        cerca |= (sin_ev & (autos.video_id == r.video_id) & (dist <= G["parecidos_radio_altos"]) &
                  (dt <= G["parecidos_max_min"])).to_numpy()
    return cerca


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="g08")
    ap.add_argument("--sin-netas", action="store_true", help="rama de interacción de 20 variables (sin apariciones netas)")
    ap.add_argument("--apariencia", default="apariencia", help="archivo de vectores de apariencia (apariencia: ViT-S/14; apariencia_b14: ViT-B/14)")
    ap.add_argument("--reanudar", action="store_true", help="recupera las particiones ya terminadas (parcial_fold*.npz)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, __import__("os").cpu_count() - 2))

    d = c.R["rasgos"] / "g08"
    out = c.salida(a.salida)
    w = pd.read_csv(d / "ventanas.csv.gz")
    autos = pd.read_csv(d / "autos.csv")
    Xw = np.load(d / "interaccion_ventanas.npy")
    if a.sin_netas:   # las dos últimas columnas son las variables netas (g08_interaccion.py, columnas 21-22)
        Xw = Xw[:, :-2]
    clave = ["video_id", "track_id"]
    # conjunto test: fold = -1 (nunca entrena); el modelo final, entrenado con train, lo puntúa
    w["fold_original"], autos["fold_original"] = w.fold, autos.fold
    w.loc[w.fold.isin(G["particiones_test"]), "fold"] = -1
    autos.loc[autos.fold.isin(G["particiones_test"]), "fold"] = -1
    folds = sorted(f for f in w.fold.unique() if f >= 0) + [-1]
    semillas, epocas = G["semillas"], G["epocas"]
    # roles para el muestreo de la rama de apariencia: se recalculan en cada partición (parecidos_por_lugar)
    clave = ["video_id", "track_id"]
    ev = pd.read_csv(d / "eventos.csv")
    wa = w[clave].merge(autos[clave + ["etiqueta", "persona_junto"]], on=clave, how="left")
    pos_auto = pd.MultiIndex.from_frame(autos[clave])
    idx_w = pos_auto.get_indexer(pd.MultiIndex.from_frame(w[clave]))
    # ventana contaminada: contiene un recorte que muestra el evento de un vecino; se guarda la partición del evento
    ins = pd.read_csv(d / "instantes.csv.gz", usecols=["contamina_video", "contamina_track"],
                      dtype={"contamina_video": str}, keep_default_na=False)
    fold_auto = dict(zip(zip(autos.video_id, autos.track_id), autos.fold))
    f_inst = np.array([fold_auto.get((v, int(t)), -99) if t != -1 else -99
                       for v, t in zip(ins.contamina_video, ins.contamina_track)])
    cont_fold = np.full(len(w), -99)
    hay = np.nonzero(f_inst != -99)[0]
    if len(hay):
        for j, (i0, i1) in enumerate(zip(w.i0.to_numpy(), w.i1.to_numpy())):
            a_, b_ = np.searchsorted(hay, i0), np.searchsorted(hay, i1)
            if b_ > a_:
                cont_fold[j] = f_inst[hay[a_]]
    # las cajas falsas no son automóviles: fuera del entrenamiento y de la prueba (g08_poblaciones.py)
    falsa = w[clave].merge(autos[clave + ["falsa"]], on=clave, how="left").falsa.fillna(False).to_numpy(bool)
    y = np.where(falsa, -9, w.etiqueta_ventana.to_numpy())
    usar_ap = True
    if usar_ap:
        arr = np.load(d / f"{a.apariencia}.npy", mmap_mode="r")
        assert arr.shape[0] == w.i1.max(), "la matriz de apariencia no cubre todos los instantes"
        datos = Apariencia(arr, w, torch)
    s_i = np.full(len(w), np.nan)
    s_a = np.full(len(w), np.nan)
    log = {"folds": {}, "parametros": G, "epocas": epocas, "semillas": semillas}
    t0 = time.time()
    for k in folds:
        te = np.nonzero((w.fold.to_numpy() == k) & ~falsa)[0]
        parcial = out / f"parcial_fold{k}.npz"
        if a.reanudar and parcial.exists():   # partición ya terminada en una corrida anterior interrumpida
            z = np.load(parcial)
            assert np.array_equal(z["te"], te), f"{parcial} no corresponde a esta partición"
            s_i[te], s_a[te] = z["s_i"], z["s_a"]
            log["folds"][str(k)] = json.loads(str(z["log"]))
            print(f"fold {k}: recuperado de {parcial.name}", flush=True)
            continue
        # contaminada por un evento de entrenamiento: fuera del entrenamiento (en la prueba nunca se quita nada)
        cont_tr = (cont_fold != -99) & (cont_fold != k)
        tr = np.nonzero((w.fold.to_numpy() != k) & (w.fold.to_numpy() >= 0) & (y >= 0) & ~cont_tr)[0]
        parecido = wa.persona_junto.fillna(False).to_numpy(bool) | parecidos_por_lugar(autos, ev, k)[idx_w]
        roles = np.where(wa.etiqueta.to_numpy() == 1, "mismo_auto", np.where(parecido & (wa.etiqueta.to_numpy() == 0),
                                                                           "parecido", "resto"))
        lk = log["folds"][str(k)] = {"ventanas_prueba": int(len(te)), "ventanas_entrenamiento": int(len(tr)),
                                     "ventanas_contaminadas_excluidas": int((cont_tr & (w.fold.to_numpy() != k)).sum()),
                                     "positivas_entrenamiento": int((y[tr] == 1).sum()), "semillas": {}}
        acum_i, acum_a = np.zeros(len(te)), np.zeros(len(te))
        if usar_ap:
            inst_tr = np.unique(datos.filas[tr][datos.mask[tr]])
            datos.normalizar(inst_tr, np.random.default_rng(k + 1))   # k = -1 es el modelo final
        for s in semillas:
            ls = lk["semillas"][str(s)] = {}
            z = logit_hgb(s, Xw[tr], y[tr], np.r_[Xw[tr], Xw[te]])
            ztr, zte = z[:len(tr)], z[len(tr):]
            acum_i += (zte - ztr.mean()) / max(ztr.std(), 1e-9)
            if usar_ap:
                modelo = entrenar_apariencia(datos, w, roles, tr, s, epocas, torch, ls)
                ztr = puntuar(modelo, datos, tr, torch)
                zte = puntuar(modelo, datos, te, torch)
                acum_a += (zte - ztr.mean()) / max(ztr.std(), 1e-9)
            print(f"fold {k} semilla {s}: {time.time() - t0:.0f} s", flush=True)
        s_i[te] = acum_i / len(semillas)
        if usar_ap:
            s_a[te] = acum_a / len(semillas)
        np.savez(parcial, te=te, s_i=s_i[te], s_a=s_a[te], log=json.dumps(lk))
    w["s_interaccion"], w["s_apariencia"] = s_i, s_a
    hechos = w.fold.isin(folds) & ~falsa
    w.loc[hechos, ["w_id", "video_id", "track_id", "fold", "fold_original", "etiqueta_ventana", "k_ini", "k_fin",
                   "s_interaccion", "s_apariencia"]].to_csv(out / "pred_ventanas.csv.gz", index=False)
    pa = w[hechos].groupby(clave).agg(S_interaccion=("s_interaccion", "max"), S_apariencia=("s_apariencia", "max"),
                                      n_ventanas=("w_id", "size")).reset_index()
    pa = autos.merge(pa, on=clave, how="inner")
    pa.to_csv(out / "pred_autos.csv", index=False)
    log["segundos"] = round(time.time() - t0, 1)
    c.guardar_json(log, out / "entrenamiento.json")
    for k in folds:
        (out / f"parcial_fold{k}.npz").unlink(missing_ok=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
