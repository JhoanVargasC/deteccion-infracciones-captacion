"""Propone correcciones para las filas en `revisar`. Solo propone cuando hay una única solución;
el resto queda para revisar en el video.

  - ID mal digitado: candidatos a un dígito de distancia (cambio, trasposición, dígito de más o de
    menos). En G10, el candidato debe pisar el carril, no ser bus, no estar ya anotado y entrar al
    carril dentro de la ventana de tiempo de sus filas vecinas de la hoja (se anotan en orden).
    En G41, G08 y M21, debe existir en el intervalo anotado (en G41, además, sobre el crucero).
  - Hora mal digitada: la sugerencia única que ya calcula validar_anotaciones.py.
  - Inicio y fin invertidos: si el ID existe en el intervalo invertido.
  - G41 sin hora de inicio: el comienzo de la detención (velocidad < umbral) que contiene la hora de fin.

Entrada: salidas/<anotador>/validacion_filas.csv.
Salida: config/correcciones_<anotador>_sugeridas.csv. Una vez revisado, se copia a correcciones_<anotador>.csv.
Uso: python src/sugerir_correcciones.py --anotador anotador_b
"""
import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((RAIZ / P["rutas"]["procesamiento"]).resolve() / "src"))
import configuracion, indice, zonas  # noqa: E402

V = P["validacion"]


def a_un_digito(a, b):
    a, b = str(int(a)), str(int(b))
    if a == b:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    if abs(len(a) - len(b)) == 1:
        largo, corto = (a, b) if len(a) > len(b) else (b, a)
        return any(largo[:i] + largo[i + 1:] == corto for i in range(len(largo)))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotador", required=True)
    a = ap.parse_args()
    p = configuracion.cargar()
    ind = indice.leer(p)
    z = zonas.cargar(p["rutas"]["zonas"])
    carril, cruce = z[V["zona_carril"]], z[V["zona_crucero"]]
    d = pd.read_csv(RAIZ / P["rutas"]["salidas"] / a.anotador / "validacion_filas.csv", dtype={"ini": str, "fin": str})
    props = []
    for vid in d[d.estado == "revisar"].video_id.dropna().unique():
        f = ind.loc[vid]
        fps = float(f.fps)
        t0 = datetime.fromisoformat(f"{f.fecha}T{f.hora_cuadro0}")
        h = lambda k: (t0 + timedelta(seconds=k / fps)).strftime("%H:%M:%S")  # noqa: E731
        k = lambda x: round((datetime.fromisoformat(f"{f.fecha}T{x}") - t0).total_seconds() * fps)  # noqa: E731
        tr = pd.read_csv(p["rutas"]["salidas"] / f.sesion / vid / f"{vid}_trayectorias_vehiculos.csv.gz").sort_values(["track_id", "frame"])
        tr["px"], tr["py"] = (tr.x1 + tr.x2) / 2, tr.y2
        tr["en"] = zonas.dentro(tr.px, tr.py, carril)
        tr["cru"] = zonas.dentro(tr.px, tr.py, cruce)
        cx, cy, hh = tr.px, (tr.y1 + tr.y2) / 2, (tr.y2 - tr.y1).clip(lower=1)
        paso = np.hypot(cx.diff(), cy.diff()) / tr.frame.diff() / hh
        paso[tr.track_id != tr.track_id.shift()] = np.nan
        tr["vel"] = paso.groupby(tr.track_id).transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
        s = tr.groupby("track_id").agg(en=("en", "sum"), bus=("clase", lambda x: (x == 5).mean()), k0=("frame", "min"), k1=("frame", "max"))
        ent = tr[tr.en].groupby("track_id").frame.min()
        g = d[d.video_id == vid]
        g10_ok = set(g[(g.infraccion == "G10") & (g.estado == "ok")].track_id.dropna().astype(int))
        for r in g[g.estado == "revisar"].itertuples():
            mot = str(r.motivo)
            fila = lambda campo, valor, tipo, ev: props.append(  # noqa: E731
                {"hoja": r.hoja, "fila_excel": r.fila_excel, "campo": campo, "valor": valor, "tipo": tipo, "evidencia": ev})
            sug = re.findall(r"¿(\d\d:\d\d:\d\d)-(\d\d:\d\d:\d\d)\?", str(r.sugerencia))
            if "invertir" in str(r.sugerencia) and "fuera del video" not in mot:
                fila("ini", r.fin, "inversion", f"Inicio y fin invertidos; el ID {int(r.track_id)} existe en {r.fin}-{r.ini}")
                fila("fin", r.ini, "inversion", "Ver la fila anterior")
                continue
            if len(set(sug)) == 1:
                a2, b2 = sug[0]
                for campo, viejo, nuevo in (("ini", r.ini, a2), ("fin", r.fin, b2)):
                    if viejo != nuevo:
                        fila(campo, nuevo, "digitacion", f"{viejo} es imposible para el ID {int(r.track_id)} (vida {r.sugerencia.split('vida del ID ')[-1]}); "
                             f"{nuevo} es la única corrección de un dígito de la hora o de la decena de minutos que cae en su vida")
                continue
            if "sin hora de inicio" in mot and r.infraccion == "G41" and isinstance(r.fin, str) and pd.notna(r.track_id):
                t = tr[tr.track_id == int(r.track_id)]
                kf = k(r.fin)
                q = t[(t.frame <= kf) & (t.vel < V["umbral_detenido"])]
                if len(q):
                    corte = q.frame.diff().fillna(1).gt(V["max_hueco_detencion"]).cumsum()
                    ki = int(q[corte == corte.iloc[-1]].frame.min())
                    fila("ini", h(ki), "inferida", f"Sin hora de inicio. La detención del ID {int(r.track_id)} que termina en {r.fin} empieza a las {h(ki)} "
                         f"(velocidad < {V['umbral_detenido']})")
                continue
            if pd.isna(r.track_id):
                continue
            tid = int(r.track_id)
            if r.infraccion == "G10" and ("carril" in mot or "no existe" in mot):
                hoja = g[g.hoja == r.hoja].sort_values("fila_excel")
                pos = list(hoja.fila_excel).index(r.fila_excel)
                vec = [int(x) for x in hoja.track_id.iloc[max(0, pos - 3):pos + 4].dropna() if int(x) != tid and int(x) in ent.index]
                c = []
                if vec:   # 1) ¿ID mal digitado? Es la explicación más fuerte: encaja en el orden de la hoja
                    lo, hi = ent[vec].min() - V["ventana_vecinos_s"] * fps, ent[vec].max() + V["ventana_vecinos_s"] * fps
                    c = [x for x in s.index if a_un_digito(tid, x) and s.en[x] >= V["min_cuadros_carril"] and s.bus[x] < V["max_frac_bus"]
                         and x not in g10_ok and x in ent.index and lo <= ent[x] <= hi]
                if len(c) == 1:
                    x = c[0]
                    fila("id", str(x), "digitacion", f"El ID {tid} nunca pisa el carril o no existe; sus filas vecinas ({min(vec)}-{max(vec)}) entran al carril "
                         f"entre {h(ent[vec].min())} y {h(ent[vec].max())}, y el {x} difiere en un dígito, entra a las {h(ent[x])} "
                         f"({int(s.en[x])} cuadros en el carril) y no estaba anotado")
                    continue
                if "carril" not in mot:
                    continue
                # 2) ¿un G41 anotado en la columna equivocada?
                g41 = g[(g.infraccion == "G41") & (g.track_id == tid) & g.ini.notna() & g.fin.notna()]
                if len(g41):
                    o = g41.iloc[0]
                    fila("infraccion", "G41", "reclasificacion", f"El ID {tid} nunca pisa el carril y la fila {o.fila_excel} ya lo registra como G41: "
                         "es el mismo caso anotado en la columna equivocada; queda fusionado con esa fila")
                    fila("ini", o.ini, "reclasificacion", f"Horas de la fila {o.fila_excel}")
                    fila("fin", o.fin, "reclasificacion", f"Horas de la fila {o.fila_excel}")
                    continue
                t = tr[tr.track_id == tid]
                q = t[(t.vel < V["umbral_detenido"]) & t.cru]
                if len(q):
                    corte = q.frame.diff().fillna(1).gt(V["max_hueco_detencion"]).cumsum()
                    tramo = q.groupby(corte).frame.agg(["min", "max"])
                    tramo["n"] = tramo["max"] - tramo["min"] + 1
                    tramo = tramo.sort_values("n").iloc[-1]
                    if tramo["n"] >= V["min_cuadros_detenido"]:
                        fila("infraccion", "G41", "reclasificacion", f"El ID {tid} nunca pisa el carril, ningún ID a un dígito encaja con sus filas vecinas, "
                             f"y está detenido sobre el crucero de {h(tramo['min'])} a {h(tramo['max'])} sin tener fila G41")
                        fila("ini", h(tramo["min"]), "inferida", "Inicio de la detención sobre el crucero según la trayectoria")
                        fila("fin", h(tramo["max"]), "inferida", "Fin de la detención sobre el crucero según la trayectoria")
            elif r.infraccion in ("G41", "G08", "M21") and ("no existe" in mot or "no aparece" in mot) and isinstance(r.ini, str) and isinstance(r.fin, str):
                ki, kf = k(r.ini), k(r.fin)
                w = tr[(tr.frame >= ki) & (tr.frame <= kf)]
                if r.infraccion == "G41":
                    w = w[w.cru]
                c = [x for x in w.track_id.unique() if a_un_digito(tid, x)]
                if len(c) == 1:
                    fila("id", str(int(c[0])), "digitacion", f"El ID {tid} no existe en {r.ini}-{r.fin}; el {int(c[0])} difiere en un dígito y es el único así "
                         "presente en ese intervalo" + (" sobre el crucero" if r.infraccion == "G41" else ""))
    out = pd.DataFrame(props, columns=["hoja", "fila_excel", "campo", "valor", "tipo", "evidencia"])
    ruta = RAIZ / "config" / f"correcciones_{a.anotador}_sugeridas.csv"
    out.to_csv(ruta, index=False, encoding="utf-8")
    print(f"{len(out)} correcciones propuestas para {out[['hoja', 'fila_excel']].drop_duplicates().shape[0]} filas -> {ruta.name}")
    print(out.groupby("tipo").size().to_string())


if __name__ == "__main__":
    main()
