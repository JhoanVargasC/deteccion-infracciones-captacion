"""Valida y normaliza las anotaciones manuales del Excel de registro de infracciones.

Lee el libro (.xlsm/.xlsx) sin modificarlo, cruza cada fila con las salidas del
procesamiento y escribe en salidas/<anotador>/:
  etiquetas_<infraccion>.csv   una fila por anotación, con video_id, track_id, horas,
                               cuadros (k_ini, k_fin), estado (ok / revisar / descartada),
                               motivo, corrección aplicada y sugerencia
  cobertura.csv                tramo anotado por video e infracción
  validacion_filas.csv         todas las filas del Excel con sus controles
  g10_candidatos_sin_anotar.csv  vehículos que cumplen la regla G10 y no están anotados

Solo se corrige automáticamente lo que tiene evidencia directa en los datos (nombre de
video, duplicados exactos, hora escrita en la columna Acción). Lo demás queda en
`revisar` con una sugerencia, y no entra a la evaluación hasta que alguien lo confirme.

Uso: python src/validar_anotaciones.py "../Anotaciones/<archivo>.xlsm" --anotador anotador_a
Las salidas van a salidas/<anotador>/ y las correcciones se leen de config/correcciones_<anotador>.csv.
"""
import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
P = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
MOD_PROC = (RAIZ / P["rutas"]["procesamiento"]).resolve()
sys.path.insert(0, str(MOD_PROC / "src"))
import configuracion, indice, zonas  # noqa: E402

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def leer_libro(ruta):
    """Hojas del libro como listas de filas (valores crudos de texto). Lector mínimo de
    Office Open XML, para no depender de openpyxl, que no está en el entorno."""
    z = zipfile.ZipFile(ruta)
    comp = []
    if "xl/sharedStrings.xml" in z.namelist():
        comp = ["".join(x.text or "" for x in si.iter(f"{{{NS['m']}}}t"))
                for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    libro = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    hojas = {}
    for h in libro.find("m:sheets", NS):
        destino = rels[h.get(f"{{{NS['r']}}}id")].lstrip("/")
        destino = destino if destino.startswith("xl/") else "xl/" + destino
        filas = []
        for fila in ET.fromstring(z.read(destino)).iter(f"{{{NS['m']}}}row"):
            celdas = {}
            for c in fila.findall("m:c", NS):
                col = re.match(r"[A-Z]+", c.get("r")).group()
                v = c.find("m:v", NS)
                if c.get("t") == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                elif v is None:
                    val = None
                elif c.get("t") == "s":
                    val = comp[int(v.text)]
                else:
                    val = v.text
                celdas[col] = val
            filas.append((int(fila.get("r")), celdas))
        hojas[h.get("name")] = filas
    return hojas


def a_hora(v):
    """Celda de hora -> 'HH:MM:SS'. Excel guarda la hora como fracción del día."""
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    if s in P["validacion"]["horas_vacias"]:
        return ""
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        return s if s.count(":") == 2 else s + ":00"
    try:
        x = float(s) % 1
    except ValueError:
        return s
    seg = round(x * 86400)
    return f"{seg // 3600:02d}:{seg % 3600 // 60:02d}:{seg % 60:02d}"


def candidatos_g10(p, ind, carril, d, salida, cfg):
    """Vehículos que no son bus, pasan al menos `min_cuadros_carril` cuadros con el punto de apoyo
    en el carril exclusivo (condición de la regla de G10) y no están anotados como G10."""
    ok = d[(d.infraccion == "G10") & (d.estado == "ok")]
    out = []
    for vid in d[d.estado == "ok"].video_id.dropna().unique():
        f = ind.loc[vid]
        c = p["rutas"]["salidas"] / f.sesion / vid
        t0 = datetime.fromisoformat(f"{f.fecha}T{f.hora_cuadro0}")
        tr = pd.read_csv(c / f"{vid}_trayectorias_vehiculos.csv.gz")
        tl = set(pd.read_csv(c / f"{vid}_tracklets.csv").track_id)
        tr["en"] = zonas.dentro((tr.x1 + tr.x2) / 2, tr.y2, carril)
        s = tr[tr.en].groupby("track_id").agg(cuadros_en_carril=("frame", "size"), k_entrada=("frame", "min"), k_salida=("frame", "max"))
        s["bus"] = tr.groupby("track_id").clase.apply(lambda x: (x == 5).mean()).reindex(s.index)
        s = s[(s.bus < cfg["max_frac_bus"]) & (s.cuadros_en_carril >= cfg["min_cuadros_carril"]) & s.index.isin(tl)]
        anot = set(ok[ok.video_id == vid].track_id)
        print(f"{vid}: G10 candidatos {len(s)}, anotados {s.index.isin(anot).sum()} ({s.index.isin(anot).mean():.0%})")
        s = s[~s.index.isin(anot)].copy()
        h = lambda k: (t0 + timedelta(seconds=k / float(f.fps))).strftime("%H:%M:%S")  # noqa: E731
        s["hora_entrada"], s["hora_salida"] = s.k_entrada.map(h), s.k_salida.map(h)
        s.insert(0, "video_id", vid)
        out.append(s.drop(columns="bus").reset_index())
    if out:
        o = pd.concat(out)
        o["revision"] = ""
        o.to_csv(salida / "g10_candidatos_sin_anotar.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("libro")
    ap.add_argument("--anotador", required=True, help="nombre corto, en minúsculas: define la carpeta de salida")
    a = ap.parse_args()
    cfg = P["validacion"]
    p = configuracion.cargar()
    ind = indice.leer(p)
    z = zonas.cargar(p["rutas"]["zonas"])
    carril, cruce = z[cfg["zona_carril"]], z[cfg["zona_crucero"]]
    salida = RAIZ / P["rutas"]["salidas"] / a.anotador
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for hoja, contenido in leer_libro(Path(a.libro)).items():
        for n, c in contenido[1:]:
            v = [c.get(k) for k in "ABCDEF"]
            if all(x in (None, "") for x in v):
                continue
            filas.append({"hoja": hoja, "fila_excel": n, "video_original": v[0], "infraccion": (v[1] or "").strip().upper(),
                          "id_original": v[2], "ini": a_hora(v[3]), "fin": a_hora(v[4]), "accion": (v[5] or "").strip()})
    d = pd.DataFrame(filas)
    d[["ini", "fin", "accion"]] = d[["ini", "fin", "accion"]].fillna("")
    d["correccion"], d["motivo"], d["sugerencia"], d["verificada"], d["forzar_estado"] = "", "", "", False, None
    ruta_corr = RAIZ / P["rutas"]["correcciones"].format(anotador=a.anotador)
    corr = pd.read_csv(ruta_corr, dtype=str) if ruta_corr.exists() else pd.DataFrame(columns=["hoja", "fila_excel", "campo", "valor", "tipo"])
    for c in corr.itertuples():
        m = (d.hoja == c.hoja) & (d.fila_excel == int(c.fila_excel))
        if m.sum() != 1:
            raise SystemExit(f"Corrección sin fila única: {c.hoja} fila {c.fila_excel}")
        campo = {"id": "id_original", "ini": "ini", "fin": "fin", "infraccion": "infraccion"}.get(c.campo)
        if campo:
            d.loc[m, "correccion"] += f"{c.campo}: {d.loc[m, campo].iloc[0] or 'vacío'} -> {c.valor} ({c.tipo}); "
            d.loc[m, campo] = c.valor
        elif c.campo == "estado":   # decisión por revisión visual: la fila se conserva con este estado
            d.loc[m, "forzar_estado"] = c.valor
            d.loc[m, "correccion"] += f"{c.evidencia} ({c.tipo}); "
        else:
            d.loc[m, "correccion"] += f"{c.valor} ({c.tipo}); "
            d.loc[m, "verificada"] = True
    d["video_id"] = d.video_original.fillna("").str.extract(r"(\d{4}-\d{2}-\d{2}_(?:manana|tarde)_h\d)", expand=False)
    d["track_id"] = pd.to_numeric(d.id_original, errors="coerce").astype("Int64")

    datos = {}

    def cargar(vid):
        if vid not in datos:
            f = ind.loc[vid]
            c = p["rutas"]["salidas"] / f.sesion / vid
            tr = pd.read_csv(c / f"{vid}_trayectorias_vehiculos.csv.gz")
            tr["px"], tr["py"] = (tr.x1 + tr.x2) / 2, tr.y2
            datos[vid] = {"f": f, "tl": set(pd.read_csv(c / f"{vid}_tracklets.csv").track_id),
                          "por_id": dict(tuple(tr.groupby("track_id")))}
        return datos[vid]

    for vid in d.video_id.dropna().unique():
        cargar(vid)
    for vid in d.video_id.dropna().unique():   # ¿el bloque cuadra mejor con otro video del índice?
        g10 = set(d[(d.video_id == vid) & (d.infraccion == "G10")].track_id.dropna())
        if len(g10) < cfg["min_ids_para_reasignar"]:
            continue
        def frac(v):
            pid = cargar(v)["por_id"]
            return np.mean([tid in pid and zonas.dentro(pid[tid].px, pid[tid].py, carril).any() for tid in g10])
        propio = frac(vid)
        if propio >= cfg["min_frac_g10_en_carril"]:
            continue
        mejor = max(ind.index, key=frac)
        if frac(mejor) >= cfg["min_frac_g10_en_carril"]:
            m = d.video_id == vid
            d.loc[m, "correccion"] = (f"video {vid} -> {mejor}: {frac(mejor):.0%} de sus G10 entran al carril en {mejor} "
                                      f"y {propio:.0%} en {vid}")
            d.loc[m, "video_id"] = mejor

    estados = []
    for r in d.itertuples():
        motivo, sug, est = [], [], "ok"
        if not isinstance(r.video_id, str):
            estados.append(("vacia", "fila sin video ni ID: no contiene una anotación", "", None, None))
            continue
        D = datos[r.video_id]
        f, fps, n = D["f"], float(D["f"].fps), int(D["f"].cuadros)
        t0 = datetime.fromisoformat(f"{f.fecha}T{f.hora_cuadro0}")
        k = lambda h: round((datetime.fromisoformat(f"{f.fecha}T{h}") - t0).total_seconds() * fps)  # noqa: E731
        fin = r.fin
        if r.accion[:1].isdigit() and not fin:
            fin = a_hora(r.accion)
            d.at[r.Index, "correccion"] = "hora de fin tomada de la columna Acción"
        if pd.isna(r.track_id) and not (r.ini or fin or r.accion):
            estados.append(("vacia", "fila sin ID ni horas: no contiene una anotación", "", None, None))
            continue
        if r.infraccion not in cfg["infracciones"]:
            estados.append(("descartada", f"infracción '{r.infraccion}'", "", None, None))
            continue
        if pd.isna(r.track_id):
            vacia = not (r.ini or fin or r.accion)
            estados.append(("vacia" if vacia else "revisar", "fila sin ID ni horas: no contiene una anotación" if vacia else "sin ID", "", None, None))
            continue
        t = D["por_id"].get(int(r.track_id))
        if t is None:
            estados.append(("revisar", "el ID no existe en el video", "", None, None))
            continue
        if int(r.track_id) not in D["tl"]:
            motivo.append("fuera del corpus: menos de 1 s en el área (válida como referencia)")
        ki = kf = None
        if r.infraccion != "G10" or r.ini:
            if not (r.ini and fin):
                estados.append(("revisar", "sin hora de inicio o de fin", "", None, None))
                continue
            try:
                ki, kf = k(r.ini), k(fin)
            except ValueError:
                estados.append(("revisar", f"hora ilegible: '{r.ini}' / '{fin}'", "", None, None))
                continue
            if kf < ki:
                est = "revisar"
                motivo.append("fin antes del inicio")
                if ((t.frame >= kf) & (t.frame <= ki)).any():
                    sug.append(f"invertir: {fin}-{r.ini}")
            tol = round(cfg["tolerancia_bordes_s"] * fps)   # la anotación tiene resolución de 1 s
            if -tol <= ki < 0:
                ki = 0
            if n - 1 < kf <= n - 1 + tol:
                kf = n - 1
            elif 0 <= ki < n and n - 1 < kf <= n - 1 + round(cfg["max_continuacion_s"] * fps):
                kf = n - 1   # el evento sigue en el archivo de la hora siguiente
                motivo.append("el evento continúa en el video de la hora siguiente; se cuenta hasta el último cuadro")
            fuera = not (0 <= ki < n and 0 <= kf < n)
            presente = ((t.frame >= min(ki, kf)) & (t.frame <= max(ki, kf))).any()
            vida0, vida1 = t.frame.min(), t.frame.max()
            if fuera or not presente:
                est = "revisar"
                motivo.append("hora fuera del video" if fuera else "el ID no aparece en ese intervalo")
                horas_video = {(t0 + timedelta(seconds=x / fps)).hour for x in (0, n - 1)}
                fmt = lambda x: x.strftime("%H:%M:%S")  # noqa: E731
                hi, hf = datetime.strptime(r.ini, "%H:%M:%S"), datetime.strptime(fin, "%H:%M:%S")
                deltas = [timedelta(minutes=m) for m in cfg["desfases_sugeridos_min"]] + [timedelta(seconds=x) for x in cfg["desfases_sugeridos_s"]]
                opciones = set()
                for dl in deltas:   # digitación en un extremo o el mismo error en ambos (hora, decena de minutos o de segundos)
                    opciones |= {(fmt(hi + dl), fin), (r.ini, fmt(hf + dl)), (fmt(hi + dl), fmt(hf + dl))}
                for x in horas_video:
                    opciones |= {(fmt(hi.replace(hour=x)), fin), (r.ini, fmt(hf.replace(hour=x))), (fmt(hi.replace(hour=x)), fmt(hf.replace(hour=x)))}
                puntaje = {}
                for o in opciones:   # fracción del intervalo corregido que cae dentro de la vida del ID
                    a2, b2 = k(o[0]), k(o[1])
                    if 0 <= a2 <= b2 < n and (b2 - a2) / fps <= cfg["max_duracion_s"]:
                        dentro = max(0, min(b2, vida1) - max(a2, vida0) + 1) / (b2 - a2 + 1)
                        if dentro >= cfg["min_frac_en_vida"]:
                            puntaje[o] = dentro
                if puntaje:
                    mejor = max(puntaje.values())
                    for o in sorted(x for x, v in puntaje.items() if v == mejor):
                        sug.append(f"¿{o[0]}-{o[1]}?")
                vida = t.frame.agg(["min", "max"]).tolist()
                sug.append("vida del ID " + " a ".join((t0 + timedelta(seconds=x / fps)).strftime("%H:%M:%S") for x in vida))
        if r.infraccion == "G10":
            en = zonas.dentro(t.px, t.py, carril).sum()
            if en == 0 and not r.verificada:
                est = "revisar"
                motivo.append("el punto de apoyo nunca entra al carril")
        if r.infraccion == "G41" and ki is not None and est == "ok":
            w = t[(t.frame >= ki) & (t.frame <= kf)]
            if len(w) and not zonas.dentro(w.px, w.py, cruce).any():
                motivo.append("el punto de apoyo no toca el crucero (puede estar dentro de la intersección)")
        estados.append((est, "; ".join(motivo), "; ".join(sug), ki, kf))
    d[["estado", "motivo", "sugerencia", "k_ini", "k_fin"]] = pd.DataFrame(estados, index=d.index)
    d["k_ini"], d["k_fin"] = d.k_ini.astype("Int64"), d.k_fin.astype("Int64")
    f_est = d.forzar_estado.notna()
    d.loc[f_est, "estado"] = d.loc[f_est, "forzar_estado"]
    d = d.drop(columns="forzar_estado")
    clave = ["video_id", "infraccion", "track_id", "ini", "fin"]
    validas = d.track_id.notna() & ~d.estado.isin(["vacia", "descartada"])
    primera = d[validas].groupby(clave, dropna=False).fila_excel.transform("first")
    dup = d[validas].duplicated(clave, keep="first").reindex(d.index, fill_value=False)
    d.loc[dup, "motivo"] = "repetida: se cuenta una vez, en la fila " + primera[dup].astype(int).astype(str)
    d.loc[dup, "estado"] = "fusionada"
    d["repeticiones"] = d[validas].groupby(clave, dropna=False).fila_excel.transform("size").reindex(d.index)
    d["tipo_evento"] = d.accion.str.upper().map(cfg["acciones_g08"])
    d.to_csv(salida / "validacion_filas.csv", index=False)

    cols = ["video_id", "track_id", "ini", "fin", "k_ini", "k_fin", "estado", "motivo", "correccion", "sugerencia", "repeticiones", "hoja", "fila_excel"]
    for inf in cfg["infracciones"]:
        x = d[(d.infraccion == inf) & d.estado.isin(["ok", "revisar", "no_confirmada"])]
        x = x.rename(columns={"ini": "hora_inicio", "fin": "hora_fin"})
        c = [{"ini": "hora_inicio", "fin": "hora_fin"}.get(k, k) for k in cols]
        if inf == "G08":
            c.insert(4, "tipo_evento")
        x[c].sort_values(["video_id", "track_id"]).to_csv(salida / f"etiquetas_{inf.lower()}.csv", index=False)
    cob = d[d.estado == "ok"].groupby(["video_id", "infraccion"]).agg(filas=("estado", "size"), hora_min=("ini", "min"), hora_max=("fin", "max"))
    cob.to_csv(salida / "cobertura.csv")
    candidatos_g10(p, ind, carril, d, salida, cfg)
    print(d.groupby(["infraccion", "estado"]).size().unstack(fill_value=0).to_string())
    print(cob.to_string())


if __name__ == "__main__":
    main()
