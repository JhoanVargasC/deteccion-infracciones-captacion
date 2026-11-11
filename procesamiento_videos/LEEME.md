# Procesamiento de videos

Convierte cada hora de video de la cámara fija en el material que se usa para anotar:
detecciones, trayectorias, recortes, manifiesto, tabla de tracklets y video anotado.

## Orden de ejecución

Desde esta carpeta, con el entorno que tenga Ultralytics y torch-directml:

| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| 0 | `src/indice.py` | Lista los videos y añade la hora de inicio medida con el reloj en pantalla | `datos/indice_videos.csv` |
| 1 | `src/correr.py --etapas detectar` | Detección con YOLO26m sobre el área de reconocimiento; reanuda si se corta | `<video_id>_detecciones.csv.gz` y `.json` |
| 2 | `src/correr.py --etapas seguir` | Supresión de duplicados, descarte de ocupantes y seguimiento con TrackTrack | `<video_id>_trayectorias_*.csv.gz`, `_seguimiento.json` |
| 3 | `src/correr.py --etapas recortar` | Recortes, manifiesto, tabla de tracklets y video para la anotación | `recortes/`, `_manifiesto.csv`, `_tracklets.csv`, `_video_anotado.mp4` |

Si no se indica `--etapas`, `correr.py` ejecuta las tres en orden. Ejemplos:

```
python src/indice.py
python src/correr.py --sesion 2026-04-06_manana
python src/correr.py --videos 2026-04-06_tarde_h3 --limite-segundos 180 --salida prueba
```

Los archivos de cada video quedan en `salidas/<sesion>/<video_id>/`. Los nombres y el
contenido de cada archivo están en `docs/convenciones.md`.

## Código (`src/`)

| Archivo | Función |
|---|---|
| `correr.py` | Orquestador: etapas sobre videos o sesiones, con registro y reanudación |
| `indice.py` | Construye el índice de videos |
| `detectar.py` | Detección |
| `seguir.py` | Seguimiento |
| `recortar.py` | Recortes y video para la anotación |
| `compat_directml.py` | Parche de Ultralytics para la GPU AMD; se importa antes que Ultralytics |
| `configuracion.py`, `video.py`, `zonas.py` | Parámetros, lectura secuencial y reloj, zonas |

## Configuración

- `config/parametros.yaml`: todos los valores y rutas, cada uno con su justificación.
  Las rutas `videos_entrada` y `salidas_secundaria` se deben ajustar a la ubicación local.
- `config/zonas.json`: polígonos de las zonas de la escena (1920x1080 px).

## Entradas que no se incluyen

- Los videos originales (carpeta indicada en `rutas.videos_entrada`).
- `datos/lecturas_reloj.csv`: lectura manual del reloj en pantalla, una fila por video.
- Los pesos `modelos/yolo26m.pt`.

Las carpetas `datos/`, `salidas/`, `registros/` y `temporal/` se crean al ejecutar.
