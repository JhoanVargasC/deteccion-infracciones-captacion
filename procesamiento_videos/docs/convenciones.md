# Convenciones

## Nombres

- Minúsculas, sin tildes ni espacios, con guion bajo.
- Sesión: `AAAA-MM-DD_manana` o `AAAA-MM-DD_tarde`. Las sesiones empiezan a las 06:57 o a las 16:57.
- Video: `<sesion>_h<N>`, donde N es la hora dentro de la sesión (1 a 4), tomada del número
  entre paréntesis del archivo original. Ejemplo: `2026-04-06_tarde_h3`.
- Archivos generados: llevan el `video_id` delante, para que sigan identificados si se
  mueven: `2026-04-06_tarde_h3_detecciones.csv.gz`.
- Recortes: `recortes/<track_id de 5 dígitos>/<video_id>_t<track_id>_f<cuadro de 6 dígitos>.jpg`.
- Tracklets: `track_id` es local a cada video. La clave única es (`video_id`, `track_id`).
- Registros: `registros/AAAAMMDD-HHMMSS_<etapas>_<videos o sesión>.log`.

## Salidas de cada video (`salidas/<sesion>/<video_id>/`)

| Archivo | Paso | Contenido |
|---|---|---|
| `_detecciones.csv.gz` / `.json` | 1 | frame, clase, conf, x1, y1, x2, y2 (cuadro completo) |
| `_trayectorias_vehiculos.csv.gz`, `_trayectorias_personas.csv.gz` | 2 | frame, track_id, clase, conf, caja |
| `_seguimiento.json` | 2 | Parámetros y conteos |
| `_tracklets.csv` | 3 | Una fila por tracklet vehicular del área |
| `_manifiesto.csv` | 3 | Una fila por recorte |
| `recortes/` | 3 | JPEG sin dibujos |
| `_video_anotado.mp4` | 3 | Área con cajas, número de tracklet y hora |
| `_recortes.json` | 3 | Parámetros, conteos, tamaños y tiempos |

## Tiempo

- Los cuadros se leen siempre en orden desde el 0. La búsqueda por índice de OpenCV no es exacta
  en estos videos.
- La hora absoluta de un cuadro es `hora_cuadro0 + k / fps`. Ambos valores se toman del índice.
- `hora_cuadro0` sale de leer el reloj en pantalla en el primer cuadro en que cambia el segundo
  (`datos/lecturas_reloj.csv`): `hora_cuadro0 = t_ini - k_ini / fps`.

## Formato numérico

El número de decimales al escribir (3 para la confianza y 1 para las coordenadas) es un
formato de salida y está en el código, no en los parámetros.
