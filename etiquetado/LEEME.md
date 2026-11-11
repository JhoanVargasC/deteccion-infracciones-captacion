# Etiquetado

Convierte las hojas de anotación en las anotaciones de referencia de cada infracción, calcula las señales
de detención y de tránsito, y prepara la validación por experto (Etapas 2 y 3 de la metodología).
Las hojas originales no se modifican.

## Orden de ejecución

Desde esta carpeta, con el mismo entorno de `procesamiento_videos`. `<anotador>` es el identificador
del anotador (`anotador_a`, `anotador_b`, ...).

| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| 1 | `src/validar_anotaciones.py "<hoja>.xlsm" --anotador <anotador>` | Normaliza la hoja y comprueba cada fila contra las trayectorias (número del vehículo, horas y duración) | `salidas/<anotador>/etiquetas_<inf>.csv`, `validacion_filas.csv`, `cobertura.csv` |
| 2 | `src/sugerir_correcciones.py --anotador <anotador>` | Propone una corrección para las filas con un error de digitación de solución única (número o hora) | `config/correcciones_<anotador>_sugeridas.csv` |
| 3 | `src/validar_anotaciones.py` (otra vez) | Aplica las correcciones revisadas (`config/correcciones_<anotador>.csv`) | igual que el paso 1 |
| 4 | `src/consolidar.py` | Junta las filas válidas de todos los anotadores | `salidas/consolidado/` |
| 5 | `src/candidatos.py` | Detenciones estrictas, cajas falsas, distancia al crucero y estado del tránsito | `salidas/candidatos/<video_id>_detenciones.csv` |
| 6 | `src/etiquetas_finales.py` | Anotaciones de referencia por infracción | `salidas/final/etiquetas_<inf>.csv` |
| 7 | `src/muestra_validacion.py` | Muestra de la validación por experto (plan de muestreo de aceptación) y sus clips | `salidas/validacion_v2/`, clips en la carpeta de entrega |
| 8 | `src/crear_excel_validacion.py` | Hoja que llena el experto (requiere `openpyxl`) | `validacion_infracciones.xlsx` |

`src/exportar_clips.py` contiene las funciones que generan los clips: cada clip muestra un solo vehículo
marcado, sin cajas, números ni zonas, y los casos de control se mezclan con los demás en orden aleatorio.

La clave de un vehículo es (`video_id`, `track_id`), igual que en `procesamiento_videos`.
`salidas/final/` y `salidas/candidatos/` son entradas del módulo `entrenamiento`.

## Estado de una fila

| Estado | Entra como caso anotado | Significado |
|---|---|---|
| `ok` | Sí | Válida, con o sin corrección documentada |
| `revisar` | No | Sin una corrección de solución única; se confirma en el video |
| `fusionada` | No (ya contada) | Repite otra fila |
| `vacia` | No | Fila de plantilla sin número ni horas |

## Configuración

`config/parametros.yaml` reúne las rutas, los criterios de validación de las hojas, los umbrales de las
señales y los parámetros de los clips y de la muestra. Las rutas de entrega se ajustan a la ubicación local.

## Entradas que no se incluyen

Las hojas de anotación, las correcciones y las salidas de `procesamiento_videos`, porque identifican a
personas y vehículos en la vía pública.
