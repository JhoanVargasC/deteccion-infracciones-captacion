# Detección de infracciones de tránsito asociadas a la captación de pasajeros

Código del trabajo final del curso **Ingeniería de la Información II** (Universidad del Pacífico, 2026):
*Técnicas de visión computacional para la detección automática de infracciones de tránsito asociadas a la
captación de pasajeros a través de señales visuales*.

El procedimiento detecta cuatro infracciones del Reglamento Nacional de Tránsito a partir del video de una
cámara fija en la intersección de las avenidas Javier Prado Este y Paseo Parodi, en Lima:

| Código | Infracción | Decisión |
|---|---|---|
| G10 | Circular por el carril exclusivo sin ser bus del corredor | Regla explícita |
| G41 | Detenerse sobre el crucero peatonal o dentro de la intersección | Regla explícita |
| M21 | Detenerse mientras el resto del tránsito circula | Regla explícita |
| G08 | Recoger o dejar pasajeros sin utilizar el carril derecho | Modelo entrenado |

## Etapas y código

El repositorio sigue las cinco etapas de la metodología:

| Etapa | Qué produce | Carpeta |
|---|---|---|
| 1. Detección y seguimiento | Detecciones con YOLO26m, *tracklets* con TrackTrack, recortes de los vehículos y video para la anotación | `procesamiento_videos/` |
| 2. Conjunto de datos etiquetado | Protocolo de anotación, anotaciones de referencia, conjunto de datos, muestra para la validación por experto y partición de G08 | `protocolo_anotacion/`, `etiquetado/`, `entrenamiento/` |
| 3. Extracción de señales | Posición, detención estricta, cajas falsas, estado del tránsito, contexto del carril, interacción y apariencia (DINOv2) | `etiquetado/src/candidatos.py`, `entrenamiento/src/g08_*.py`, `entrenamiento/src/evaluar_m21.py` |
| 4. Reglas y modelo | Reglas de G10, G41 y M21; modelo de G08 con selección de candidatos, dos ramas y fusión tardía | `entrenamiento/` |
| 5. Evaluación | Precisión, recall, F1, AP y AUROC, con intervalos *bootstrap* por vehículo, en total y por período | `entrenamiento/` |

Cada carpeta tiene un `LEEME.md` con el orden de ejecución y un `config/parametros.yaml` con todos los
parámetros. Ningún parámetro está escrito en el código.

## Resultados principales

| Infracción | Unidad | Medida | Total | Día | Noche |
|---|---|---|---|---|---|
| G10 | *tracklet* | F1 | 0,980 | 0,985 | 0,969 |
| G41 | vehículo | F1 | 0,970 | 0,970 | 0,971 |
| M21 | detención | Precisión | 0,94 | 1,00 | 0,87 |
| G08 | automóvil candidato (*test*) | AP (referencia) | 0,201 (0,026) | | |

La validación por experto revisó 318 clips (254 casos anotados y 64 controles). Encontró un error en
los casos de M21 y ninguno en las demás infracciones, por lo que las etiquetas de las cuatro se aceptan.

## Documentación

| Documento | Contenido |
|---|---|
| [`protocolo_anotacion/`](protocolo_anotacion/LEEME.md) | Protocolo de anotación: reglas comunes, un protocolo por infracción, plantillas de registro e imágenes de referencia de la escena |
| [`ficha_conjunto_datos.md`](ficha_conjunto_datos.md) | Ficha del conjunto de datos: composición, recolección, anotación, validación, partición, usos y privacidad |

## Ejecución

1. `procesamiento_videos/`: `python src/correr.py --sesion <sesión>` (detección, seguimiento y recortes).
2. Anotación manual con los protocolos de `protocolo_anotacion/`, sobre el video con los números de
   seguimiento.
3. `etiquetado/`: validación y consolidación de las anotaciones, señales de detención y muestra para la
   validación por experto.
4. `entrenamiento/`: `python src/correr.py` (conjunto de datos, reglas de G10 y G41, y modelo de G08) y
   `python src/evaluar_m21.py` (regla de M21).

## Video de validación visual

[Video de 4 minutos](https://drive.google.com/file/d/16ZAoR0Cd-mw37AGDydFj3XhDTbHiiz0E/view?usp=sharing) con casos reales de cada infracción (aciertos y errores). Cada clip muestra el plano completo de la cámara, con las reglas y el modelo aplicados a todos los vehículos de la escena.

## Datos no incluidos

El video completo de las ocho horas, las hojas de anotación, las salidas intermedias y los pesos de YOLO26m no se incluyen, porque
el video muestra rostros y placas de personas en la vía pública. El video de validación visual, con clips seleccionados, sí se publica. Las rutas a esos datos se ajustan en el
`config/parametros.yaml` de cada carpeta.

## Entorno

Python 3.11 en Windows, con una GPU AMD mediante `torch-directml`; sin ella, el código corre en la CPU.
Las dependencias y sus versiones están en `requirements.txt`.
