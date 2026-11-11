# Ficha del conjunto de datos

Esta ficha describe el conjunto de datos del trabajo según la estructura de las fichas de conjuntos de
datos (*datasheets for datasets*) de Gebru et al. (2021, *Communications of the ACM*, 64(12), 86-92).

## Motivación

| Apartado | Descripción |
|---|---|
| Propósito | Evaluar la detección automática de las infracciones G08, G10, G41 y M21 a partir del video de una cámara fija. |
| Responsables | Los autores del trabajo final del curso Ingeniería de la Información II (Universidad del Pacífico). |

## Composición

| Apartado | Descripción |
|---|---|
| Video | Ocho archivos de una hora, grabados el 1 y el 6 de abril de 2026 entre las 7:57 y las 9:57 a. m. y entre las 5:57 y las 7:57 p. m.; 1920×1080 píxeles, 15 fotogramas por segundo; 432 399 fotogramas; 30 tramos de 10 minutos de día y 18 de noche. |
| Detecciones y seguimiento | 14 254 *tracklets* de vehículos con al menos 1 s en el área de reconocimiento y 22 614 de personas. |
| Recortes | 1 416 578 imágenes alrededor de los vehículos, tres por segundo, en 14 254 carpetas, una por *tracklet*, con un índice de vehículo, fotograma y posición de la caja. |
| Tablas por infracción | Una fila por unidad de análisis: 14 254 *tracklets* en G10, 5 399 vehículos que se detienen junto al crucero en G41, 12 037 automóviles en G08 y 14 383 detenciones estrictas en M21, con su etiqueta, su período de iluminación y, en G08, su partición. |
| Anotaciones | Casos registrados por los anotadores: 1 962 en G10, 1 555 en G41, 48 en G08 y 33 en M21. Unidades con al menos un caso: 1 961 *tracklets* en G10, 1 508 vehículos en G41 y 45 automóviles en G08. |

## Recolección

| Apartado | Descripción |
|---|---|
| Fuente | Cámara municipal ubicada en la intersección de las avenidas Javier Prado Este y Paseo Parodi, en San Isidro, Lima. El video se obtuvo mediante una solicitud de acceso a la información pública a la Municipalidad de San Isidro. |
| Anotación | Cuatro anotadores revisaron completas las ocho horas según el protocolo de [`protocolo_anotacion/`](protocolo_anotacion/LEEME.md). |
| Validación | Un oficial en retiro de la Policía Nacional del Perú revisó 318 clips: una muestra de 254 casos anotados, definida por un plan de muestreo de aceptación por atributos (p0 = 0,05; α = 0,05; a lo sumo un error), y 64 casos de control sin infracción anotada. Encontró un error entre los 33 casos de M21 y ninguno en G10, G41 y G08, por lo que las etiquetas de las cuatro infracciones se aceptan. En los controles señaló un evento de G08 no anotado. |

## Procesamiento

| Apartado | Descripción |
|---|---|
| Etapas | Detección con YOLO26m, seguimiento con TrackTrack, recortes y señales visuales ([`procesamiento_videos/`](procesamiento_videos/), [`etiquetado/`](etiquetado/)). |
| Partición | Solo G08 se divide: tramos de 10 minutos asignados en ciclo a cinco particiones; P0 a P2 forman *train* y P3 y P4, *test* ([`entrenamiento/`](entrenamiento/)). |

## Usos

| Apartado | Descripción |
|---|---|
| Uso previsto | Investigación sobre la detección automática de infracciones de tránsito. |
| Usos no previstos | Identificar personas o vehículos, o sancionar sin revisión humana. |

## Distribución y privacidad

| Apartado | Descripción |
|---|---|
| Datos personales | El video y los recortes muestran rostros y placas; el procesamiento no los identifica. |
| Acceso | El video, los recortes, las hojas de anotación y las salidas intermedias no se publican, porque contienen datos personales. Este repositorio publica el código, los parámetros, el protocolo y las plantillas de anotación. |
