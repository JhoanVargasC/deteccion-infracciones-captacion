# G41: detención sobre el crucero peatonal o dentro de la intersección

**Norma.** «Estacionar o detener el vehículo sobre la línea demarcatoria de intersección, dentro de éstas o
en el crucero peatonal (paso peatonal)» (Reglamento Nacional de Tránsito, DS 016-2009-MTC).

**Unidad anotada.** La detención del vehículo: queda quieto en el mismo lugar durante al menos 2 s.

## Procedimiento

1. Cada vez que el tránsito de la avenida se detiene ante el crucero, se registran el vehículo más
   adelantado de cada carril y todo vehículo que quede detenido sobre el crucero o dentro de la
   intersección.
2. Para cada uno se registran la hora de inicio y de fin de la detención.
3. Se registra la posición del frente del vehículo durante la detención. Si el frente se mueve, se elige
   la posición que mantiene la mayor parte del tiempo.

Los vehículos avanzan hacia la cámara: el frente es el extremo más cercano a la parte inferior de la imagen.
El borde del crucero que importa es el superior, por donde llegan los vehículos (`referencias/zonas_dia.png`).

## Posición del frente (`posicion`)

| Valor | Cuándo |
|---|---|
| `antes_del_crucero` | El frente no llega a la línea de parada |
| `entre_linea_y_crucero` | El frente pasó la línea de parada pero no toca el crucero |
| `sobre_crucero` | El frente está sobre el crucero |
| `dentro_interseccion` | El frente pasó el crucero y está en la intersección |
| `dudoso` | No se puede decidir entre dos categorías vecinas |
| `error_deteccion` | La caja no corresponde a un vehículo |

**Caso positivo:** `posicion` es `sobre_crucero` o `dentro_interseccion`.

## Hoja de registro (`plantillas/g41.csv`)

`video_id`, `track_id`, `clase`, `carril`, `hora_inicio`, `hora_fin`, `posicion`, `agente` (si hay un
agente de tránsito dirigiendo), `confianza`, `nota`, `anotador`, `fecha_anotacion`, `version_protocolo`.
