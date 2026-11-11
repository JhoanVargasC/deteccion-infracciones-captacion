# G10: uso del carril exclusivo

**Norma.** «Incumplir las disposiciones sobre el uso de las vías de tránsito rápido y/o de acceso
restringido» (Reglamento Nacional de Tránsito, DS 016-2009-MTC). El carril exclusivo del corredor es una vía
de acceso restringido.

**Unidad anotada.** El paso del vehículo por el carril exclusivo.

## Procedimiento

1. Se revisa el video mirando el carril exclusivo (`referencias/zonas_dia.png`).
2. Se registra todo vehículo que entra al carril y no es un bus del corredor, que tiene permitido usarlo.
3. Se registran la hora de entrada y la de salida del carril, la ocupación y el tipo de vehículo.

## Valores

| Columna | Valores |
|---|---|
| `ocupacion` | `circula` (avanza dentro del carril), `parcial` (lo ocupa solo en parte), `fuera` (parecía estar en el carril pero no lo está), `error_deteccion` |
| `tipo` | `bus_otro`, `auto`, `taxi`, `van`, `camion`, `moto`, `emergencia` (ambulancia o bomberos), `patrullero`, `otro` |

**Caso positivo:** `ocupacion = circula` y `tipo` distinto de `emergencia` y `patrullero`.

## Hoja de registro (`plantillas/g10.csv`)

`video_id`, `track_id`, `hora_entrada`, `hora_salida`, `ocupacion`, `tipo`, `confianza`, `nota`, `anotador`,
`fecha_anotacion`, `version_protocolo`.
