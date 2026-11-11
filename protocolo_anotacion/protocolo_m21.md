# M21: detención mientras el resto del tránsito circula

**Norma.** «Estacionar interrumpiendo totalmente el tránsito» (Reglamento Nacional de Tránsito,
DS 016-2009-MTC). La anotación registra la conducta observable más próxima: un vehículo que se detiene
mientras el resto del tránsito circula.

**Unidad anotada.** La detención del vehículo: queda quieto en el mismo lugar durante al menos 2 s.

## Procedimiento

1. Se revisa el video buscando vehículos quietos mientras los de los carriles vecinos avanzan.
2. Las filas ante el semáforo en rojo no se anotan, porque allí todo el tránsito está detenido.
3. Se registran la hora de inicio y de fin de la detención y su situación. Si la situación cambia durante
   la detención, se elige la que describe la mayor parte de ella.
4. Los buses no se anotan, porque sus detenciones forman parte de su operación.

## Situación (`situacion`)

| Valor | Cuándo |
|---|---|
| `detenido_transito_circula` | Detenido, el resto del tránsito circula y no hay otro vehículo detenido delante |
| `detras_de_detenido` | Detenido porque otro vehículo detenido delante le impide avanzar |
| `transito_detenido` | Detenido con el resto del tránsito también detenido |
| `no_detenido` | Parecía detenido pero avanza en marcha lenta |
| `dudoso` | No se puede decidir |
| `error_deteccion` | La caja no corresponde a un vehículo |

**Caso positivo:** solo `detenido_transito_circula`.

## Hoja de registro (`plantillas/m21.csv`)

`video_id`, `track_id`, `clase`, `carril`, `hora_inicio`, `hora_fin`, `situacion`, `motivo_observable`
(`ascenso_descenso`, `carga`, `averia`, `sin_motivo_visible` u `otro`), `confianza`, `nota`, `anotador`,
`fecha_anotacion`, `version_protocolo`.
