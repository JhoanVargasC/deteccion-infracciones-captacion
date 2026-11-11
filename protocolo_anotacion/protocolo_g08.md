# G08: ascenso o descenso de pasajeros de un automóvil

**Norma.** «No utilizar el carril derecho para recoger o dejar pasajeros o carga» (Reglamento Nacional de
Tránsito, DS 016-2009-MTC). En esta vía, el carril derecho es el carril exclusivo, que los automóviles no
pueden ocupar; por ello, todo ascenso o descenso de pasajeros de un automóvil configura G08.

**Unidad anotada.** El evento de ascenso o descenso, asociado a su automóvil.

## Procedimiento

1. Se revisa el video mirando a las personas junto a los automóviles y a los automóviles que se detienen.
2. Ante un posible evento, se retrocede y se mira cuadro a cuadro.
3. Se registran la hora de inicio y de fin:
   - **ascenso:** desde que la persona llega a la puerta hasta el último segundo en que se la ve fuera del
     automóvil;
   - **descenso:** desde que la persona empieza a salir hasta que camina separada del automóvil.
4. Si varias personas suben o bajan del mismo automóvil en una misma detención, es un solo evento, con el
   número total de personas. Dos detenciones separadas del mismo automóvil son dos eventos.

Solo se anotan automóviles (incluidos taxis y colectivos de ese tamaño). Una persona que se acerca a hablar
con el conductor y se va sin subir no es un evento.

## Valores

| Columna | Valores |
|---|---|
| `tipo` | `ascenso`, `descenso` o `ambos` |
| `n_personas` | número de personas que suben o bajan |
| `carril` | carril donde ocurre el evento |
| `estado_transito` | `circula`, `detenido` o `no_determinable` |

**Caso positivo:** todo evento registrado con `confianza = seguro`.

## Hoja de registro (`plantillas/g08.csv`)

`video_id`, `track_id`, `tipo`, `n_personas`, `hora_inicio`, `hora_fin`, `carril`, `estado_transito`,
`confianza`, `nota`, `anotador`, `fecha_anotacion`, `version_protocolo`.
