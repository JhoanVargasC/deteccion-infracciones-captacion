# Protocolo de anotación

Esta carpeta contiene el protocolo que siguieron los anotadores para registrar las infracciones G08, G10,
G41 y M21 en las ocho horas de video (Etapa 2 de la metodología). Hay un protocolo por infracción, y todos
comparten las reglas comunes de este documento.

| Archivo | Infracción | Unidad anotada |
|---|---|---|
| `protocolo_g41.md` | G41: detenerse sobre el crucero peatonal o dentro de la intersección | Detención del vehículo |
| `protocolo_g10.md` | G10: circular por el carril exclusivo sin ser bus del corredor | Paso del vehículo por el carril |
| `protocolo_g08.md` | G08: recoger o dejar pasajeros sin utilizar el carril derecho | Evento de ascenso o descenso de un automóvil |
| `protocolo_m21.md` | M21: detenerse mientras el resto del tránsito circula | Detención del vehículo |
| `plantillas/` | Hoja de registro de cada infracción y hoja de cobertura | |
| `referencias/` | Imagen de la escena sin vehículos, con y sin las zonas, de día y de noche | |

## Material

Cada hora de video se exportó con las cajas y los números de seguimiento dibujados sobre la imagen. El
número de cada caja identifica al vehículo en el registro. La hora que muestra el video es la referencia
temporal de todas las anotaciones. Los anotadores no consultan las salidas de las reglas ni del modelo.

## Reglas comunes

- **Registro:** una hoja por infracción y por video, a partir de las plantillas. Las columnas con categorías
  solo admiten los valores listados en cada protocolo.
- **Resolución temporal:** 1 s. La hora de inicio es el primer segundo en que se cumple la condición, y la
  de fin, el último.
- **Confianza:** `seguro` o `dudoso`. Un caso dudoso se registra igual, con una nota que lo explica.
- **Errores del seguimiento:** si el vehículo cambia de número durante el caso, se registran todos sus
  números (`45;52`); si no tiene caja, se registra `sin_id` y su posición en la nota; si una caja no
  corresponde a un vehículo, se registra como `error_deteccion`.
- **Cobertura:** al terminar cada video, se registra qué infracción se revisó, entre qué horas del video y
  si la revisión fue completa. Una revisión completa permite tratar como negativo todo lo que no se anotó.
- **Orden de trabajo:** una revisión por infracción sobre cada video, porque cada infracción exige mirar
  una parte distinta de la escena.
