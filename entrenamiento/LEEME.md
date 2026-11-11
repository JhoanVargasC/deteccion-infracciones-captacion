# Entrenamiento y evaluación

Construye el conjunto de datos, evalúa las reglas de G10, G41 y M21, y entrena y evalúa el modelo de G08
(Etapas 2 a 5 de la metodología). Lee las salidas de `../procesamiento_videos` y de `../etiquetado`.

- G10, G41 y M21 se deciden con reglas explícitas, que no se entrenan.
- G08 se decide con un modelo de dos partes: la selección de candidatos y la puntuación de sus ventanas
  con una rama de interacción y una rama de apariencia.

## Orden de ejecución

`src/correr.py` ejecuta los pasos en este orden y se detiene en el primero que falla. La salida de cada
paso queda en `registros/<paso>.log`.

```
python src/correr.py                       # todos los pasos
python src/correr.py --desde entrenar      # desde un paso
python src/correr.py --pasos apariencia --reanudar
```

| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| iluminacion | `iluminacion.py` | Período de iluminación de cada tramo de 10 minutos (brillo del cielo, método de Otsu) | `dataset/iluminacion.csv` |
| dataset | `construir_dataset.py` | Una tabla por infracción, con su etiqueta, su período y su partición | `dataset/g10.csv`, `g41.csv`, `g08.csv` |
| validar | `validar_dataset.py` | Comprobaciones del conjunto: cobertura, unidades, particiones y recortes | `resultados/validacion/` |
| reglas | `evaluar_reglas.py` | Precisión, recall y F1 de G10 y G41, en total, de día y de noche | `resultados/reglas/` |
| ventanas | `g08_ventanas.py` | Instantes de muestreo y ventanas de 2 s | `rasgos/g08/` |
| interaccion | `g08_interaccion.py` | Vector de interacción u_w de cada ventana | `rasgos/g08/` |
| poblaciones | `g08_poblaciones.py` | Candidatos y automóviles parecidos | `rasgos/g08/autos.csv` |
| particiones | `particiones.py` | Composición de P0 a P4 y de los conjuntos train y test | `dataset/particiones.md` |
| apariencia | `g08_apariencia.py` | Vectores de apariencia con DINOv2 ViT-S/14 (GPU, reanudable) | `rasgos/g08/apariencia.npy` |
| entrenar | `g08_entrenar.py` | Validación cruzada en train (P0 a P2) y modelo final | `resultados/g08/` |
| evaluar | `g08_evaluar.py` | Peso de la fusión, umbral de alerta y medidas en train y test | `resultados/g08/` |

Se ejecutan aparte:

| Script | Qué hace |
|---|---|
| `evaluar_m21.py --muestra` y `evaluar_m21.py` | Regla de M21: genera los clips para la revisión visual y, con la hoja completada, calcula la precisión y el recall |
| `comparar_configuraciones.py --base A --variante B` | Selección de la configuración de G08: adopta la variante solo si mejora la AP y el intervalo de la diferencia excluye el cero |

Las dos comparaciones de la configuración de G08 se obtienen así:

```
python src/g08_entrenar.py --salida g08_v2                                        # con apariciones netas
python src/g08_entrenar.py --sin-netas --apariencia apariencia_b14 --salida g08_b14   # ViT-B/14
python src/comparar_configuraciones.py --base g08 --variante g08_v2
python src/comparar_configuraciones.py --base g08 --variante g08_b14
```

## Partición de G08

- Cada video se corta en tramos de 10 minutos; el tramo b va a la partición b mod 5.
- Dos tracklets contiguos de un mismo vehículo reciben la partición del primero.
- P0, P1 y P2 forman el conjunto train; P3 y P4, el conjunto test, que solo se usa en la evaluación final.
- Todo lo que se estima con datos (estandarización, ejemplos difíciles, peso de la fusión y umbral de
  alerta) usa solo train.

## Configuración

`config/parametros.yaml` contiene las rutas y todos los parámetros. `config/carriles.json` contiene los
carriles del acceso trazados sobre la imagen, que usa el contexto del carril de M21.
