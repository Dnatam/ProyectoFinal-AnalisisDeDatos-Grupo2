# 2.3 ¿Las predicciones con menos evidencia se equivocan más?

## Resultado directo

Las 320 filas de prueba se agruparon en quintiles de $N_{ef}$ del bosque:

| Quintil | Rango de $N_{ef}$ | Con gemelo en train | Distancia mediana al vecino más cercano | MAE |
|---|---|---|---|---|
| Q1 (menor evidencia) | 3.4 – 15.2 | 57 de 64 | 0.00 | **0.13** |
| Q2 | 15.4 – 44.4 | 16 de 64 | 0.73 | 0.43 |
| Q3 | 44.4 – 60.3 | 1 de 64 | 1.10 | 0.54 |
| Q4 | 60.6 – 76.9 | 0 de 64 | 0.98 | 0.54 |
| Q5 (mayor evidencia) | 77.0 – 148.4 | 0 de 64 | 1.05 | 0.33 |

**La respuesta es no:** el quintil con menos evidencia es el que **menos** se equivoca, lo
contrario de lo esperado.

## Por qué: Wine tiene filas repetidas

El archivo tiene **240 filas repetidas** de 1 599. Con la partición `random_state=0`, **74 de
las 320 filas de prueba tienen un gemelo exacto en entrenamiento** (las mismas 11 variables).
De esas 74, 57 están en Q1.

| Grupo | Filas | MAE |
|---|---|---|
| Con gemelo en train | 74 | **0.09** |
| Sin gemelo | 246 | 0.49 |

El mecanismo es este: el bosque tiene `min_samples_leaf = 2` y no usa bootstrap, así que cada
árbol ve todas las filas. Un vino de prueba con gemelo cae, en casi todos los árboles, en una
hoja diminuta junto a ese gemelo. El gemelo se lleva una parte enorme del peso, con lo que
$N_{ef}$ baja y $c(x)$ sube (la mediana de $c(x)$ en Q1 es 0.89). Y como el gemelo casi siempre
tiene la misma nota, la predicción acierta.

Entonces, un $N_{ef}$ bajo **no indica falta de evidencia: indica que la respuesta se copió del
entrenamiento**. Es una forma de **fuga de datos**: esas filas de prueba no son realmente vinos
nuevos para el modelo, y el error de prueba global sale más optimista de lo que sería con vinos
nunca vistos. La columna de distancia lo confirma: en Q1, la mediana de la distancia al vino de
entrenamiento más cercano es **0**.

## Sin los gemelos, aparece la relación esperada

Recalculando los quintiles solo con las 246 filas sin gemelo:

| Quintil | Rango de $N_{ef}$ | MAE |
|---|---|---|
| Q1 (menor evidencia) | 3.4 – 41.6 | 0.53 |
| Q2 | 42.1 – 53.0 | 0.56 |
| Q3 | 53.5 – 68.7 | 0.53 |
| Q4 | 68.9 – 81.6 | 0.49 |
| Q5 (mayor evidencia) | 82.6 – 148.4 | **0.33** |

Ahora sí, **más evidencia va con menos error**, aunque la relación no es gradual: los
primeros cuatro quintiles tienen un error parecido (alrededor de 0.5) y la mejora fuerte está
en Q5, con $N_{ef} \ge 83$.

## ¿Por qué Q5 se equivoca tan poco?

Las notas reales de cada quintil (con todas las filas):

| Quintil | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Q1 | 0 | 0 | 21 | 35 | 8 | 0 |
| Q2 | 1 | 4 | 27 | 25 | 6 | 1 |
| Q3 | 1 | 3 | 19 | 31 | 9 | 1 |
| Q4 | 0 | 4 | 30 | 26 | 3 | 1 |
| Q5 | 0 | 0 | **38** | 25 | 1 | 0 |

Q5 está formado por vinos **típicos**: casi todos de nota 5 o 6, y ninguno de 3, 4 u 8. Son
vinos en la zona más densa de los datos, rodeados de muchos vinos parecidos. El bosque reparte
el peso entre muchos de ellos (por eso $N_{ef}$ es alto), y promediar muchas notas de 5 y 6
funciona bien. En cambio, Q2 a Q4 contienen los vinos de notas extremas, que son justo
donde el modelo falla (ver 2.4, criterio 4).

## Conclusión

Con estos datos, $N_{ef}$ sirve, pero **solo después de separar los duplicados**:

1. $N_{ef}$ muy bajo con un gemelo en entrenamiento: la predicción es una copia de la nota
   del gemelo. Acierta, pero no mide la capacidad del modelo de predecir vinos nuevos.
2. Sin gemelo: $N_{ef}$ alto ($\ge 83$) identifica las predicciones más confiables.
   Por debajo, el error es de aproximadamente medio punto de nota.

Lo más importante para el resto del proyecto: **el error de prueba del bosque está inflado a
favor por los duplicados**. Si se quitan las filas repetidas antes de partir los datos, es de
esperar que su ventaja sobre los otros modelos se reduzca.