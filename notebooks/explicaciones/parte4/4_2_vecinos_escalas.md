# 4.2 Vecinos en escala cruda y estandarizada — D06.T20

**Responsable:** I1 · **Código:** `src/parte4/vecinos_escalas.py` · **Datos:** Wine (tinto) · **Tablas:** `resultados/tablas/4_2_*.csv`

## Qué se hace

Para las tres consultas de la Parte 2 (2.1: $N_{\text{ef}}$ mediano, $N_{\text{ef}}$ mínimo y máxima concentración), se buscan los 10 vecinos más cercanos en el entrenamiento con distancia euclidiana, primero sobre las variables crudas y luego sobre las estandarizadas, y se cuenta cuántos vecinos comparten los dos conjuntos.

Las consultas se leen de `resultados/tablas/parte2_consultas_evidencia.csv` (columna `Idx_Test`), la tabla que genera `src/parte2/seleccion_consultas.py`, para usar exactamente las mismas filas de prueba.

Para entender **en qué se parecen** los vecinos de cada escala, se reporta la distancia media entre la consulta y sus 10 vecinos en cinco variables clave, medida en desviaciones estándar en ambos casos para poder comparar variables con unidades distintas.

## Vecinos compartidos

| Consulta | Fila de prueba | $y$ real | Vecinos en común | Predicción cruda | Predicción estandarizada |
|---|---|---|---|---|---|
| $N_{\text{ef}}$ mediano | 176 | 6 | **3 de 10** (382, 1222, 253) | 5.7 | 6.2 |
| $N_{\text{ef}}$ mínimo | 67 | 5 | **3 de 10** (219, 1165, 574) | 6.2 | 5.9 |
| Máx. concentración | 12 | 6 | **2 de 10** (479, 240) | 5.2 | 5.5 |

En total, solo 8 de los 30 vecinos coinciden. Los que sí coinciden suelen ser los más cercanos de todos: en las tres consultas, el primer vecino es el mismo en ambas escalas, y en dos de ellas también el segundo. Un vino que se parece a la consulta en todas las variables queda cerca con cualquiera de las dos distancias; el desacuerdo está en el resto del vecindario.

## En qué se parecen los vecinos

Distancia media entre la consulta y sus 10 vecinos, en desviaciones estándar:

| Variable | Fila 176: crudos | Fila 176: estand. | Fila 67: crudos | Fila 67: estand. | Fila 12: crudos | Fila 12: estand. |
|---|---|---|---|---|---|---|
| total sulfur dioxide | **0.036** | 0.405 | **0.027** | 0.316 | **0.015** | 0.268 |
| free sulfur dioxide | **0.048** | 0.390 | **0.057** | 0.333 | **0.019** | 0.238 |
| alcohol | 0.613 | 0.481 | 1.065 | 0.396 | 0.405 | 0.349 |
| volatile acidity | 1.178 | 0.619 | 1.222 | 0.485 | 0.997 | 0.208 |
| sulphates | 0.401 | 0.635 | 0.584 | 0.246 | 0.669 | 0.286 |

- **Vecinos crudos:** casi idénticos a la consulta en sulfitos (a menos de 0.06 desviaciones estándar), pero lejos en todo lo demás. En acidez volátil están en promedio a una desviación estándar o más, es decir, tan lejos como dos vinos cualquiera. Es lo que anticipaba la 4.1: en escala cruda los sulfitos son el 99.5 % de la distancia.
- **Vecinos estandarizados:** sacrifican cercanía en sulfitos (0.24 a 0.41) a cambio de parecerse en todas las variables (entre 0.2 y 0.65). Estandarizar no hace a los vecinos más cercanos en cada variable (en la fila 176 los crudos están más cerca en sulfatos), sino que reparte la cercanía entre todas.

## Lectura

- **Escoger la escala es escoger quiénes votan.** Con las mismas filas de entrenamiento y la misma consulta, cambiar las unidades cambia 7 u 8 de los 10 vecinos. Los testigos de una predicción, que en la Parte 2 sirven para explicarla, no son una propiedad del vino consultado: dependen de la distancia que se eligió.
- **La predicción cambia hasta medio punto de nota** (fila 176: 5.7 frente a 6.2). En las tres consultas la predicción estandarizada quedó más cerca del valor real, coherente con el menor error de la 4.1, aunque tres consultas no bastan para generalizar.
- **En escala cruda, el modelo ignora variables que importan.** Para la fila 67, los vecinos crudos tienen un contenido de alcohol que difiere en más de una desviación estándar del de la consulta: el modelo compara vinos con graduaciones muy distintas solo porque tienen sulfitos parecidos.