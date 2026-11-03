# 4.1 k-NN en tres escalas — D05.T17

**Responsable:** I1 · **Código:** `src/parte4/knn_escalas.py` · **Datos:** Wine (tinto) · **Tablas:** `resultados/tablas/4_1_*.csv`

`KNeighborsRegressor(n_neighbors=10)` sobre tres representaciones de las mismas 11 variables. El escalador y los PCA se ajustan solo con el entrenamiento.

## Error de prueba

| Escala | Dimensiones | RMSE | MAE |
|---|---|---|---|
| Crudas | 11 | 0.6988 | 0.5484 |
| Estandarizadas | 11 | **0.6659** | **0.5103** |
| PCA 90 % (sobre estandarizadas) | 7 | 0.6707 | 0.5184 |
| PCA 90 % (sobre crudas, contraste) | 1 | 0.7866 | 0.6119 |

Referencia: predecir siempre la media de entrenamiento da RMSE $= 0.9618$. El PCA principal usa 7 componentes, que explican el 90.7 % de la varianza.

## ¿Qué variables dominan la distancia en escala cruda?

Para dos vinos tomados al azar, $\mathbb E[(x_j - x'_j)^2] = 2\sigma_j^2$, así que la variable $j$ aporta en promedio la fracción $\sigma_j^2 / \sum_l \sigma_l^2$ de la distancia euclidiana al cuadrado. Con las desviaciones de la preparación (0.4):

| Variable | Desv. estándar | Participación en $d^2$ | Acumulada |
|---|---|---|---|
| total sulfur dioxide | 33.571 | 90.6 % | 90.6 % |
| free sulfur dioxide | 10.517 | 8.9 % | 99.5 % |
| fixed acidity | 1.734 | 0.24 % | 99.7 % |
| residual sugar | 1.452 | 0.17 % | 99.9 % |
| alcohol | 1.061 | 0.09 % | 99.99 % |
| las otras 6 | $\le 0.2$ | $< 0.01$ % | 100 % |

**El dióxido de azufre total y el libre explican el 99.5 % de la distancia.** En escala cruda, "dos vinos son parecidos" significa casi exclusivamente "tienen niveles de sulfitos parecidos"; el alcohol, la acidez volátil o los sulfatos prácticamente no participan en la elección de los vecinos. Aun así, el k-NN crudo mejora mucho sobre predecir la media ($0.70$ frente a $0.96$): un vecindario definido casi solo por los sulfitos sí contiene información sobre la nota.

## ¿Estandarizar siempre ayuda?

No necesariamente. Estandarizar no "corrige" la distancia: escoge otra, en la que cada variable pesa lo mismo. Es otra elección de kernel.

- **Aquí ayuda, pero poco:** el RMSE baja de $0.699$ a $0.666$ (≈5 %). Estandarizar le devuelve voz a variables que en escala cruda no contaban (alcohol, acidez volátil, sulfatos).
- **Puede empeorar:** si una variable irrelevante tiene unidades pequeñas, estandarizarla multiplica su ruido y la mete en la distancia con el mismo peso que las importantes. Y si la variable de unidades grandes fuera justo la más predictiva, la escala cruda sería una buena elección por accidente.
- Lo correcto es tratar la escala como un hiperparámetro y compararla con validación, no asumir que estandarizar es siempre mejor.

## PCA

- **Sobre estandarizadas** (7 componentes): casi igual que estandarizar ($0.671$ frente a $0.666$). Descartar el 9.3 % de varianza cuesta un poco porque PCA elige las direcciones por su varianza en $X$, sin mirar $y$: nada garantiza que las direcciones descartadas no lleven información sobre la nota.
- **Sobre crudas** (contraste): un solo componente explica el 94.8 % de la varianza, con cargas $0.977$ en *total sulfur dioxide* y $0.215$ en *free sulfur dioxide*. PCA hereda el problema de escala y resume los 11 datos en "cuántos sulfitos tiene el vino": es el peor de los cuatro (RMSE $0.787$). Por eso el PCA principal se ajusta sobre las variables estandarizadas.