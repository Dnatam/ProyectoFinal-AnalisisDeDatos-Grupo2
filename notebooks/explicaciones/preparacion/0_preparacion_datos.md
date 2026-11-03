# Preparación de datos (0.1 y 0.4) — D01.T02

**Responsable:** I1 · **Código:** `src/comunes/datos.py` · **Particiones:** `data/wine/particiones/`, `data/california/particiones/`

## Qué se hace

1. **Partición 80/20.** `train_test_split(X, y, test_size=0.2, random_state=0)` (constantes en `src/comunes/config.py`), igual para California y Wine. Con la semilla fija, todos los integrantes obtienen exactamente las mismas filas. Los índices originales se guardan en `indices_train` e `indices_test` y en los CSV de particiones, para poder verificarlo.
2. **Estandarización solo con train.** El `StandardScaler` se ajusta con `X_train` (`fit_transform`) y el test se transforma con esos mismos parámetros (`transform`). Si el scaler viera el test, el modelo recibiría información de datos que supuestamente no conoce y el error de prueba saldría optimista.
3. **Desviaciones originales (0.4).** `desviaciones_originales` es la desviación estándar de cada variable en train antes de estandarizar, con `ddof=0` (poblacional), que es la misma que usa el scaler (coincide con `scaler.scale_`). `df.std()` de pandas usa `ddof=1` y daría valores apenas distintos.

El diccionario que devuelven `cargar_wine()` y `cargar_california()` trae las variables crudas (`X_train`, `X_test`, para la Parte 4) y las estandarizadas (`X_train_std`, `X_test_std`, para todo lo demás).

## Resultados — Wine Quality (tinto)

1 599 vinos → **1 279 de entrenamiento y 320 de prueba**, 11 variables.

| Variable | Desv. estándar (train) |
|---|---|
| fixed acidity | 1.7343 |
| volatile acidity | 0.1825 |
| citric acid | 0.1955 |
| residual sugar | 1.4520 |
| chlorides | 0.0478 |
| free sulfur dioxide | 10.5168 |
| total sulfur dioxide | 33.5711 |
| density | 0.0019 |
| pH | 0.1539 |
| sulphates | 0.1748 |
| alcohol | 1.0609 |

Las escalas difieren en varios órdenes de magnitud: *total sulfur dioxide* y *free sulfur dioxide* tienen desviaciones miles de veces mayores que *density*. En escala cruda, una distancia euclidiana (k-NN, k-means) estará dominada casi por completo por el dióxido de azufre. Esto se retoma en la Parte 4.

**Observación:** Wine tiene 240 filas repetidas. Es relevante para la fórmula tr(S) = n/k de k-NN (3.1) y para los pesos de k-NN (1.3), porque una fila de entrenamiento repetida tiene más de un vecino a distancia cero.

## Resultados — California Housing

20 640 distritos → **16 512 de entrenamiento y 4 128 de prueba**, 8 variables.

| Variable | Desv. estándar (train) |
|---|---|
| MedInc | 1.9067 |
| HouseAge | 12.6006 |
| AveRooms | 2.5233 |
| AveBedrms | 0.4895 |
| Population | 1143.0281 |
| AveOccup | 6.4411 |
| Latitude | 2.1365 |
| Longitude | 2.0052 |

Aquí también hay escalas muy distintas: *Population* tiene una desviación más de 2 000 veces mayor que *AveBedrms*, así que en escala cruda dominaría cualquier distancia. Por eso todo, salvo la Parte 4, se trabaja con las variables estandarizadas.