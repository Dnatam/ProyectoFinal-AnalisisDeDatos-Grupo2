# 1.5 Verificación de los pesos — D04.T13

**Responsable:** I1 · **Código:** `src/parte1/verificacion.py` · **Datos:** California (estandarizada) · **Tablas:** `resultados/tablas/1_5_*.csv`

## La función `verificar`

`verificar(w, y_train, pred)` comprueba con `assert` que

$$\Big|\sum_i w_i\, y_i - \text{pred}\Big| < 10^{-8}.$$

Si los pesos no reproducen la predicción del modelo, el programa se detiene con un error que indica la diferencia. La tolerancia $10^{-8}$ deja margen de sobra para el error de redondeo (nuestras diferencias son del orden de $10^{-15}$) y aun así detecta cualquier error real: la fórmula ingenua del bosque con bootstrap (1.4) falla por $0.04$.

## Los cinco modelos

| Modelo | Configuración | Pesos (sección) |
|---|---|---|
| OLS | `LinearRegression()` | 1.1 |
| Ridge | `Ridge(alpha=100)` | 1.2 |
| k-NN | `KNeighborsRegressor(n_neighbors=30)` | 1.3 |
| Árbol | `DecisionTreeRegressor(max_depth=6, random_state=0)` | 1.3 |
| Random forest | `RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, bootstrap=False, random_state=0)` | 1.4 |

## Verificación en las primeras 20 filas de prueba

Las 100 verificaciones (20 filas × 5 modelos) pasaron. Diferencia máxima por modelo:

| Modelo | Filas verificadas | $\max \lvert w^\top y - \text{pred}\rvert$ |
|---|---|---|
| OLS | 20 | $3.6\times10^{-15}$ |
| Ridge | 20 | $1.8\times10^{-15}$ |
| k-NN | 20 | $6.7\times10^{-16}$ |
| Árbol | 20 | $4.0\times10^{-15}$ |
| Random forest | 20 | $3.1\times10^{-15}$ |

Todas están siete órdenes de magnitud por debajo de la tolerancia: los pesos reproducen `.predict()` salvo error de redondeo.

## Tabla de la fila de prueba 7

La "fila 7" es `X_test[7]`, contando desde 0. Su valor real es $y = 3.369$ (unos \$337 000) y hay $n = 16\,512$ filas de entrenamiento.

| Modelo | Predicción | $\sum_i w_i$ | Pesos $\neq 0$ | Pesos $< 0$ |
|---|---|---|---|---|
| OLS | 3.0566 | 1.0 | 16 512 | 4 144 |
| Ridge | 3.0561 | 1.0 | 16 512 | 4 130 |
| k-NN | 3.2082 | 1.0 | 30 | 0 |
| Árbol | 3.6990 | 1.0 | 328 | 0 |
| Random forest | 3.7180 | 1.0 | 388 | 0 |

## Lectura de la tabla

- **Todos suman 1.** Los cinco modelos predicen con un promedio ponderado de las $y$ de entrenamiento; ninguno agrega ni quita "masa".
- **Dos familias de pesos.** OLS y ridge reparten peso entre las 16 512 filas, y cerca de una de cada cuatro (≈25 %) vota con signo negativo. k-NN, el árbol y el bosque solo dan peso a unas decenas o cientos de filas, todos positivos: su predicción es un promedio genuino de casos reales.
- **Ridge casi no se aleja de OLS.** Con $\lambda = 100$ frente a $n = 16\,512$, la penalización es pequeña: la predicción cambia en la cuarta cifra decimal y solo 14 pesos cambian de signo.
- **Cuánta evidencia.** k-NN se apoya siempre en exactamente 30 filas; el árbol, en las 328 de su hoja (el mismo tamaño que reportó la 1.3 para esta consulta); el bosque reparte el peso entre 388 filas, porque junta las hojas de 200 árboles que cortan distinto.
- **Predicciones distintas con los mismos datos.** Los lineales subestiman ($\approx 3.06$ frente a $3.37$), los de árboles sobreestiman ($\approx 3.70$) y k-NN queda en medio. Son cinco maneras distintas de decidir qué filas se parecen a la consulta, y eso basta para mover la predicción unos \$65 000 de un extremo a otro.