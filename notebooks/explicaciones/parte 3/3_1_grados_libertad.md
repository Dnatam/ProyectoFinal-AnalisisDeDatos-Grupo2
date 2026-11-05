# Parte 3.1 — Grados de libertad efectivos y matrices suavizadoras

## Objetivo

Comparar la flexibilidad de cinco familias de regresión mediante los grados de libertad efectivos, definidos como la traza de la matriz suavizadora $S$:

$$
\hat{\mathbf y}=S\mathbf y,\qquad df_{\mathrm{ef}}=\operatorname{tr}(S)=\sum_{i=1}^{n}S_{ii}.
$$

Cada $S_{ii}$ representa el peso que la observación $i$ asigna a su propia respuesta al predecirse.

## Datos y preparación

Se utilizó Wine Quality (red), con 1599 observaciones y 11 predictores. La partición fue de 1279 observaciones para entrenamiento y 320 para prueba (`random_state=0`). Los predictores se estandarizaron ajustando el escalador únicamente con entrenamiento. Las verificaciones de $S$ se hicieron sobre las 1279 filas de entrenamiento.

## Fórmulas

**OLS.** Con intercepto y matriz de diseño $X_d=[\mathbf 1,X]$, $S=X_dX_d^+$, donde $X_d^+$ es la pseudoinversa. Si las $p$ columnas predictoras son linealmente independientes después de centrar, $\operatorname{tr}(S)=p+1$. En general, $\operatorname{tr}(S)=\operatorname{rango}(X_d)$. En Wine: $p=11$ y $df=12$.

**Ridge.** Con intercepto no penalizado y valores singulares $d_j$ de $X_c=X-\bar X$:

$$
\operatorname{tr}(S)=1+\sum_j\frac{d_j^2}{d_j^2+\lambda}.
$$

**k-NN.** Para vecinos uniformes e inclusión explícita de la propia observación, $S_{ii}=1/k$, de modo que

$$
\operatorname{tr}(S)=\frac nk.
$$

**Filas repetidas.** La fórmula $\operatorname{tr}(S)=n/k$ supone que cada fila de entrenamiento es su propio vecino más cercano, así que $S_{ii}=1/k$. En Wine eso no siempre pasa: el entrenamiento tiene 303 filas con al menos un gemelo exacto (157 copias de más), y una fila está a distancia 0 tanto de sí misma como de su gemelo. Construyendo $S$ con los índices que devuelve `KNeighborsRegressor.kneighbors()` (los mismos que usa `.predict()`), con $k=1$ las 157 copias eligen a su gemelo en vez de a sí mismas, su $S_{ii}$ vale 0 y la traza es **1122 en lugar de 1279**. Con $k=2$ da 634 contra 639.5, con $k=3$ da 425.67 contra 426.33, y desde $k=5$ coincide exactamente, porque ya caben la fila y todos sus gemelos entre los vecinos.

Las predicciones no cambian, porque en Wine los gemelos siempre tienen la misma nota. Eso muestra algo más fino: **con filas repetidas $S$ no es única**. Dos matrices distintas producen el mismo $\hat y$, y la traza depende de cómo se desempate. La verificación de `verificar_matrices.py` fuerza la autoinclusión y por eso coincide por construcción; la verificación real está en `src/parte3/knn_duplicados.py` y `resultados/tablas/parte3_knn_duplicados.csv`.

**Árbol de regresión.** Condicionando en una partición ya construida, cada hoja con $m$ observaciones promedia sus respuestas con pesos $1/m$. La contribución de cada hoja a la traza es $m(1/m)=1$:

$$
\operatorname{tr}(S)=\text{número de hojas}.
$$

**Random Forest.** Para el experimento de verificación se emplearon 200 árboles, `max_features=0.33` y `bootstrap=False`. La matriz suavizadora condicional es el promedio de las matrices de los árboles:

$$
S_{RF}=\frac1B\sum_{b=1}^{B}S_b,\qquad
\operatorname{tr}(S_{RF})=\sum_i(S_{RF})_{ii}.
$$

En esta configuración sin bootstrap, la traza equivale al promedio del número de hojas de los árboles. **No se afirma que esta traza capture toda la flexibilidad estadística de los árboles adaptativos**, ya que sus particiones se aprenden utilizando $y$.

## Comprobación con una matriz pequeña

Para $X=(0,1,10,11)^\top$ y $k=2$, con autoinclusión:

$$
S=\begin{pmatrix}
0.5&0.5&0&0\\
0.5&0.5&0&0\\
0&0&0.5&0.5\\
0&0&0.5&0.5
\end{pmatrix}.
$$

Por lo tanto, $\operatorname{tr}(S)=2=n/k=4/2$. Se comprobó también que todos los elementos de la matriz calculada coinciden con los esperados.

## Resultados de Wine

| Modelo y configuración | $\operatorname{tr}(S)$ |
|---|---:|
| OLS | 12.0000 |
| Ridge, $\lambda=1$ | 11.9738 |
| Ridge, $\lambda=100$ | 10.3286 |
| Ridge, $\lambda=1000$ | 6.1393 |
| k-NN, $k=5$ | 255.8000 |
| k-NN, $k=30$ | 42.6333 |
| Árbol, profundidad 6 | 50.0000 |
| Árbol, profundidad 10 | 213.0000 |
| Random Forest, mínimo 5 por hoja | 191.6200 |
| Random Forest, mínimo 10 por hoja | 98.9450 |

Se realizaron **20 verificaciones** de $\operatorname{tr}(S)$ contra las fórmulas implementadas, todas satisfactorias dentro de la tolerancia numérica. El código está en `src/parte3/grados_libertad.py` y `src/parte3/verificar_matrices.py`; los resultados completos se encuentran en `resultados/tablas/parte3_grados_libertad.csv`, `resultados/tablas/parte3_verificacion_matrices.csv` y `resultados/graficas/parte3_grados_libertad.png`.

**Interpretación.** En Ridge, una penalización mayor reduce los grados de libertad. En k-NN, aumentar $k$ reduce la influencia de cada respuesta individual. En árboles, particiones con más hojas incrementan la traza condicional. En el bosque, hojas más grandes reducen la traza condicional promedio.