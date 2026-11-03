# 1.4 Pesos del random forest — D03.T10

**Responsable:** I1 · **Código:** `src/parte1/pesos_bosque.py` · **Datos:** California (estandarizada)

`RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, random_state=0)`, con `bootstrap=False` y `bootstrap=True`.

## Sin bootstrap

El bosque predice el promedio de sus $T$ árboles, y cada árbol es un voto ponderado (1.3). Intercambiando el orden de las sumas:

$$\hat y(x) = \frac1T\sum_{t=1}^T \sum_i \frac{\mathbf 1\{x_i \in L_t(x)\}}{|L_t(x)|}\,y_i
= \sum_i \underbrace{\frac1T\sum_{t=1}^T \frac{\mathbf 1\{x_i \in L_t(x)\}}{|L_t(x)|}}_{w_i(x)}\,y_i ,$$

donde $L_t(x)$ es la hoja donde cae $x$ en el árbol $t$. Los pesos son no negativos y suman 1, porque son el promedio de $T$ vectores de pesos que suman 1 cada uno.

Sin bootstrap, cada árbol se entrena con todas las filas; los árboles difieren solo por `max_features=0.33`, que sortea las variables candidatas en cada corte. Por eso $|L_t(x)|$ es exactamente el número de filas de entrenamiento en la hoja y la fórmula reproduce `.predict()`.

## Con bootstrap

Con `bootstrap=True`, la misma fórmula deja de coincidir. Cada árbol calcula el promedio de sus hojas **solo con su muestra bootstrap**: $n$ filas sorteadas con reemplazo. scikit-learn lo implementa entrenando con todas las filas, pero dándole a cada una como peso $c_{ti}$, el número de veces que salió en el sorteo del árbol $t$. Eso tiene dos efectos que la fórmula ingenua ignora:

1. **Filas fuera de la muestra** ($c_{ti} = 0$): aproximadamente $(1 - 1/n)^n \approx 1/e \approx 36.8\%$ de las filas por árbol. Caen en alguna hoja al hacer `.apply()`, pero su $y$ no entró al promedio de esa hoja.
2. **Filas repetidas** ($c_{ti} \ge 2$): pesan más en el promedio de su hoja.

La fórmula correcta pesa por los conteos:

$$w_i(x) = \frac1T\sum_{t=1}^T \frac{c_{ti}\,\mathbf 1\{x_i \in L_t(x)\}}{\sum_j c_{tj}\,\mathbf 1\{x_j \in L_t(x)\}} .$$

Sin bootstrap, $c_{ti} = 1$ para todo $t, i$ y se recupera la fórmula anterior. Los conteos se reconstruyen repitiendo el sorteo de scikit-learn con la semilla guardada en cada árbol.

Los pesos ingenuos también suman 1: siguen siendo un promedio, pero de filas que el árbol no usó y sin las repeticiones que sí usó. Las filas fuera de la muestra de cada árbol son las que hacen posible el error *out-of-bag* (3.6).

## Comprobación

Con California estandarizada ($n = 16\,512$), en las primeras 20 filas de prueba:

**Sin bootstrap:**
- $\max |Wy - \texttt{predict}| = 2.7\times10^{-15}$: la fórmula del enunciado reproduce la predicción.
- Suma de los pesos: $0.9999999999999998$ a $1.0$.
- Cada predicción se apoya en entre 247 y 545 filas con peso no nulo: muchas más que las 5 del mínimo por hoja, porque cada árbol aporta una hoja distinta y el bosque junta las de los 200 árboles.

**Con bootstrap, fórmula del enunciado:**
- $\max |Wy - \texttt{predict}| = 0.044$ (unos \$4 400 en el valor mediano) y diferencia media de $-0.0018$: el error cambia de signo y tamaño según la consulta, no es un corrimiento constante.
- Los pesos siguen sumando 1 ($0.9999999999999998$ a $1.0$): es un promedio válido, pero de las filas equivocadas.

**Con bootstrap, pesando por los conteos del sorteo:**
- $\max |Wy - \texttt{predict}| = 4.9\times10^{-15}$: con los conteos $c_{ti}$ la fórmula vuelve a ser exacta.
- Suma de los pesos: $0.9999999999999999$ a $1.0$.
- Fracción media de filas fuera de la muestra de cada árbol: $0.3676$, frente a la teórica $(1-1/n)^n \approx 1/e = 0.3679$.
- La fila más repetida salió 9 veces en una misma muestra.

**Nota de versión.** Los conteos se reconstruyen con funciones internas de scikit-learn (`_generate_sample_indices`, `_get_n_samples_bootstrap`), cuya firma cambió en la versión 1.9 (piden `sample_weight`). El código detecta la firma y funciona en ambas.