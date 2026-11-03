# 1.2 Pesos de ridge — D02.T06

**Responsable:** I1 · **Código:** `src/parte1/pesos_ridge.py` · **Datos:** California (estandarizada), $\lambda = 100$

## Intercepto libre (lo que hace scikit-learn)

`Ridge` no penaliza el intercepto:

$$\min_{b_0,\,b}\; \|y - b_0\mathbf 1 - Xb\|^2 + \lambda\|b\|^2 .$$

**Intercepto óptimo.** Derivando respecto a $b_0$: $-2\,\mathbf 1^\top(y - b_0\mathbf 1 - Xb) = 0$, de donde $b_0 = \bar y - \bar x^\top b$.

**Centrado.** Sustituyendo, $y - b_0\mathbf 1 - Xb = y_c - X_c b$, con $y_c = y - \bar y\mathbf 1$ y $X_c = X - \mathbf 1\bar x^\top$. Queda un ridge sin intercepto sobre datos centrados:

$$\hat b = (X_c^\top X_c + \lambda I)^{-1}X_c^\top y_c = (X_c^\top X_c + \lambda I)^{-1}X_c^\top y,$$

donde la última igualdad usa $X_c^\top\mathbf 1 = 0$.

**Pesos.**

$$\hat y(x) = \bar y + (x - \bar x)^\top\hat b
\quad\Longrightarrow\quad
w(x)^\top = \tfrac1n\mathbf 1^\top + (x - \bar x)^\top(X_c^\top X_c + \lambda I)^{-1}X_c^\top .$$

**Suma.** $w(x)^\top\mathbf 1 = \tfrac1n\,n + (x-\bar x)^\top(X_c^\top X_c + \lambda I)^{-1}X_c^\top\mathbf 1 = 1 + 0 = 1$.

Cada predicción es la media de $y$ más una corrección según qué tan lejos está $x$ del centro de los datos.

## Intercepto penalizado (el contraste)

Agregando la columna de unos y penalizándola junto con lo demás:

$$w(x)^\top = \tilde x^\top(\tilde X^\top\tilde X + \lambda I)^{-1}\tilde X^\top .$$

**No cuadra con `.predict()`**: es otro modelo, en el que el intercepto también se encoge hacia cero.

**Cuánto suman los pesos.** Como $X$ está estandarizada con train, sus columnas tienen media cero: $X^\top\mathbf 1 = 0$. Entonces

$$\tilde X^\top\tilde X + \lambda I = \begin{pmatrix} n+\lambda & 0 \\ 0 & X^\top X + \lambda I\end{pmatrix},
\qquad \tilde X^\top\mathbf 1 = \begin{pmatrix} n \\ 0\end{pmatrix},$$

$$\sum_i w_i(x) = \tilde x^\top\begin{pmatrix} n/(n+\lambda) \\ 0\end{pmatrix} = \frac{n}{n+\lambda}\quad\text{para cualquier } x.$$

**Qué cambia en la predicción.** Por la estructura de bloques, las pendientes salen idénticas a las del intercepto libre (con $\bar x = 0$, $X_c = X$). Solo cambia el intercepto: pasa de $\bar y$ a $\tfrac{n}{n+\lambda}\bar y$. Toda predicción se corre la misma cantidad hacia cero:

$$\hat y_{\text{penalizado}}(x) - \hat y_{\text{libre}}(x) = -\bar y\,\frac{\lambda}{n+\lambda}.$$

Penalizar el intercepto no reduce varianza en nada útil: solo introduce un sesgo que depende de dónde está el cero de $y$. Por eso scikit-learn (y ESL) lo dejan libre. Con Wine el efecto sería grande: $n/(n+\lambda) = 1279/1379 \approx 0.927$ y el corrimiento es de unas $-0.41$ unidades de nota.

## Implementación

- `pesos_ridge(X_train, X_consulta, lam)`: intercepto libre; reproduce `Ridge(alpha=lam).predict`.
- `pesos_ridge_intercepto_penalizado(X_train, X_consulta, lam)`: el contraste. Como verificación, coincide con `Ridge(alpha=lam, fit_intercept=False)` ajustado sobre $[\mathbf 1, X]$.

Ambas resuelven el sistema lineal con `np.linalg.solve` en lugar de invertir la matriz.

## Comprobación

Con California estandarizada ($n = 16\,512$), $\lambda = 100$, en las primeras 20 filas de prueba:

**Intercepto libre:**
- $\max |Wy - \texttt{Ridge(alpha=100).predict}| = 2.2\times10^{-15}$: los pesos reproducen la predicción salvo error de redondeo.
- Suma de los pesos: mínimo $0.9999999999999998$, máximo $1.0000000000000002$.

**Intercepto penalizado:**
- $\max |Wy - \texttt{predict}| = 0.012476$ y la diferencia media es $-0.012476$: el máximo y la media coinciden en valor absoluto, así que todas las predicciones se corren exactamente la misma cantidad.
- Suma de los pesos: $0.9939802552371777$ a $0.9939802552371783$, igual a $n/(n+\lambda) = 0.9939802552371779$ en todas las filas.
- Corrimiento esperado $-\bar y\,\lambda/(n+\lambda) = -0.012476$, que coincide con la diferencia observada.
- Como verificación, los pesos penalizados coinciden con `Ridge(alpha=100, fit_intercept=False)` ajustado sobre $[\mathbf 1, X]$.

En California el efecto es pequeño (unos \$1 250 en el valor mediano) porque $n$ es grande frente a $\lambda$; con Wine ($n = 1\,279$) el corrimiento sería de $-0.41$ unidades de nota.