# 1.1 Pesos de OLS — D02.T05

**Responsable:** I1 · **Código:** `src/parte1/pesos_ols.py` · **Datos:** California (estandarizada)

## Planteamiento

Sea $\tilde X = [\mathbf 1 \;\; X] \in \mathbb R^{n\times(p+1)}$ la matriz de diseño con una columna de unos y $\tilde x = (1, x)$ una consulta. El estimador de OLS es

$$\hat\beta = (\tilde X^\top \tilde X)^{-1}\tilde X^\top y.$$

## Los pesos

$$\hat y(x) = \tilde x^\top \hat\beta = \tilde x^\top (\tilde X^\top \tilde X)^{-1}\tilde X^\top y = \sum_{i=1}^n w_i(x)\, y_i,
\qquad
w(x)^\top = \tilde x^\top (\tilde X^\top \tilde X)^{-1}\tilde X^\top .$$

Los pesos dependen solo de las $x$ (entrenamiento y consulta), **no de $y$**.

## Los pesos suman 1

$\sum_i w_i(x) = w(x)^\top \mathbf 1$. Como $\mathbf 1$ es la primera columna de $\tilde X$, $\mathbf 1 = \tilde X e_1$ con $e_1 = (1, 0, \dots, 0)^\top$. Entonces

$$(\tilde X^\top \tilde X)^{-1}\tilde X^\top \mathbf 1 = (\tilde X^\top \tilde X)^{-1}(\tilde X^\top \tilde X)\,e_1 = e_1
\quad\Longrightarrow\quad
w(x)^\top \mathbf 1 = \tilde x^\top e_1 = 1,$$

porque la primera entrada de $\tilde x$ es 1. Vale para **cualquier** $x$.

**Con la matriz sombrero.** En los puntos de entrenamiento los pesos forman $H = \tilde X(\tilde X^\top \tilde X)^{-1}\tilde X^\top$, la proyección ortogonal sobre el espacio columna de $\tilde X$. Como $\mathbf 1$ está en ese espacio, $H\mathbf 1 = \mathbf 1$: cada fila de $H$ suma 1.

**Intuición.** Si $y_i = c$ para todo $i$, OLS con intercepto ajusta $\hat y = c$ exactamente; para que $\sum_i w_i c = c$, los pesos deben sumar 1. Nada obliga a que sean positivos (ver 2.2).

## Implementación

`pesos_ols(X_train, X_consulta)` devuelve la matriz $W$ de forma $(m, n)$, una fila de pesos por consulta, de modo que la predicción es `W @ y_train`. En lugar de invertir $\tilde X^\top \tilde X$, se resuelve el sistema $(\tilde X^\top \tilde X)A = \tilde X^\top$ con `np.linalg.solve`, que es equivalente y numéricamente más estable.

## Comprobación

Con California estandarizada (16 512 filas de entrenamiento), en las primeras 20 filas de prueba:

- $W$ tiene forma $(20, 16\,512)$: cada predicción es un voto de las 16 512 filas de entrenamiento.
- $\max |Wy - \texttt{LinearRegression().predict}| = 3.8\times10^{-15}$: los pesos reproducen la predicción salvo error de redondeo.
- Suma de los pesos: mínimo $0.9999999999999998$, máximo $1.0$.
- En un $x$ muy lejano de los datos (todas las variables en 50 desviaciones estándar), la suma es $0.9999999999999989$: suman 1 para cualquier $x$, no solo cerca de los datos.
- $H\mathbf 1 = \mathbf 1$ se cumple en una submuestra de 2 000 filas.