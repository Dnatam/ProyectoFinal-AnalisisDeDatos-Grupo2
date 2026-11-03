# 1.3 Pesos de k-NN y del árbol — D03.T09

**Responsable:** I1 · **Código:** `src/parte1/pesos_knn_arbol.py` · **Datos:** California (estandarizada)

## k-NN

`KNeighborsRegressor(n_neighbors=30)` (con `weights='uniform'`, el valor por defecto) predice el promedio de las $y$ de los $k$ vecinos más cercanos. Si $N_k(x)$ es el conjunto de esos vecinos:

$$w_i(x) = \frac{\mathbf 1\{x_i \in N_k(x)\}}{k}, \qquad \sum_i w_i(x) = k\cdot\frac1k = 1.$$

Exactamente $k$ filas votan, todas con peso $1/k$. Los índices de los vecinos se obtienen con `.kneighbors()`.

## Árbol

`DecisionTreeRegressor(max_depth=6, random_state=0)` asigna a cada hoja el promedio de las $y$ de entrenamiento que cayeron en ella. Si $L(x)$ es la hoja de la consulta:

$$w_i(x) = \frac{\mathbf 1\{x_i \in L(x)\}}{|L(x)|}, \qquad \sum_i w_i(x) = |L(x)|\cdot\frac{1}{|L(x)|} = 1.$$

Votan solo las filas que comparten hoja con la consulta. La hoja de cada fila se obtiene con `.apply()`.

## Comparación con OLS y ridge

- **Pesos no negativos.** En k-NN y en el árbol todos los pesos son $\ge 0$ y suman 1: la predicción es un promedio de casos reales y no puede salirse del rango de las $y$ de entrenamiento. En OLS y ridge votan todas las filas y hay pesos negativos.
- **Quién define el parecido.** En k-NN el vecindario lo define una distancia fija (euclidiana sobre las variables estandarizadas). En el árbol lo aprende el modelo usando las $y$ para escoger los cortes. Esto importa en la Parte 3: la condición "$S$ no depende de $y$" se cumple para k-NN pero no para el árbol.
- **Evidencia variable.** En k-NN siempre votan $k$ filas; en el árbol el número de votantes es el tamaño de la hoja, que puede variar muchísimo de una consulta a otra.

## Validación con vecinos y hojas

Además de comprobar que $Wy$ coincide con `.predict()`:

- **k-NN:** las posiciones con peso no nulo son exactamente los índices que devuelve `.kneighbors()`.
- **Árbol:** el número de pesos no nulos coincide con `tree_.n_node_samples` de la hoja de cada consulta, el tamaño que scikit-learn guarda internamente.

## Comprobación

Con California estandarizada ($n = 16\,512$), en las primeras 20 filas de prueba:

**k-NN ($k = 30$):**
- $\max |Wy - \texttt{predict}| = 8.9\times10^{-16}$.
- Suma de los pesos: exactamente 1 en todas las consultas.
- Cada consulta tiene 30 pesos no nulos, todos iguales a $1/30 \approx 0.0333$.
- Las posiciones con peso no nulo coinciden con los índices de `kneighbors`.

**Árbol (profundidad 6):**
- $\max |Wy - \texttt{predict}| = 4.0\times10^{-15}$.
- Suma de los pesos: $0.9999999999999999$ a $1.0$.
- El árbol tiene 64 hojas, el máximo posible con profundidad 6 ($2^6$): con 16 512 filas ninguna rama se queda sin datos antes del fondo.
- El número de pesos no nulos coincide con `tree_.n_node_samples` en todas las consultas.
- Tamaños de hoja de las 20 consultas: entre 208 y 1 132 filas. Cuatro de las 20 caen en la misma hoja de 1 132 filas, así que reciben exactamente la misma predicción.

**Contraste.** k-NN siempre se apoya en 30 filas; el árbol se apoya en cientos, pero en un número que varía más de cinco veces de una consulta a otra. La cantidad de evidencia detrás de una predicción depende del modelo, y en el árbol también de dónde cae la consulta (Parte 2).