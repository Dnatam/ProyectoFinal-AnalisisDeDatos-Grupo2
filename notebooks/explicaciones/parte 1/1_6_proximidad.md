# 1.6 Proximidad del bosque — D04.T14

**Responsable:** I1 · **Código:** `src/parte1/proximidad.py` · **Datos:** California (estandarizada), fila de prueba 7 · **Tablas:** `resultados/tablas/1_6_*.csv`

## Definición

La proximidad entre una fila de entrenamiento $x_i$ y la consulta $x$ es la fracción de árboles en los que caen en la misma hoja:

$$\text{prox}_i(x) = \frac1T\sum_{t=1}^T \mathbf 1\{x_i \in L_t(x)\} \in [0, 1].$$

Comparada con el peso del bosque (1.4),

$$w_i(x) = \frac1T\sum_{t=1}^T \frac{\mathbf 1\{x_i \in L_t(x)\}}{|L_t(x)|},$$

la única diferencia es la división entre el tamaño de la hoja dentro de cada árbol. Por eso las dos tienen el mismo soporte: $\text{prox}_i(x) > 0 \iff w_i(x) > 0$.

## ¿Cuánto suma?

$$\sum_i \text{prox}_i(x) = \frac1T\sum_{t=1}^T \sum_i \mathbf 1\{x_i \in L_t(x)\} = \frac1T\sum_{t=1}^T |L_t(x)|,$$

el tamaño promedio de la hoja donde cae $x$. Con `min_samples_leaf=5` es al menos 5; nunca vale 1.

## ¿Por qué no es un peso y qué le falta?

Un peso debe cumplir $\hat y(x) = \sum_i w_i(x)\,y_i$, y para eso tiene que sumar 1 (si todas las $y_i$ valen $c$, la predicción debe ser $c$). La proximidad suma el tamaño medio de hoja, así que $\sum_i \text{prox}_i\,y_i$ no es una predicción, sino algo del orden de $\bar y$ multiplicado por ese tamaño.

**Lo que le falta es normalizar dentro de cada árbol:** dividir la indicadora entre $|L_t(x)|$ antes de promediar, que es exactamente la fórmula de $w_i(x)$.

**Normalizar al final no basta.** $\text{prox}_i / \sum_j \text{prox}_j$ suma 1, pero equivale a

$$\frac{\text{prox}_i(x)}{\sum_j \text{prox}_j(x)} = \sum_{t=1}^T \frac{|L_t(x)|}{\sum_s |L_s(x)|}\cdot\frac{\mathbf 1\{x_i \in L_t(x)\}}{|L_t(x)|},$$

un promedio de los árboles en el que cada uno pesa según el tamaño de su hoja. En el bosque real cada árbol pesa $1/T$. Solo coinciden si $x$ cae en hojas del mismo tamaño en todos los árboles.

## Resultados

Fila de prueba 7 de California, bosque sin bootstrap ($T = 200$, `min_samples_leaf=5`).

**La proximidad no suma 1:**
- $\sum_i \text{prox}_i(x) = 6.495$, igual al tamaño promedio de la hoja de $x$ en los 200 árboles ($6.495$). Las hojas miden entre 5 y 9 filas.
- La proximidad máxima es $0.23$: ninguna fila de entrenamiento comparte hoja con la consulta en más de 46 de los 200 árboles, y ninguna lo hace en todos.
- $\sum_i \text{prox}_i\, y_i = 24.18$, que no es una predicción (la del bosque es $3.718$): es del orden de la predicción multiplicada por el tamaño medio de hoja ($6.495 \times 3.72 \approx 24.2$).

**Comparación con los pesos:**
- Los pesos suman $1.0$.
- Mismo soporte: 388 filas con proximidad $> 0$ y exactamente las mismas 388 con peso $> 0$.

**Proximidad normalizada al final:**
- Suma 1, pero $\max_i \lvert \text{prox}_i/\textstyle\sum_j \text{prox}_j - w_i \rvert = 0.0026$, y la predicción resultante es $3.7236$ frente a $3.7180$ de `.predict()` (diferencia $0.0056$).
- La diferencia es pequeña porque las hojas de $x$ tienen tamaños parecidos (5 a 9) en todos los árboles; con tamaños más dispares, los árboles de hojas grandes dominarían y la diferencia crecería.

**Las 5 filas con mayor peso:**

| Fila train | $y$ | Proximidad | Peso | Prox. normalizada |
|---|---|---|---|---|
| 15189 | 3.418 | 0.230 | 0.0380 | 0.0354 |
| 14089 | 3.597 | 0.185 | 0.0290 | 0.0285 |
| 270 | 5.000 | 0.150 | 0.0247 | 0.0231 |
| 10304 | 4.955 | 0.145 | 0.0237 | 0.0223 |
| 13408 | 4.658 | 0.135 | 0.0214 | 0.0208 |

El orden es el mismo con las tres medidas, pero las magnitudes no: las filas que comparten hojas pequeñas reciben relativamente más peso que proximidad. La fila 270 tiene $y = 5.00001$, el valor tope con el que viene censurado California Housing (todo distrito de más de \$500 000 aparece como \$500 001): uno de los testigos principales de esta predicción es un valor truncado.