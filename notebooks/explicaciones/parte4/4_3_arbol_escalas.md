# 4.3 El árbol en escala cruda y estandarizada — D06.T21

**Responsable:** I1 · **Código:** `src/parte4/arbol_escalas.py` · **Datos:** Wine (tinto) · **Tablas:** `resultados/tablas/4_3_*.csv`

## Por qué, en teoría, la escala no le importa

Un árbol de regresión parte el espacio con reglas de una sola variable, $x_j \le t$. Estandarizar transforma cada variable con $x'_j = (x_j - \bar x_j)/s_j$, una función afín **creciente** ($s_j > 0$), que preserva el orden de los valores:

$$x_j \le t \iff x'_j \le t' = \frac{t - \bar x_j}{s_j}.$$

Cualquier partición posible en una escala es posible en la otra, con la misma reducción de error (que solo depende de qué filas quedan a cada lado, no de los valores de $x$). El árbol debería escoger los mismos cortes, en las mismas variables, con los umbrales transformados. A diferencia de k-NN, el árbol no usa distancias entre filas: compara cada variable consigo misma.

## ¿Da exactamente lo mismo?

`DecisionTreeRegressor(random_state=0)` ajustado sobre crudas y sobre estandarizadas, con varias profundidades:

| `max_depth` | Hojas | Misma estructura | Misma partición de train | Filas de prueba distintas (de 320) | Máx. dif. en predicción | RMSE crudo | RMSE estand. |
|---|---|---|---|---|---|---|---|
| 3 | 8 | Sí | Sí | 0 | 0 | 0.6514 | 0.6514 |
| 6 | 50 | Sí | Sí | 1 | 1.125 | 0.6472 | 0.6442 |
| 10 | 213 | Sí | Sí | 3 | 1.929 | 0.8179 | 0.8215 |
| None | 340 | Sí | Sí | 6 | 1.000 | 0.7886 | 0.7886 |

- **El árbol es el mismo:** a toda profundidad, mismos nodos, mismas variables en cada corte y exactamente la misma partición de las 1 279 filas de entrenamiento. Los umbrales coinciden al convertirlos de escala, salvo diferencias de redondeo (hasta $2\times10^{-5}$).
- **Pero algunas predicciones de prueba cambian:** entre 0 y 6 filas, según la profundidad, y el cambio puede ser de casi 2 puntos de nota.

## Por qué cambian: filas sobre el umbral

scikit-learn pone cada umbral en el **punto medio** de los dos valores de entrenamiento consecutivos entre los que corta, y trabaja internamente en `float32`.

**Ejemplo (`max_depth = 6`, fila de prueba 42, nodo 87).** El nodo corta *volatile acidity* entre los valores de entrenamiento 0.38 y 0.42. El punto medio exacto es 0.40, y la fila 42 tiene acidez volátil **exactamente 0.40**: cae sobre el umbral.

- Con aritmética exacta, $0.40 \le 0.40$ y la fila iría a la izquierda.
- En escala cruda, el punto medio calculado en `float32` queda en $0.3999999911$, apenas por debajo de 0.4: la fila va a la **derecha** (predicción 7.125).
- En escala estandarizada, el redondeo cae del otro lado: la fila va a la **izquierda** (predicción 6.0).

**Las 10 diferencias encontradas (en todas las profundidades) tienen la misma causa:** la fila de prueba vale exactamente el punto medio de los dos valores de entrenamiento del corte (0.40 entre 0.38 y 0.42; 0.69 entre 0.68 y 0.70; 10.1 entre 10.0 y 10.2; 10 entre 7 y 13…). El redondeo no favorece siempre la misma dirección: en unas filas la versión cruda va a la derecha y en otras a la izquierda.

**¿Por qué pasa tanto en Wine?** Las variables vienen con dos o tres decimales, así que el punto medio de dos valores observados suele ser otro valor que también aparece en los datos. Con datos continuos sin redondear, los empates exactos serían rarísimos. Mientras más profundo el árbol, más cortes hay y más oportunidades de que una fila caiga justo en uno.

## Conclusión

La escala no cambia el árbol: cambia solo el resultado de empates exactos que, en teoría, son de medida cero, y que en la práctica decide el redondeo en punto flotante. Es lo opuesto de k-NN (4.1, 4.2), donde la escala redefine qué filas son vecinas. En el lenguaje del proyecto: el kernel de un árbol depende solo del **orden** de los valores dentro de cada variable, no de sus unidades.