# 2.1 Selección de consultas y ficha de evidencia

## Hiperparámetros escogidos

Se ajustaron con validación cruzada de 5 pliegues sobre el entrenamiento (1 279 vinos),
minimizando el MSE (`cv_modelos.py`, tabla `parte2_hiperparametros.csv`).

| Modelo | Hiperparámetros | MSE (CV) |
|---|---|---|
| OLS | ninguno | 0.4407 |
| Ridge | λ = 100 | 0.4387 |
| k-NN | k = 29 | 0.4392 |
| Árbol | profundidad 6, `min_samples_leaf` = 10 | 0.4681 |
| Random forest | sin límite de profundidad, `min_samples_leaf` = 2, `max_features` = 0.33 | 0.3574 |

El bosque es claramente el mejor. Los otros cuatro quedan casi empatados.

## Cómo se escogieron las tres consultas

Con los pesos del bosque se calculó $N_{ef}$ para las 320 filas de prueba. Antes se verificó
que $\sum_i w_i y_i$ reproduce `.predict()` en todas (diferencia máxima de $8 \times 10^{-15}$).

| Consulta | Fila de prueba | $N_{ef}$ (bosque) | $c(x)$ (bosque) | Nota real |
|---|---|---|---|---|
| $N_{ef}$ mediano | 176 | 51.97 | 0.27 | 6 |
| $N_{ef}$ mínimo | 67 | 3.42 | 0.50 | 5 |
| Interesante: máxima concentración | 12 | 7.01 | 0.96 | 6 |

**Por qué la fila 12 es interesante:** es la de mayor $c(x)$ de todo el conjunto de prueba,
y además tiene un **gemelo exacto en entrenamiento** (la fila 479, con las mismas 11 variables
y la misma nota, 6). Sirve para ver qué hace cada modelo cuando la respuesta "ya está" en los datos.

Las variables que se muestran en los testigos son las 4 de mayor importancia en el bosque
(`feature_importances_`): **alcohol, sulphates, volatile acidity y density**, en unidades originales.

## Ficha de evidencia por modelo

| Consulta | Modelo | Predicción | $N_{ef}$ | $c(x)$ | Pesos ≠ 0 | Pesos < 0 |
|---|---|---|---|---|---|---|
| 176 (real 6) | OLS | 6.17 | 41.62 | 0.066 | 1279 | 582 |
| | Ridge | 6.14 | 50.21 | 0.067 | 1279 | 577 |
| | k-NN | 6.17 | 29.00 | 0.035 | 29 | 0 |
| | Árbol | 5.45 | 11.00 | 0.091 | 11 | 0 |
| | Bosque | 5.84 | 51.97 | 0.273 | 194 | 0 |
| 67 (real 5) | OLS | 6.01 | 41.65 | 0.024 | 1279 | 589 |
| | Ridge | 5.95 | 52.27 | 0.023 | 1279 | 597 |
| | k-NN | 5.79 | 29.00 | 0.035 | 29 | 0 |
| | Árbol | 5.53 | 17.00 | 0.059 | 17 | 0 |
| | Bosque | 5.00 | 3.42 | 0.497 | 101 | 0 |
| 12 (real 6) | OLS | 5.71 | 202.79 | 0.012 | 1279 | 361 |
| | Ridge | 5.70 | 281.21 | 0.015 | 1279 | 325 |
| | k-NN | 5.69 | 29.00 | 0.035 | 29 | 0 |
| | Árbol | 5.80 | 10.00 | 0.100 | 10 | 0 |
| | Bosque | 5.84 | 7.01 | 0.964 | 162 | 0 |

En todos los casos los pesos suman 1. Los 5 testigos de cada consulta y modelo están en
`parte2_testigos.csv`.

## Qué dice la tabla de cada modelo

* **k-NN:** $N_{ef} = 29 = k$ siempre, porque los 29 vecinos pesan lo mismo ($1/29$). Su
  evidencia no cambia de una consulta a otra, aunque los vecinos estén lejos.
* **Árbol:** $N_{ef}$ es el número de filas de la hoja (11, 17 y 10), porque todas pesan lo mismo dentro de la hoja.
* **OLS y ridge:** usan las 1 279 filas, y cerca de la mitad con peso negativo. Ridge tiene
  $N_{ef}$ más alto que OLS porque la penalización reparte los pesos de forma más pareja.
  En la fila 12, OLS llega a $N_{ef} = 203$: esa consulta está cerca del centro de los
  datos, donde un plano tiene mucha evidencia (ver 2.4, criterio 3).
* **Bosque:** es el único cuyo $N_{ef}$ cambia mucho de una consulta a otra (de 3.4 a 52).
  Esa adaptación es la que lo vuelve interesante: concentra el peso cuando encuentra vinos muy parecidos.

## Lectura de cada consulta

**Fila 176 ($N_{ef}$ mediano).** El bosque reparte el peso entre 194 vinos. Sus dos testigos
principales (filas 382 y 1222) son idénticos entre sí (alcohol 11.0, sulphates 0.58) y
tienen nota 5, lo que jala la predicción hacia abajo (5.84, real 6).

**Fila 67 ($N_{ef}$ mínimo).** Los dos testigos principales del bosque (filas 219 y 1165)
reciben 0.38 cada uno, es decir, el **76% del voto**. Esas dos filas son idénticas entre sí, y
la consulta es **casi un gemelo** de ellas: las 11 variables coinciden salvo `fixed acidity`
(15.6 contra 15.5). Por eso el bosque predice 5.00 y acierta exactamente. El $N_{ef}$ bajo
no viene de falta de datos, sino de que hay dos vinos prácticamente iguales a la consulta.

**Fila 12 (máxima concentración).** El gemelo exacto (fila 479, nota 6) recibe un peso de
0.37, 14 veces el del siguiente testigo. Ojo: $c(x) = 0.96$ **no** significa que el gemelo
tenga el 96% del voto. $c(x)$ es la parte de la suma de **cuadrados** de los pesos que se lleva
la fila dominante; en el voto real, el gemelo tiene el 37%. El resto se reparte entre 161
vinos con pesos pequeños, casi todos con nota 6. Los modelos menos flexibles apenas notan
al gemelo: k-NN le da 1/29 y OLS solo 0.005.

## Conclusión

Las dos consultas con menor $N_{ef}$ (67 y 12) no son predicciones "con poca evidencia" en el
sentido usual: son vinos que ya están, exactos o casi exactos, en el entrenamiento. Esto
anticipa lo que se ve en 2.3.