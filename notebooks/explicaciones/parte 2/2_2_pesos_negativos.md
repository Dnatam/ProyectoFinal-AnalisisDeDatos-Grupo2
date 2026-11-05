# 2.2 Pesos negativos de OLS y peso contra distancia

## ¿Cuántos pesos son negativos?

| Consulta | Pesos negativos | % | Suma de negativos | Suma de positivos | Peso más negativo |
|---|---|---|---|---|---|
| 176 ($N_{ef}$ mediano) | 582 de 1 279 | 45.5% | −1.22 | 2.22 | −0.0114 |
| 67 ($N_{ef}$ mínimo) | 589 de 1 279 | 46.1% | −1.58 | 2.58 | −0.0145 |
| 12 (máx. concentración) | 361 de 1 279 | 28.2% | −0.46 | 1.46 | −0.0074 |

Los pesos suman 1 en los tres casos, pero OLS lo logra sumando mucho más de 1 en positivos
y restando el exceso con negativos. k-NN, el árbol y el bosque nunca tienen pesos negativos.

## ¿Por qué hay pesos negativos?

El peso de OLS es $w_i(x) = \tilde x^\top (\tilde X^\top \tilde X)^{-1} \tilde x_i$: es
**lineal en $\tilde x_i$**, no depende de qué tan cerca esté $x_i$ de la consulta. Una fila
recibe peso negativo cuando queda "del lado contrario" del centro de los datos respecto a la
consulta. OLS ajusta un plano, y esas filas sirven para fijar su inclinación: si la consulta
está de un lado, los vinos del otro lado cuentan en contra.

## ¿Qué filas reciben los pesos más negativos y qué tan lejos están?

Para cada consulta, las 5 filas con el peso más negativo (tabla `parte2_pesos_extremos_ols.csv`):

* **Fila 176:** distancias entre 8.0 y 9.8, en los **percentiles 95 a 99** de distancia.
  Son vinos con `volatile acidity` muy alta (0.79 a 1.24), cuando la consulta tiene 0.28.
* **Fila 67:** distancias entre 8.0 y 10.7, en los **percentiles 80 a 99**.
* **Fila 12:** distancias entre 8.5 y 9.6, en los **percentiles 98 a 99**. Son vinos con
  `sulphates` extremadamente altos (1.61 a 1.98), cuando la consulta tiene 0.52.

Es decir: los pesos más negativos van a los vinos **más lejanos y más raros** del entrenamiento.
Eso es peligroso, porque la predicción depende, en sentido contrario, de vinos que no se
parecen en nada a la consulta.

Los pesos positivos tampoco siguen la distancia. En la fila 176, dos de los cinco pesos más
positivos (filas 1115 y 1150) están en el **percentil 96** de distancia. La correlación entre
la distancia y $|w_i|$ es débil en las tres consultas (−0.12, −0.40 y 0.04): OLS **no es un
método local**.

En la fila 12 hay menos pesos negativos (28%) y todos son más chicos porque la consulta está
cerca del centro de los datos. Ahí el plano casi no tiene que "inclinarse" para llegar a ella.

## Gráfica: peso contra distancia

`resultados/graficas/parte2_pesos_vs_distancia.png` (consulta 176) muestra las tres geometrías:

* **k-NN (k = 29):** una línea horizontal en $1/29 \approx 0.034$ hasta la distancia del
  vecino 29, y cero después. Es un corte duro por distancia.
* **Random forest:** todos los pesos son positivos y los más grandes están a distancias cortas,
  pero el bosque no usa la distancia euclidiana. Decide por las particiones de sus árboles,
  así que puede darle peso a un vino algo lejano si cae en sus mismas hojas.
* **OLS:** una nube que cruza el cero a todas las distancias, con los negativos más grandes en el
  extremo derecho (los vinos más lejanos).

## Conclusión

k-NN y el bosque son **kernels locales y positivos**: su predicción es un promedio de vinos
parecidos y siempre queda entre la nota mínima y la máxima de sus testigos. OLS es un
**kernel global con signo**: usa a todos los vinos, y los más lejanos pesan en contra. Por eso
OLS puede extrapolar fuera del rango de sus testigos, y por eso conviene vigilar su leverage
(criterio 3 de 2.4).