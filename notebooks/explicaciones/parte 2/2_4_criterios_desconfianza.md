# 2.4 Criterios para desconfiar de una predicción

Criterios para avisarle a la bodega cuándo **no** confiar en una predicción de la nota del
panel. Todos salen de los resultados de 2.1–2.3 sobre Wine tinto (320 filas de prueba).

Antes, un hallazgo que cambia la lectura de todo lo demás: el archivo tiene **240 filas repetidas**,
y **74 de las 320 filas de prueba tienen un gemelo exacto en entrenamiento** (las mismas 11 variables).

---

### 1. Una sola fila de entrenamiento domina el voto: $c(x) > 0.5$

* **Qué mide:** $c(x) = \max_i w_i^2 / \sum_j w_j^2$ es la parte de la suma de cuadrados de los
  pesos que se lleva **una sola fila de entrenamiento** (no un árbol ni una hoja). Si es grande,
  falla la condición de Hájek–Šidák y la predicción no es un promedio estable de muchos casos.
* **Evidencia:** en la fila de prueba 12, el bosque tiene $c(x) = 0.9641$.
  Esa fila tiene un gemelo exacto en entrenamiento, que recibe un peso de 0.371
  (14 veces el peso del siguiente testigo) y tiene la misma nota. En todo el test, 88 filas
  tienen $c(x) > 0.5$, y el 70% de ellas tiene gemelo.
* **Qué hacer:** revisar quién es esa fila dominante. Si es un gemelo, ver el criterio 2.
  Si no lo es, la predicción depende de un solo vino histórico y es frágil
  (MAE de 0.48 en esas 26 filas).

### 2. $N_{ef}$ bajo: primero preguntar si hay un gemelo

* **Lo esperado era** que poca evidencia significara más error. Los datos dicen lo contrario:
  el quintil de menor $N_{ef}$ es el de menor error (ver 2.3).
* **Por qué:** con gemelo exacto el MAE es **0.09**; sin gemelo es **0.49**.
  El bosque (`min_samples_leaf=2`, sin bootstrap) pone casi todo el peso en el gemelo, que tiene
  la misma nota. Un $N_{ef}$ bajo en esos casos no indica falta de evidencia: indica que **la
  respuesta se copió del entrenamiento**, una forma de fuga de datos. El error de prueba de
  esas filas exagera qué tan bien funcionaría el modelo con un vino nuevo.
* **Sin gemelos, la relación esperada aparece:** con $N_{ef} < 83$ el MAE es
  0.53, y con $N_{ef} \ge 83$ baja a 0.33.
* **Regla práctica:** si el vino ya está en la base histórica, la predicción es una copia de su
  nota anterior (útil, pero no es una predicción). Si no está, confiar más cuando
  $N_{ef} \ge 83$.

### 3. OLS extrapolando: leverage alto

* **Evidencia:** en las tres consultas, entre el 28% y el 46% de los pesos de OLS son negativos.
  Los cinco pesos más negativos caen en filas muy lejanas: entre el percentil
  80 y el 99 de distancia a la consulta (ver 2.2).
  OLS ajusta un plano, así que usa vinos lejanos en sentido contrario para "inclinarlo".
* **Cómo detectarlo:** el leverage $h(x) = \tilde x^\top (\tilde X^\top \tilde X)^{-1} \tilde x$
  es exactamente $\sum_i w_i(x)^2$ de OLS. Como los pesos suman 1, $N_{ef}^{OLS} = 1/h(x)$.
  Con la regla usual $h(x) > 2(p+1)/n = 0.0188$, 18 filas de prueba quedan
  marcadas, y en ellas el MAE de OLS es 0.63, contra 0.46
  en las demás.
* **Qué hacer:** si $h(x)$ supera el umbral, la consulta está lejos de los datos y la predicción
  lineal es una extrapolación.

### 4. Vinos excepcionales: el voto ponderado no llega a los extremos

* **Evidencia:** el error del bosque por nota real del panel:

| Nota real | Filas | MAE |
|---|---|---|
| 3 | 2 | 2.44 |
| 4 | 11 | 1.36 |
| 5 | 135 | 0.31 |
| 6 | 142 | 0.32 |
| 7 | 27 | 0.58 |
| 8 | 3 | 1.48 |

  En las notas 3, 4 y 8 el MAE es 1.52, contra 0.31 en las notas 5 y 6.
  Las predicciones del bosque solo van de 4.66 a 7.51.
* **Por qué:** con pesos positivos que suman 1, la predicción es un promedio de notas vecinas.
  Como casi no hay vinos de 3, 4 u 8, sus vecinos son de 5 y 6 y el promedio se jala al centro.
* **Qué hacer:** no usar el modelo para detectar vinos excepcionales (muy malos o muy buenos).
* **Lo que probamos y no sirvió:** el desacuerdo entre los cinco modelos (rango de sus
  predicciones) no se relaciona con el error del bosque (correlación 0.007),
  así que no lo usamos como alarma.
