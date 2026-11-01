# Parte 3.2 — Error de predicción frente a complejidad

## Objetivo y procedimiento

Estudiar cómo varían el error de entrenamiento y el de prueba según $\operatorname{tr}(S)$ para OLS, Ridge, k-NN, árboles de regresión y Random Forest, usando Wine Quality (red).

Se reutilizó la partición 80/20 (`random_state=0`): $n_{train}=1279$ y $n_{test}=320$, con estandarización ajustada solo en entrenamiento. Para cada configuración se calculó:

$$
\operatorname{MSE}=\frac1m\sum_{i=1}^{m}(y_i-\hat y_i)^2,
\qquad
\operatorname{Brecha}=\operatorname{MSE}_{test}-\operatorname{MSE}_{train}.
$$

Se evaluaron **32 configuraciones**: OLS (1), Ridge (6 valores de $\lambda$), k-NN (8 valores de $k$), árbol (7 profundidades y 5 mínimos de hoja con profundidad 6) y Random Forest (5 mínimos de hoja). Para el bosque se usaron 200 árboles, `max_features=0.33` y `bootstrap=False`, con traza condicional calculada a partir de las hojas.

La figura presenta las cinco familias conjuntamente: línea continua para entrenamiento y discontinua para prueba. El eje de $\operatorname{tr}(S)$ utiliza escala logarítmica para hacer visibles tanto los modelos con pocos grados de libertad como k-NN con $k=1$ ($df=1279$).

## Resultados representativos

| Modelo | Configuración | $df$ | MSE entrenamiento | MSE prueba |
|---|---|---:|---:|---:|
| OLS | Sin regularización | 12.00 | 0.4258 | 0.3845 |
| Ridge | $\lambda=100$ | 10.33 | 0.4268 | 0.3840 |
| Ridge | $\lambda=10000$ | 2.16 | 0.5943 | 0.5103 |
| k-NN | $k=1$ | 1279.00 | 0.0000 | 0.5938 |
| k-NN | $k=100$ | 12.79 | 0.4406 | 0.4112 |
| Árbol | profundidad 6 | 50.00 | 0.2947 | 0.4150 |
| Árbol | profundidad 12 | 273.00 | 0.0430 | 0.6932 |
| Random Forest | mínimo 1 por hoja | 434.74 | 0.0000 | 0.3189 |
| Random Forest | mínimo 10 por hoja | 98.95 | 0.2062 | 0.3419 |

### Mínimos observados de MSE de prueba por familia

| Familia | Configuración con menor MSE de prueba en el barrido | MSE prueba |
|---|---|---:|
| OLS | Sin regularización | 0.3845 |
| Ridge | $\lambda=100$ | 0.3840 |
| k-NN | $k=100$ | 0.4112 |
| Árbol | profundidad 6 (mínimo 1 por hoja) | 0.4150 |
| Random Forest | mínimo 1 por hoja | 0.3189 |

**Advertencia metodológica:** estos mínimos describen únicamente los resultados observados en el conjunto de prueba. No deben utilizarse para elegir hiperparámetros finales; esa selección debe hacerse con validación cruzada dentro del entrenamiento.

## Interpretación

- **OLS y Ridge:** penalizaciones pequeñas producen errores similares. Una penalización excesiva reduce mucho la flexibilidad y eleva ambos errores.
- **k-NN:** con $k=1$, el error de entrenamiento es cero, pero el error de prueba es alto. Aumentar $k$ reduce los grados de libertad y, en el rango probado, mejora el error de prueba.
- **Árboles:** al crecer la profundidad, disminuye el error de entrenamiento; el error de prueba se deteriora a profundidades altas, señal de sobreajuste.
- **Random Forest:** el error de prueba observado fue relativamente bajo, aunque el bosque sin bootstrap y con hojas de tamaño 1 también puede interpolar el entrenamiento.
- **Misma traza, diferente error:** dos árboles con 50 hojas (`max_depth=6` y `depth=6, leaf=5`) obtuvieron MSE de prueba 0.4150 y 0.4496, respectivamente. La traza resume la complejidad, no determina por sí sola la calidad de las particiones.
- **Brecha negativa:** OLS presentó $0.3845-0.4258\approx-0.0413$. Una brecha observada negativa es posible en una partición particular y no contradice que el optimismo *esperado* sea no negativo bajo el modelo teórico.

**Precauciones:** el trazado une configuraciones de profundidad y mínimo de hoja dentro de una sola curva de árbol, aunque son dos barridos diferentes. Además, la fórmula $n/k$ presupone autoinclusión explícita, mientras que las predicciones de `KNeighborsRegressor` pueden resolver de otro modo empates entre filas duplicadas.

## Archivos para integrar

- Código: `src/parte3/error_complejidad.py`.
- Tabla completa: `resultados/tablas/parte3_error_complejidad.csv`.
- Figura: `resultados/graficas/parte3_error_complejidad.png`.

Los resultados obtenidos permiten comparar el comportamiento de los modelos frente a distintos niveles de complejidad. La comparación se realizó utilizando la misma partición de entrenamiento y prueba, sin seleccionar un modelo final a partir de los errores observados en el conjunto de prueba.

### Consideraciones sobre la evaluación de la complejidad

Las curvas de error de entrenamiento y prueba permiten analizar cómo cambia la capacidad de generalización de los modelos al modificar sus hiperparámetros. Una disminución del error de entrenamiento no garantiza una mejora en el conjunto de prueba, ya que una mayor flexibilidad puede aumentar el riesgo de sobreajuste.

Asimismo, en árboles de regresión y Random Forest, la traza de la matriz construida a partir de las particiones ajustadas describe su estructura de promedios, pero no representa necesariamente toda la complejidad adaptativa asociada a seleccionar dichas particiones. Por lo tanto, modelos con trazas similares pueden presentar errores de predicción diferentes.