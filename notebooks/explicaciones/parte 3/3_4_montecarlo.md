# 3.4. Verificación del optimismo mediante simulación Monte Carlo

## Objetivo

Contrastar el optimismo empírico con la identidad teórica para suavizadores lineales fijos:

$$
\operatorname{Opt}=\mathbb{E}[\operatorname{MSE}_{\mathrm{new}}-\operatorname{MSE}_{\mathrm{train}}]=\frac{2\sigma^2}{n}\operatorname{tr}(S).
$$

## Diseño experimental

Se empleó el conjunto de entrenamiento estandarizado de Wine Quality Red, con $n=1279$ observaciones y matriz de predictores $X$ fija. Se ajustó un Random Forest de referencia a las respuestas observadas, con 300 árboles, `min_samples_leaf=10`, `max_features=0.33`, bootstrap y predicciones fuera de bolsa (OOB). La función de referencia fue $\hat f(X)$ y se utilizó el RMSE OOB como estimación operativa de la desviación del ruido, $\hat\sigma=0.613551$ ($\hat\sigma^2=0.376444$). Esta estimación puede incluir error de aproximación del modelo, por lo que no debe interpretarse como una identificación exacta del ruido irreducible real.

En cada una de 50 réplicas independientes se generaron dos vectores de respuestas en los mismos puntos $X$:

$$
y^{(b)}=\hat f(X)+\hat\sigma\varepsilon^{(b)},\qquad
 y_{\mathrm{new}}^{(b)}=\hat f(X)+\hat\sigma\varepsilon_{\mathrm{new}}^{(b)},
$$

con errores normales estándar independientes. Cada configuración se ajustó a $y^{(b)}$ y se evaluó sobre las respuestas de entrenamiento y las nuevas respuestas independientes, **sin volver a entrenar para evaluar**:

$$
\widehat{\operatorname{Opt}}^{(b)}=\operatorname{MSE}(y_{\mathrm{new}}^{(b)},\hat y^{(b)})-\operatorname{MSE}(y^{(b)},\hat y^{(b)}).
$$

Se compararon Ridge ($\alpha=100$), k-NN ($k=5$ y $k=30$, con autoinclusión en la matriz suavizadora) y árboles de regresión (`max_depth=6` y `max_depth=10`). Se obtuvieron 250 registros: 50 réplicas por cada una de cinco configuraciones.

## Resultados

| Configuración | $\overline{\operatorname{Opt}}$ empírico | Error estándar | Referencia $2\hat\sigma^2\operatorname{tr}(S)/n$ | Diferencia |
|---|---:|---:|---:|---:|
| Ridge, $\alpha=100$ | 0.006581 | 0.002901 | 0.006080 | +0.000501 |
| k-NN, $k=30$ | 0.026394 | 0.003265 | 0.025096 | +0.001297 |
| k-NN, $k=5$ | 0.150297 | 0.003348 | 0.150578 | -0.000281 |
| Árbol, profundidad 6 | 0.172837 | 0.002935 | 0.035437 | +0.137400 |
| Árbol, profundidad 10 | 0.460840 | 0.004744 | 0.220639 | +0.240200 |

El error estándar se estimó como la desviación estándar muestral de los 50 optimismos, dividida entre $\sqrt{50}$.

### Figura 1. Comparación de optimismo empírico y referencia de traza

Insertar `resultados/graficas/parte3_montecarlo_comparacion.png` en el notebook. Las barras de error del optimismo empírico representan $\pm1$ error estándar.

### Figura 2. Distribución de optimismo entre réplicas

Insertar `resultados/graficas/parte3_montecarlo_distribucion.png` en el notebook. Cada diagrama de caja resume las 50 réplicas de una configuración; el rombo representa la media y la línea central, la mediana.

## Interpretación

En Ridge y k-NN, el optimismo empírico promedio se aproxima a la predicción teórica; las diferencias son pequeñas frente al error estándar de Monte Carlo. Este comportamiento es consistente con la identidad de optimismo para matrices suavizadoras $S$ fijas, independientes de las respuestas.

En los árboles, la referencia calculada usando $\operatorname{tr}(S)$ igual al número de hojas es sustancialmente inferior al optimismo empírico. La cantidad de hojas representa la traza de la matriz de promedios **si se mantienen fijas las particiones**, pero los árboles ajustados vuelven a seleccionar las particiones según las respuestas en cada réplica. Por ello, el valor de traza mostrado **no constituye una predicción teórica exacta del optimismo de los árboles adaptativos**.

El árbol de profundidad 10 mostró el mayor optimismo (0.460840), consistente con una mayor capacidad para adaptarse al ruido de entrenamiento. Los resultados evidencian que el error de entrenamiento por sí solo no refleja el costo de esa flexibilidad.

### Alcance y limitaciones de la simulación

La simulación reproduce un escenario de diseño fijo, en el que se mantienen los mismos predictores y se generan nuevas respuestas independientes. Por lo tanto, el optimismo estimado no equivale directamente al error de generalización sobre nuevas observaciones con valores diferentes de las variables predictoras.

Aunque la escala del ruido se obtuvo a partir del error OOB de un Random Forest de referencia, una vez fijada la varianza \(\hat{\sigma}^2\), esta constituye la varianza conocida del ruido generado artificialmente en el experimento.

Las diferencias observadas en los árboles son consistentes con el efecto de seleccionar adaptativamente las particiones, pero no deben interpretarse como una medición exacta y aislada del costo de seleccionar los cortes. Los resultados corresponden a 50 réplicas de un escenario simulado específico y no representan una conclusión universal para todos los árboles o conjuntos de datos.

## Archivos para reproducibilidad

- `src/parte3/simulacion_optimismo.py`: generación de réplicas y ajuste de modelos.
- `src/parte3/resultados_montecarlo.py`: agregación estadística y elaboración de figuras.
- `resultados/tablas/parte3_montecarlo_repeticiones.csv`: 250 resultados individuales.
- `resultados/tablas/parte3_montecarlo_resumen.csv`: resumen de medias, errores estándar y referencias.
- `resultados/graficas/parte3_montecarlo_comparacion.png`: comparación entre optimismos.
- `resultados/graficas/parte3_montecarlo_distribucion.png`: variabilidad entre réplicas.