# Parte 3.3 — Derivación del optimismo y condiciones de validez

## Objetivo

Demostrar que el optimismo, definido como la diferencia esperada entre el error con nuevas respuestas para los **mismos predictores** y el error de entrenamiento, satisface:

$$
\operatorname{Opt}=\frac{2}{n}\sum_{i=1}^{n}\operatorname{Cov}(\hat y_i,y_i).
$$

Y, cuando $\hat{\mathbf y}=S\mathbf y$, $S$ no depende de $y$ y los errores son independientes y homocedásticos:

$$
\boxed{\operatorname{Opt}=\frac{2\sigma^2}{n}\operatorname{tr}(S)}.
$$

## Supuestos y notación

Condicionamos en una matriz de predictores fija $X$. Para cada fila $i$:

$$
y_i=f(x_i)+\varepsilon_i,\qquad
\mathbb E[\varepsilon_i]=0,\qquad
\operatorname{Var}(\varepsilon_i)=\sigma^2.
$$

Los errores de observaciones diferentes son independientes. Construimos una nueva respuesta $y_i^{new}=f(x_i)+\varepsilon_i'$ con ruido independiente del conjunto original, pero la misma distribución. Las predicciones $\hat y_i$ se obtienen ajustando el modelo **solo con los datos originales**.

## Paso 1. Definir ambos errores

$$
\operatorname{Err}_{train}=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2,
\qquad
\operatorname{Err}_{new}=\frac1n\sum_{i=1}^{n}(y_i^{new}-\hat y_i)^2.
$$

El optimismo es:

$$
\operatorname{Opt}=\mathbb E[\operatorname{Err}_{new}-\operatorname{Err}_{train}\mid X].
$$

## Paso 2. Expandir la diferencia

Para una observación:

$$
\begin{aligned}
&(y_i^{new}-\hat y_i)^2-(y_i-\hat y_i)^2\\
&=(y_i^{new})^2-y_i^2+2y_i\hat y_i-2y_i^{new}\hat y_i.
\end{aligned}
$$

Como $y_i$ y $y_i^{new}$ tienen igual distribución condicional en $X$:

$$
\mathbb E[(y_i^{new})^2\mid X]=\mathbb E[y_i^2\mid X].
$$

Como $y_i^{new}$ es independiente de $\hat y_i$ condicional en $X$:

$$
\mathbb E[y_i^{new}\hat y_i\mid X]
=\mathbb E[y_i^{new}\mid X]\,\mathbb E[\hat y_i\mid X].
$$

Y $\mathbb E[y_i^{new}\mid X]=\mathbb E[y_i\mid X]$. Al sustituir:

$$
\begin{aligned}
\operatorname{Opt}
&=\frac2n\sum_{i=1}^{n}
\left(\mathbb E[y_i\hat y_i\mid X]
-\mathbb E[y_i\mid X]\mathbb E[\hat y_i\mid X]\right)\\
&=\boxed{\frac2n\sum_{i=1}^{n}\operatorname{Cov}(\hat y_i,y_i\mid X)}.
\end{aligned}
$$

Esta identidad de covarianzas no requiere por sí sola que el estimador sea lineal.

## Paso 3. Introducir la matriz suavizadora

Si $\hat{\mathbf y}=S\mathbf y$, entonces:

$$
\hat y_i=\sum_{j=1}^{n}S_{ij}y_j.
$$

Suponiendo que $S$ es fija condicional en $X$ y no depende de las respuestas observadas:

$$
\begin{aligned}
\operatorname{Cov}(\hat y_i,y_i\mid X)
&=\operatorname{Cov}\left(\sum_{j=1}^{n}S_{ij}y_j,y_i\mid X\right)\\
&=\sum_{j=1}^{n}S_{ij}\operatorname{Cov}(y_j,y_i\mid X).
\end{aligned}
$$

Por independencia y varianza común:

$$
\operatorname{Cov}(y_j,y_i\mid X)=
\begin{cases}
\sigma^2,&j=i,\\
0,&j\ne i.
\end{cases}
$$

Por tanto:

$$
\operatorname{Cov}(\hat y_i,y_i\mid X)=\sigma^2S_{ii}.
$$

Sustituyendo en la identidad del optimismo:

$$
\begin{aligned}
\operatorname{Opt}
&=\frac2n\sum_{i=1}^{n}\sigma^2S_{ii}\\
&=\frac{2\sigma^2}{n}\sum_{i=1}^{n}S_{ii}\\
&=\boxed{\frac{2\sigma^2}{n}\operatorname{tr}(S)}.
\end{aligned}
$$

La identidad del optimismo expresada mediante covarianzas es válida para procedimientos de ajuste más generales, siempre que las nuevas respuestas sean independientes de los datos utilizados para entrenar el modelo. En cambio, su simplificación mediante la traza requiere que la matriz suavizadora sea fija respecto de las respuestas observadas, además de los supuestos de independencia y homocedasticidad.

Por lo tanto, es importante distinguir entre la traza de una matriz construida después de ajustar un modelo y los grados de libertad efectivos del procedimiento completo. Esta diferencia resulta especialmente relevante para los árboles de regresión y Random Forest, cuyas particiones se seleccionan utilizando las respuestas de entrenamiento.

## Condiciones de validez e interpretación

1. **Diseño fijo:** se comparan nuevas respuestas en los mismos $x_i$, no nuevos valores de $X$.
2. **Independencia de la nueva respuesta:** $y^{new}$ no interviene en el ajuste de $\hat y$.
3. **Suavizador fijo respecto de $y$:** permite extraer $S_{ij}$ de las covarianzas.
4. **Errores independientes y homocedásticos:** la matriz de covarianzas de $y$ es $\sigma^2I$.
5. **Hiperparámetros fijados sin usar las respuestas:** por ejemplo, elegir $\lambda$ o $k$ de antemano conserva la condición de suavizador fijo; seleccionarlos adaptativamente usando $y$ puede invalidarla.

**OLS y Ridge:** con especificación e hiperparámetros fijos, satisfacen la condición de linealidad con $S$ independiente de $y$.

**k-NN:** con distancias, vecindarios y reglas de desempate determinados exclusivamente por $X$, también satisface la condición.

**Árboles y Random Forest:** sus particiones se seleccionan utilizando las respuestas de entrenamiento. Como se comprobó en el inciso 3.1, para un árbol ajustado la traza de su matriz suavizadora coincide con el número de hojas, mientras que para un bosque sin bootstrap coincide con el promedio del número de hojas de sus árboles. Sin embargo, estas igualdades describen las matrices correspondientes a las particiones aprendidas y no incorporan necesariamente toda la complejidad asociada a seleccionarlas. Por ello, la expresión \(2\sigma^2\operatorname{tr}(S)/n\) debe interpretarse como una referencia estructural y no como una fórmula exacta del optimismo para estos procedimientos adaptativos.

## Ejemplos con Wine ($n=1279$)

Para OLS, $\operatorname{tr}(S)=12$:

$$
\operatorname{Opt}_{OLS}=\frac{2\sigma^2(12)}{1279}\approx0.01876\sigma^2.
$$

Para k-NN con $k=5$, $\operatorname{tr}(S)=1279/5=255.8$:

$$
\operatorname{Opt}_{kNN,k=5}=\frac{2\sigma^2(255.8)}{1279}=0.4\sigma^2.
$$

Estos ejemplos mantienen $\sigma^2$ simbólica. Su valor se estimará para la simulación del inciso 3.4 mediante el RMSE out-of-bag de un Random Forest de referencia.

## Relación con los siguientes incisos

En **3.4**, se mantendrá $X_{train}$ fijo, se generarán 50 pares independientes de respuestas simuladas y se comparará el optimismo empírico con la expresión teórica. En **3.5**, se explicará por qué la brecha observada entre prueba y entrenamiento no es lo mismo que este optimismo de diseño fijo. En **3.6**, se compararán los errores de entrenamiento, OOB y prueba de un bosque con bootstrap.

**Referencia conceptual del enunciado:** Proyecto final, Parte 3, inciso 3.3; referencia allí indicada a *The Elements of Statistical Learning*, §7.4.
