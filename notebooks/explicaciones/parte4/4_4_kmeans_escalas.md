# 4.4 k-means en escala cruda y estandarizada — D07.T24

**Responsable:** I1 · **Código:** `src/parte4/kmeans_escalas.py` · **Datos:** Wine (tinto), entrenamiento · **Tablas:** `resultados/tablas/4_4_*.csv` · **Figura:** `resultados/graficas/4_4_kmeans_eleccion_k_wine.png`

`KMeans(n_clusters=k, n_init=10, random_state=0)` sobre las 1 279 filas de entrenamiento, con variables crudas y estandarizadas. La nota de calidad no se usa para agrupar; se reporta después solo para interpretar.

## Elección de k

| $k$ | Inercia (crudas) | Inercia (estand.) | Silueta (crudas) | Silueta (estand.) |
|---|---|---|---|---|
| 2 | 573 898 | 11 444 | 0.607 | **0.217** |
| 3 | 331 708 | 10 076 | 0.517 | 0.193 |
| 4 | 237 199 | 8 997 | 0.497 | 0.210 |
| 5 | 186 349 | 8 148 | 0.447 | 0.184 |
| 6 | 142 707 | 7 465 | 0.449 | 0.193 |
| 7 | 121 479 | 6 891 | 0.394 | 0.188 |
| 8 | 106 684 | 6 616 | 0.394 | 0.146 |

- **En escala estandarizada no hay grupos naturales bien definidos:** la inercia baja sin un codo claro y la silueta es baja y casi plana entre $k = 2$ y $k = 4$ (0.19 a 0.22). Los índices no deciden por sí solos; la elección se apoya en la interpretación.
- **Escogemos $k = 3$.** Con $k = 2$ los grupos estandarizados solo separan vinos ácidos de poco ácidos. Con $k = 3$ aparecen tres perfiles químicos reconocibles (ver abajo). Con $k = 4$ uno de los grupos tiene apenas 25 vinos con cloruros y sulfatos extremos: k-means empieza a aislar atípicos en lugar de encontrar perfiles.
- **La silueta cruda es alta, pero engañosa.** Se mide con la distancia cruda, que en Wine es casi unidimensional (los sulfitos son el 99.5 % de la distancia, 4.1). Partir una sola dimensión da grupos "limpios" según esa distancia, aunque ignoren las otras 10 variables. Una silueta alta no significa que los grupos sean más útiles: solo que están bien separados en la geometría que se escogió.

Se usa el mismo $k$ en ambas escalas para que la tabla cruzada compare particiones del mismo tamaño.

## Tabla cruzada ($k = 3$)

| Grupo crudo ↓ / estandarizado → | 0 | 1 | 2 | Total |
|---|---|---|---|---|
| 0 | 196 | 123 | 120 | 439 |
| 1 | 28 | 169 | 6 | 203 |
| 2 | 352 | 0 | 285 | 637 |
| Total | 576 | 292 | 411 | 1 279 |

Índice de Rand ajustado: **0.19** (0 sería coincidencia al azar, 1 particiones idénticas).

- El grupo crudo 1 (sulfitos altos) cae mayoritariamente (169 de 203) en el grupo estandarizado 1, que también es de sulfitos altos: es la única coincidencia clara.
- El grupo crudo 2 (sulfitos bajos, 637 vinos) se reparte entre los grupos estandarizados 0 y 2 (352 y 285), que en escala estandarizada se distinguen por su acidez; ninguno de sus vinos cae en el grupo 1.
- El grupo crudo 0 (sulfitos intermedios) se dispersa entre los tres.

## ¿Qué variable separa los grupos en escala cruda?

$\eta^2_j$ es la fracción de la varianza de la variable $j$ explicada por los grupos (entre 0 y 1, independiente de las unidades):

| Variable | $\eta^2$ grupos crudos | $\eta^2$ grupos estandarizados |
|---|---|---|
| total sulfur dioxide | **0.829** | 0.523 |
| free sulfur dioxide | 0.456 | 0.365 |
| residual sugar | 0.048 | 0.067 |
| alcohol | 0.036 | 0.106 |
| fixed acidity | 0.016 | 0.502 |
| volatile acidity | 0.011 | 0.265 |
| pH | 0.006 | 0.336 |
| density | 0.006 | 0.167 |
| chlorides | 0.004 | 0.032 |
| sulphates | 0.004 | 0.151 |
| citric acid | 0.000 | **0.609** |

**En escala cruda, el dióxido de azufre total separa los grupos** ($\eta^2 = 0.83$), acompañado del libre, que está correlacionado con él. Ninguna otra variable supera 0.05. Los centros crudos de dióxido de azufre total son **22, 54 y 109 mg/L**: k-means crudo simplemente parte los vinos en tres niveles de sulfitos. El ácido cítrico, que es lo que más separa en escala estandarizada, no explica nada ($\eta^2 = 0.000$) en la agrupación cruda.

## Interpretación de los grupos estandarizados

Centros en unidades originales:

| | Grupo 0 | Grupo 1 | Grupo 2 |
|---|---|---|---|
| Vinos | 576 | 292 | 411 |
| fixed acidity | 7.21 | 8.18 | 10.04 |
| volatile acidity | 0.618 | 0.537 | 0.401 |
| citric acid | 0.120 | 0.291 | 0.472 |
| total sulfur dioxide | 36.0 | 91.6 | 31.1 |
| sulphates | 0.605 | 0.629 | 0.757 |
| alcohol | 10.46 | 9.83 | 10.76 |
| Nota media (no usada) | 5.54 | 5.34 | 6.02 |

- **Grupo 0:** poco ácido fijo y cítrico, acidez volátil alta (la que da olor avinagrado).
- **Grupo 1:** muchos sulfitos y menos alcohol; la peor nota media.
- **Grupo 2:** ácido, con más ácido cítrico y sulfatos, poca acidez volátil y más alcohol; la mejor nota media.

Aunque la nota no se usó para agrupar, los grupos estandarizados se relacionan más con ella (medias de 5.34 a 6.02, rango 0.68) que los crudos (5.33 a 5.76, rango 0.43).

## Conclusión

Como en k-NN (4.1, 4.2), la escala decide qué significa "parecido": en crudo, k-means agrupa por un solo eje (sulfitos); estandarizado, encuentra perfiles que combinan acidez, sulfitos, sulfatos y alcohol. Las dos particiones casi no coinciden (ARI 0.19). Ninguna es "la correcta": cada una responde a una noción distinta de distancia, y escoger la escala es escoger el kernel.