# Proyecto final: Todo modelo es un voto ponderado

**Análisis y Ciencia de Datos · Facultad de Ingeniería, Universidad del Istmo de Guatemala**
**Grupo 2:** Sofía Blanco · Katherine Diéguez · Daniela Natareno

Casi todos los modelos del curso predicen como un voto ponderado de las etiquetas de entrenamiento,
$\hat y(x) = \sum_i w_i(x)\, y_i$. En este proyecto le sacamos los pesos a cinco modelos (OLS, ridge,
k-NN, árbol y random forest), comprobamos que reproducen exactamente su `.predict()`, medimos cuánta
evidencia respalda cada predicción y cuánto cuesta la flexibilidad. Basado en la serie
*The Learned Kernel* de Agus Sudjianto.

## Datos

| Parte | Datos | Archivo |
|---|---|---|
| 1 | California Housing (20 640 distritos, 8 variables) | `fetch_california_housing()`, con caché en `data/california/` |
| 2 a 5 | Wine Quality, tinto (1 599 vinos, 11 variables, `y = quality`) | `data/wine/winequality-red.csv` (separado por `;`) |

Partición 80/20 con `random_state=0`; `StandardScaler` ajustado solo con entrenamiento.

## Cómo correrlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_final.ipynb   # Run All, ~3 minutos
```

El notebook ejecuta todos los scripts en orden y regenera `resultados/` desde los archivos originales.
Cada script también se puede correr solo desde la raíz, por ejemplo `python -m src.parte1.verificacion`.
`src.parte2.cv_modelos` debe correr antes que el resto de la Parte 2, porque guarda los modelos ajustados.

Probado con Python 3.12, scikit-learn 1.9, numpy 2.5 y pandas 3.0.

## Estructura

```
data/                      datos originales y particiones train/test
src/
  comunes/                 configuración, carga, partición y escalado
  parte1/                  pesos de los cinco modelos, verificación y proximidad (California)
  parte2/                  validación cruzada, N_ef, c(x), testigos, quintiles, criterios
  parte3/                  grados de libertad, error vs tr(S), optimismo, Monte Carlo, OOB
  parte4/                  escala: k-NN, vecinos, árbol y k-means con y sin estandarizar
notebooks/
  proyecto_final.ipynb     corre todo de principio a fin
  explicaciones/           derivaciones y respuestas por inciso (.md)
resultados/                tablas (.csv) y gráficas (.png) que genera el notebook
informe/memo_parte5.md     memo para la bodega (Parte 5)
presentacion/              diapositivas en PDF
```

## Resultados principales

- **Parte 1.** Los pesos de los cinco modelos reproducen `.predict()` con error menor a $10^{-14}$.
  Ridge con el intercepto penalizado suma $n/(n+\lambda)$ y se sesga hacia cero. El random forest
  con bootstrap solo cuadra si se pesa por cuántas veces salió cada fila en el sorteo.
- **Parte 2.** El quintil con menos evidencia ($N_{ef}$ bajo) es el que *menos* se equivoca: el 23 %
  de los vinos de prueba tiene un gemelo exacto en entrenamiento y el bosque copia su nota.
  Sin gemelos, más evidencia sí significa menos error (MAE 0.53 contra 0.33).
- **Parte 3.** La fórmula $2\sigma^2\,\mathrm{tr}(S)/n$ cuadra con la simulación para ridge y k-NN, y
  falla para los árboles porque eligen sus cortes mirando $y$. Con filas repetidas, la traza de
  k-NN con $k=1$ es 1122 y no 1279.
- **Parte 4.** En escala cruda, el dióxido de azufre total explica el 91 % de la distancia entre
  vinos. Estandarizar mejora k-NN y cambia por completo los grupos de k-means.
- **Parte 5.** Recomendamos el random forest. Su error de prueba sube de 0.32 a 0.47 (MSE) al
  quitar los vinos repetidos antes de partir, y esa es la decisión que más cambia la respuesta.

## Referencias

- A. Sudjianto, *The Learned Kernel* (Substack, 2026) y su código: github.com/asudjianto-xml/Learned-Kernel
- T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, 2a ed., 2009.
- P. Cortez et al., Modeling Wine Preferences by Data Mining from Physicochemical Properties,
  *Decision Support Systems* 47(4), 2009.
- R. K. Pace y R. Barry, Sparse Spatial Autoregressions, *Statistics & Probability Letters* 33(3), 1997.