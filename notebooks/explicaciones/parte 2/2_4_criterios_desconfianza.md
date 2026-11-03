# Criterios para Desconfiar de una Predicción (Parte 2.4)

Basado en los análisis geométricos y empíricos del modelo **Random Forest** y modelos lineales sobre el dataset **Wine Quality (tinto)**, se establecen los siguientes criterios clave para alertar a un usuario final (ej. panel evaluador o bodega) que **no debe confiar** en una predicción:

---

### 1. Alta Concentración del Peso $c(x) > 0.50$ (Violación del TLC)
* **Evidencia:** En la consulta de máxima concentración (Test #12), el peso $c(x)$ del Random Forest alcanzó **0.9641**, lo que significa que un solo árbol/hoja o una observación histórica dominó prácticamente el 96% del voto.
* **Riesgo:** Se rompe la condición de Hájek-Šidák del Teorema del Límite Central. La predicción no es un promedio estable de casos similares, sino una réplica casi directa de una sola botella histórica.

### 2. Tamaño Efectivo de Muestra Extremadamente Bajo ($N_{ef} < 10$)
* **Evidencia:** Para la consulta con $N_{ef}$ mínimo (Test #67), el tamaño efectivo de muestra fue de apenas **3.42** observaciones.
* **Riesgo:** Apoyar una decisión comercial o de calidad en menos de 4 botellas históricas incrementa fuertemente la varianza y la fragilidad del resultado.

### 3. Presencia Masiva de Pesos Negativos en Modelos Lineales (OLS > 40%)
* **Evidencia:** OLS asignó entre **45.5% y 46.1% de pesos negativos** en las consultas evaluadas (con pesos individuales tan bajos como $-0.0145$).
* **Riesgo:** OLS extrapola extrapolando puntos lejanos del espacio de características mediante penalizaciones. Si la consulta está en una región dispersa, la predicción puede ser distorsionada por datos sin relación fisicoquímica directa.

### 4. Regiones de Incertidumbre y Desviación Extrema
* **Evidencia:** Aunque la evidencia ($N_{ef}$) sea alta, un cambio brusco en variables críticas como `volatile acidity` o `alcohol` respecto a la media histórica debe marcarse como zona de alerta operativa.
