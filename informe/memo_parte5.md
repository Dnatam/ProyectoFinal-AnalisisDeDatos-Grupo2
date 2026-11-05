# Memo: modelo para predecir la nota del panel de cata

**Para:** Dirección técnica de la bodega
**De:** Grupo 2 — Análisis y Ciencia de Datos, UNIS
**Fecha:** 6 de noviembre de 2026
**Asunto:** Qué modelo usar, cuándo no confiar en él y qué decisión pesa más

## Resumen

Recomendamos un **random forest**. Predice la nota del panel con un error típico de
**medio punto** en vinos nuevos, y cada predicción viene con su propia explicación: la lista
de vinos históricos en los que se apoya. Pero el error que se ve en una evaluación ingenua
(0.40 puntos) es **demasiado optimista**: la base de datos tiene vinos repetidos, y al
evaluarlo bien el error real sube a unos 0.5 puntos. Además, el modelo **no sirve para
detectar vinos excepcionales** (notas 3, 4 u 8).

## 1. Qué modelo recomendamos y por qué

Comparamos cinco modelos con los 1 599 vinos tintos (80% para ajustar y 20% para evaluar).
Todos predicen igual: la nota de un vino nuevo es un **promedio ponderado de las notas de
vinos históricos**. Lo que cambia entre ellos es a qué vinos escuchan y cuánto.

**Por precisión.** El random forest tiene el menor error en todos los escenarios que
probamos: 0.32 de error cuadrático medio en la evaluación original, contra 0.38 de los
modelos lineales (OLS y ridge) y 0.43–0.46 de k-NN y del árbol. Cuando se evalúa con vinos
que no estaban en la base, la ventaja se reduce, pero se mantiene: 0.47 contra 0.53 de ridge.

**Por evidencia.** En cada predicción del bosque, los vinos que más pesan son vinos
parecidos al consultado, y todos pesan a favor. En un caso típico, la predicción se apoya
en el equivalente a unos 50 vinos. Los modelos lineales, en cambio, usan los 1 279 vinos a
la vez y le dan peso **negativo** a cerca de la mitad, sobre todo a los más distintos al vino
consultado. Es decir: su predicción depende, en sentido contrario, de vinos que no se le parecen.

**Por facilidad de explicación.** Del bosque se puede decir algo concreto a un enólogo:
*"este vino se parece a estos cinco de la base, que sacaron 5, 5, 6, 7 y 6"*. Un modelo
lineal solo permite decir cuánto sube la nota por cada grado de alcohol, en promedio sobre
todos los vinos, aunque el vino en cuestión no se parezca al promedio.

**Advertencia.** Ridge queda cerca del bosque en vinos nuevos y es más simple de mantener.
Si la bodega prefiere una regla fija y global antes que explicaciones caso por caso,
ridge es una alternativa razonable.

## 2. Cuándo no confiar en una predicción

Recomendamos entregar cada predicción con una **ficha de evidencia** y aplicar estas alarmas:

1. **El vino ya está en la base.** El 23% de los vinos de evaluación tenía un gemelo
   idéntico en la base histórica. En esos casos el modelo copia la nota del gemelo (error de
   0.09). Es útil saberlo, pero no es una predicción: es un dato que ya se tenía.
2. **Pocos vinos de respaldo.** Para vinos nuevos, cuando la predicción se apoya en el
   equivalente a 83 o más vinos, el error baja a **0.33 puntos**. Con menos respaldo,
   el error es de unos 0.53.
3. **Un solo vino domina.** Si un único vino histórico se lleva casi todo el peso y no es
   un gemelo, la predicción depende de ese solo caso y es frágil (error de 0.48).
4. **Vinos excepcionales.** El modelo nunca predice menos de 4.7 ni más de 7.5. En vinos
   que el panel calificó con 3, 4 u 8, el error promedio es de **1.5 puntos**, contra 0.31
   en vinos de 5 o 6. Para detectar vinos muy malos o muy buenos hace falta otro método.

En la práctica, la ficha muestra la predicción, el número de vinos de respaldo, si hay
un gemelo y los cinco vinos históricos que más pesan. Con eso cualquier enólogo puede
juzgar si la predicción tiene sentido.

## 3. Qué decisión de modelado cambiaría más la respuesta

**La partición de los datos, por los vinos repetidos.** Es la decisión que más pesa. La
base tiene 240 filas repetidas. Si se parte sin quitarlas, copias del mismo vino quedan en
entrenamiento y en evaluación, y el modelo parece mejor de lo que es. Quitándolas antes de
partir, el error del bosque sube de 0.32 a 0.47 (un 48% más), y su ventaja sobre ridge baja
del 17% al 11%. Cualquier cifra de precisión que se reporte debe venir de una evaluación
sin repetidos.

**La flexibilidad del modelo.** Un árbol de profundidad 6 tiene un error de evaluación de
0.41, y uno de profundidad 10 sube a 0.67: memoriza los datos en vez de aprender patrones.
Además, el error medido sobre los mismos datos de ajuste engaña: en los árboles, la
diferencia real es unas cinco veces mayor que la que predice la fórmula teórica, porque el
árbol elige sus cortes mirando las notas. Para el bosque, la evaluación *out-of-bag* da un
error muy cercano al de prueba (0.34 contra 0.32) y es una buena forma de medirlo sin
apartar datos.

**La escala de las variables.** En los modelos basados en distancias, si no se estandarizan
las variables, el dióxido de azufre total explica el 91% de la distancia entre vinos y las
demás casi no cuentan. Estandarizar mejora a k-NN (error absoluto de 0.55 a 0.51) y cambia
por completo los grupos que encuentra k-means. Para el bosque, la escala casi no importa.

## Recomendación final

Usar el random forest, **reportar su error con una evaluación sin vinos repetidos**
(alrededor de 0.5 puntos) y entregar cada predicción con su ficha de evidencia. No usarlo
para identificar vinos excepcionales.