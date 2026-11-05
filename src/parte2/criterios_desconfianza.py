from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte1.pesos_ols import agregar_intercepto
from src.parte2.modulo_evidencia import (
    DIR_EXPLICACIONES,
    DIR_TABLAS,
    ORDEN_MODELOS,
    cargar_artefactos,
)

UMBRAL_C = 0.5


def leer_tabla(nombre):
    ruta = DIR_TABLAS / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Corre antes seleccion_consultas.py y quintiles_nef_mae.py"
        )
    return pd.read_csv(ruta)


def leverage_ols(X_train, X):
    """h(x) = x~^T (X~^T X~)^{-1} x~. Coincide con sum_i w_i(x)^2 de OLS, es decir 1/N_ef."""
    Xt = agregar_intercepto(X_train)
    xt = agregar_intercepto(X)
    A = np.linalg.solve(Xt.T @ Xt, xt.T)       # (p+1, m)
    return np.einsum("ij,ji->i", xt, A)


def redactar_criterios_y_validar():
    print("2.4 Criterios para desconfiar de una predicción\n")

    consultas = leer_tabla("parte2_consultas_evidencia.csv")
    ev = leer_tabla("parte2_evaluacion_test.csv")
    q_sin = leer_tabla("parte2_quintiles_sin_gemelo.csv")
    modelos, datos = cargar_artefactos()
    X_train_std, X_test_std = datos["X_train_std"], datos["X_test_std"]
    y_test = datos["y_test"]

    # --- Criterio 1: concentración alta ---
    rf_12 = consultas[(consultas["Consulta"] == "Interés (Máx Concentración)")
                      & (consultas["Modelo"] == "Random Forest")].iloc[0]
    alta_c = ev["c_x"] > UMBRAL_C
    n_alta_c = int(alta_c.sum())
    pct_gemelo_alta_c = 100 * ev.loc[alta_c, "tiene_gemelo"].mean()
    testigos_12 = leer_tabla("parte2_testigos.csv")
    testigos_12 = testigos_12[(testigos_12["Idx_Test"] == rf_12["Idx_Test"])
                              & (testigos_12["Modelo"] == "Random Forest")]["peso_w"].abs().values
    veces_siguiente = testigos_12[0] / testigos_12[1]
    mae_alta_c_sin = ev.loc[alta_c & ~ev["tiene_gemelo"], "error_abs"].mean()
    n_alta_c_sin = int((alta_c & ~ev["tiene_gemelo"]).sum())

    # --- Criterio 2: N_ef bajo, separando gemelos ---
    n_dup = int(pd.read_csv(ROOT_DIR / "data" / "wine" / "winequality-red.csv",
                            sep=";").duplicated().sum())
    mae_con = ev.loc[ev["tiene_gemelo"], "error_abs"].mean()
    mae_sin = ev.loc[~ev["tiene_gemelo"], "error_abs"].mean()
    n_con = int(ev["tiene_gemelo"].sum())
    umbral_nef = float(q_sin["N_ef_min"].iloc[-1])  # inicio del quintil de mayor evidencia
    sin = ev[~ev["tiene_gemelo"]]
    mae_bajo = sin.loc[sin["N_ef"] < umbral_nef, "error_abs"].mean()
    mae_alto = sin.loc[sin["N_ef"] >= umbral_nef, "error_abs"].mean()

    # --- Criterio 3: OLS extrapolando (leverage alto) ---
    p = X_train_std.shape[1]
    n = X_train_std.shape[0]
    h = leverage_ols(X_train_std, X_test_std)
    umbral_h = 2 * (p + 1) / n
    lejos = h > umbral_h
    e_ols = np.abs(y_test - modelos["OLS"].predict(X_test_std))
    neg = leer_tabla("parte2_pesos_extremos_ols.csv")
    pct_neg = neg.loc[neg["tipo"] == "más negativo", "percentil_distancia"]
    resumen_neg = leer_tabla("parte2_pesos_negativos_ols.csv")["porcentaje_negativos"]

    # --- Criterio 4: vinos excepcionales y desacuerdo entre modelos ---
    P = np.column_stack([modelos[m].predict(X_test_std) for m in ORDEN_MODELOS])
    rango = P.max(axis=1) - P.min(axis=1)
    corr_rango = np.corrcoef(rango, ev["error_abs"])[0, 1]
    por_nota = ev.groupby("y_real")["error_abs"].agg(["size", "mean"])
    extremas = ev["y_real"].isin([3, 4, 8])
    mae_extremas = ev.loc[extremas, "error_abs"].mean()
    mae_centrales = ev.loc[ev["y_real"].isin([5, 6]), "error_abs"].mean()

    resumen = pd.DataFrame([
        {"criterio": "1. c(x) > 0.5", "filas_test": n_alta_c,
         "detalle": f"{pct_gemelo_alta_c:.0f}% con gemelo; MAE sin gemelo {mae_alta_c_sin:.3f}"},
        {"criterio": "2. gemelo en train", "filas_test": n_con,
         "detalle": f"MAE con gemelo {mae_con:.3f} vs sin gemelo {mae_sin:.3f}"},
        {"criterio": f"2. N_ef < {umbral_nef:.1f} (sin gemelo)",
         "filas_test": int((sin["N_ef"] < umbral_nef).sum()),
         "detalle": f"MAE {mae_bajo:.3f} vs {mae_alto:.3f} con N_ef alto"},
        {"criterio": f"3. leverage OLS > {umbral_h:.4f}", "filas_test": int(lejos.sum()),
         "detalle": f"MAE OLS {e_ols[lejos].mean():.3f} vs {e_ols[~lejos].mean():.3f}"},
        {"criterio": "4. nota real 3, 4 u 8", "filas_test": int(extremas.sum()),
         "detalle": f"MAE {mae_extremas:.3f} vs {mae_centrales:.3f} en notas 5-6"},
        {"criterio": "4. desacuerdo entre modelos", "filas_test": len(ev),
         "detalle": f"corr(rango de predicciones, error RF) = {corr_rango:.3f}"},
    ])
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(DIR_TABLAS / "parte2_criterios_resumen.csv", index=False)
    print(resumen.to_string(index=False), "\n")

    tabla_notas = "\n".join(
        f"| {int(nota)} | {int(fila['size'])} | {fila['mean']:.2f} |"
        for nota, fila in por_nota.iterrows()
    )

    criterios_md = f"""# 2.4 Criterios para desconfiar de una predicción

Criterios para avisarle a la bodega cuándo **no** confiar en una predicción de la nota del
panel. Todos salen de los resultados de 2.1–2.3 sobre Wine tinto (320 filas de prueba).

Antes, un hallazgo que cambia la lectura de todo lo demás: el archivo tiene **{n_dup} filas repetidas**,
y **{n_con} de las 320 filas de prueba tienen un gemelo exacto en entrenamiento** (las mismas 11 variables).

---

### 1. Una sola fila de entrenamiento domina el voto: $c(x) > {UMBRAL_C}$

* **Qué mide:** $c(x) = \\max_i w_i^2 / \\sum_j w_j^2$ es la parte de la suma de cuadrados de los
  pesos que se lleva **una sola fila de entrenamiento** (no un árbol ni una hoja). Si es grande,
  falla la condición de Hájek–Šidák y la predicción no es un promedio estable de muchos casos.
* **Evidencia:** en la fila de prueba {int(rf_12["Idx_Test"])}, el bosque tiene $c(x) = {rf_12["c(x)"]:.4f}$.
  Esa fila tiene un gemelo exacto en entrenamiento, que recibe un peso de {rf_12["peso_en_gemelos"]:.3f}
  ({veces_siguiente:.0f} veces el peso del siguiente testigo) y tiene la misma nota. En todo el test, {n_alta_c} filas
  tienen $c(x) > {UMBRAL_C}$, y el {pct_gemelo_alta_c:.0f}% de ellas tiene gemelo.
* **Qué hacer:** revisar quién es esa fila dominante. Si es un gemelo, ver el criterio 2.
  Si no lo es, la predicción depende de un solo vino histórico y es frágil
  (MAE de {mae_alta_c_sin:.2f} en esas {n_alta_c_sin} filas).

### 2. $N_{{ef}}$ bajo: primero preguntar si hay un gemelo

* **Lo esperado era** que poca evidencia significara más error. Los datos dicen lo contrario:
  el quintil de menor $N_{{ef}}$ es el de menor error (ver 2.3).
* **Por qué:** con gemelo exacto el MAE es **{mae_con:.2f}**; sin gemelo es **{mae_sin:.2f}**.
  El bosque (`min_samples_leaf=2`, sin bootstrap) pone casi todo el peso en el gemelo, que tiene
  la misma nota. Un $N_{{ef}}$ bajo en esos casos no indica falta de evidencia: indica que **la
  respuesta se copió del entrenamiento**, una forma de fuga de datos. El error de prueba de
  esas filas exagera qué tan bien funcionaría el modelo con un vino nuevo.
* **Sin gemelos, la relación esperada aparece:** con $N_{{ef}} < {umbral_nef:.0f}$ el MAE es
  {mae_bajo:.2f}, y con $N_{{ef}} \\ge {umbral_nef:.0f}$ baja a {mae_alto:.2f}.
* **Regla práctica:** si el vino ya está en la base histórica, la predicción es una copia de su
  nota anterior (útil, pero no es una predicción). Si no está, confiar más cuando
  $N_{{ef}} \\ge {umbral_nef:.0f}$.

### 3. OLS extrapolando: leverage alto

* **Evidencia:** en las tres consultas, entre el {resumen_neg.min():.0f}% y el {resumen_neg.max():.0f}% de los pesos de OLS son negativos.
  Los cinco pesos más negativos caen en filas muy lejanas: entre el percentil
  {pct_neg.min():.0f} y el {pct_neg.max():.0f} de distancia a la consulta (ver 2.2).
  OLS ajusta un plano, así que usa vinos lejanos en sentido contrario para "inclinarlo".
* **Cómo detectarlo:** el leverage $h(x) = \\tilde x^\\top (\\tilde X^\\top \\tilde X)^{{-1}} \\tilde x$
  es exactamente $\\sum_i w_i(x)^2$ de OLS. Como los pesos suman 1, $N_{{ef}}^{{OLS}} = 1/h(x)$.
  Con la regla usual $h(x) > 2(p+1)/n = {umbral_h:.4f}$, {int(lejos.sum())} filas de prueba quedan
  marcadas, y en ellas el MAE de OLS es {e_ols[lejos].mean():.2f}, contra {e_ols[~lejos].mean():.2f}
  en las demás.
* **Qué hacer:** si $h(x)$ supera el umbral, la consulta está lejos de los datos y la predicción
  lineal es una extrapolación.

### 4. Vinos excepcionales: el voto ponderado no llega a los extremos

* **Evidencia:** el error del bosque por nota real del panel:

| Nota real | Filas | MAE |
|---|---|---|
{tabla_notas}

  En las notas 3, 4 y 8 el MAE es {mae_extremas:.2f}, contra {mae_centrales:.2f} en las notas 5 y 6.
  Las predicciones del bosque solo van de {ev["y_pred"].min():.2f} a {ev["y_pred"].max():.2f}.
* **Por qué:** con pesos positivos que suman 1, la predicción es un promedio de notas vecinas.
  Como casi no hay vinos de 3, 4 u 8, sus vecinos son de 5 y 6 y el promedio se jala al centro.
* **Qué hacer:** no usar el modelo para detectar vinos excepcionales (muy malos o muy buenos).
* **Lo que probamos y no sirvió:** el desacuerdo entre los cinco modelos (rango de sus
  predicciones) no se relaciona con el error del bosque (correlación {corr_rango:.3f}),
  así que no lo usamos como alarma.
"""

    DIR_EXPLICACIONES.mkdir(parents=True, exist_ok=True)
    ruta_md = DIR_EXPLICACIONES / "2_4_criterios_desconfianza.md"
    ruta_md.write_text(criterios_md, encoding="utf-8")
    print(f"Documento generado en:\n  {ruta_md}")


if __name__ == "__main__":
    redactar_criterios_y_validar()