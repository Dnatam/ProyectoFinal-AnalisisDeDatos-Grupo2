from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte1.pesos_bosque import pesos_bosque
from src.parte2.modulo_evidencia import (
    DIR_TABLAS,
    calcular_concentracion,
    calcular_nef,
    cargar_artefactos,
    marcar_gemelos,
)

ETIQUETAS = ["Q1 (menor evidencia)", "Q2", "Q3", "Q4", "Q5 (mayor evidencia)"]


def resumen_por_quintil(df, columna_quintil):
    return (
        df.groupby(columna_quintil, observed=False)
        .agg(
            N_obs=("N_ef", "size"),
            N_ef_min=("N_ef", "min"),
            N_ef_max=("N_ef", "max"),
            con_gemelo=("tiene_gemelo", "sum"),
            dist_vecino_mediana=("dist_vecino", "median"),
            c_mediana=("c_x", "median"),
            MAE=("error_abs", "mean"),
        )
        .round(4)
        .reset_index()
    )


def ejecutar_analisis_quintiles():
    print("2.3 Quintiles de N_ef y error absoluto medio (MAE)\n")

    modelos, datos = cargar_artefactos()
    rf = modelos["Random Forest"]
    X_train_std, X_test_std = datos["X_train_std"], datos["X_test_std"]
    y_train, y_test = datos["y_train"], datos["y_test"]

    # 1. Pesos del bosque para todo el test, verificados contra .predict()
    W = pesos_bosque(rf, X_train_std, X_test_std)
    y_pred = rf.predict(X_test_std)
    diferencia = np.max(np.abs(W @ y_train - y_pred))
    assert diferencia < 1e-8, f"Los pesos no reproducen .predict(): {diferencia:.2e}"

    # Distancia de cada fila de prueba a su fila de train más cercana
    dist_vecino = np.array([np.min(np.linalg.norm(X_train_std - x, axis=1)) for x in X_test_std])

    df_eval = pd.DataFrame({
        "Idx_Test": np.arange(len(y_test)),
        "y_real": y_test,
        "y_pred": y_pred,
        "error_abs": np.abs(y_test - y_pred),
        "N_ef": [calcular_nef(w) for w in W],
        "c_x": [calcular_concentracion(w) for w in W],
        "tiene_gemelo": marcar_gemelos(X_test_std, X_train_std),
        "dist_vecino": dist_vecino,
    })
    df_eval["Quintil_Nef"] = pd.qcut(df_eval["N_ef"], q=5, labels=ETIQUETAS)

    # 2. Cuántos duplicados hay
    df_crudo = pd.read_csv(ROOT_DIR / "data" / "wine" / "winequality-red.csv", sep=";")
    n_dup = int(df_crudo.duplicated().sum())
    n_gem = int(df_eval["tiene_gemelo"].sum())
    print(f"Filas repetidas en el archivo completo: {n_dup} de {len(df_crudo)}")
    print(f"Filas de prueba con gemelo exacto en entrenamiento: {n_gem} de {len(df_eval)}\n")

    # 3. Quintiles con todas las filas
    tabla_todas = resumen_por_quintil(df_eval, "Quintil_Nef")
    print("--- Quintiles con todas las filas de prueba ---")
    print(tabla_todas.to_string(index=False))

    mae_con = df_eval.loc[df_eval["tiene_gemelo"], "error_abs"].mean()
    mae_sin = df_eval.loc[~df_eval["tiene_gemelo"], "error_abs"].mean()
    print(f"\nMAE con gemelo en train: {mae_con:.4f} | sin gemelo: {mae_sin:.4f}")

    # 4. Quintiles recalculados solo con las filas SIN gemelo
    sin = df_eval[~df_eval["tiene_gemelo"]].copy()
    sin["Quintil_sin_gemelo"] = pd.qcut(sin["N_ef"], q=5, labels=ETIQUETAS)
    tabla_sin = resumen_por_quintil(sin, "Quintil_sin_gemelo")
    print("\n--- Quintiles solo con filas SIN gemelo en train ---")
    print(tabla_sin.to_string(index=False))

    # 5. ¿Qué notas hay en cada quintil? (ayuda a explicar el MAE bajo de Q5)
    dist_y = pd.crosstab(df_eval["Quintil_Nef"], df_eval["y_real"])
    print("\n--- Notas reales (y) por quintil, todas las filas ---")
    print(dist_y.to_string())

    # 6. Guardar
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    rutas = {
        "parte2_quintiles_nef_mae.csv": tabla_todas,
        "parte2_quintiles_sin_gemelo.csv": tabla_sin,
        "parte2_quintiles_notas.csv": dist_y.reset_index(),
        "parte2_evaluacion_test.csv": df_eval.round(5),
    }
    for nombre, tabla in rutas.items():
        tabla.to_csv(DIR_TABLAS / nombre, index=False)
    pd.DataFrame([
        {"grupo": "con gemelo", "N_obs": n_gem, "MAE": round(mae_con, 4)},
        {"grupo": "sin gemelo", "N_obs": len(df_eval) - n_gem, "MAE": round(mae_sin, 4)},
    ]).to_csv(DIR_TABLAS / "parte2_mae_gemelos.csv", index=False)

    print("\nTablas guardadas en resultados/tablas/:")
    for nombre in list(rutas) + ["parte2_mae_gemelos.csv"]:
        print(f"  {nombre}")
    return tabla_todas, tabla_sin


if __name__ == "__main__":
    ejecutar_analisis_quintiles()