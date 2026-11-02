from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte2.modulo_evidencia import calcular_nef
from src.parte2.seleccion_consultas import extraer_pesos_rf


def ejecutar_analisis_quintiles():
    print(" 2.3 Quintiles de N_ef y Error Absoluto Medio (MAE)")

    # Cargar artefactos guardados
    ruta_modelos = ROOT_DIR / "resultados" / "modelos_entrenados.joblib"
    if not ruta_modelos.exists():
        raise FileNotFoundError(
            "Primero ejecuta cv_modelos.py para generar los modelos."
        )

    artefactos = joblib.load(ruta_modelos)
    rf = artefactos["modelos"]["Random Forest"]
    datos = artefactos["datos"]

    X_train_std = datos["X_train_std"]
    X_test_std = datos["X_test_std"]
    y_test = np.asarray(datos["y_test"])

    n_test = len(X_test_std)
    nef_list = []
    y_pred_list = []

    print("--> Evaluando N_ef y predicciones de Random Forest en test set...")
    for i in range(n_test):
        w_rf = extraer_pesos_rf(rf, X_train_std, X_test_std[i])
        nef_list.append(calcular_nef(w_rf))
        y_pred_list.append(rf.predict(X_test_std[i].reshape(1, -1))[0])

    y_pred_rf = np.array(y_pred_list)
    nef_rf = np.array(nef_list)
    error_abs = np.abs(y_test - y_pred_rf)

    df_eval = pd.DataFrame(
        {
            "y_real": y_test,
            "y_pred": y_pred_rf,
            "error_abs": error_abs,
            "N_ef": nef_rf,
        }
    )

    # Agrupar en 5 quintiles según N_ef
    labels_quintiles = [
        "Q1 (Menor Evidencia)",
        "Q2",
        "Q3",
        "Q4",
        "Q5 (Mayor Evidencia)",
    ]
    df_eval["Quintil_Nef"] = pd.qcut(
        df_eval["N_ef"], q=5, labels=labels_quintiles
    )

    # Resumen por quintil
    resumen_quintiles = (
        df_eval.groupby("Quintil_Nef", observed=False)
        .agg(
            N_obs=("N_ef", "count"),
            N_ef_min=("N_ef", "min"),
            N_ef_max=("N_ef", "max"),
            N_ef_promedio=("N_ef", "mean"),
            MAE=("error_abs", "mean"),
        )
        .reset_index()
    )

    # Guardar en CSV
    dir_tablas = ROOT_DIR / "resultados" / "tablas"
    dir_tablas.mkdir(parents=True, exist_ok=True)
    ruta_csv = dir_tablas / "parte2_quintiles_nef_mae.csv"
    resumen_quintiles.to_csv(ruta_csv, index=False)

    print(f"\nResultados por quintil guardados en:\n   {ruta_csv}\n")
    print(resumen_quintiles.to_string(index=False))

    return resumen_quintiles


if __name__ == "__main__":
    ejecutar_analisis_quintiles()