"""Parte 5 — ¿Cuánto cambia la comparación de modelos por los duplicados?

Compara el error de prueba de los cinco modelos en tres escenarios:
  A. Partición original (con filas repetidas), todas las filas de prueba.
  B. Partición original, solo las filas de prueba SIN gemelo en entrenamiento.
  C. Quitando las filas repetidas ANTES de partir (80/20, random_state=0),
     reajustando cada modelo con sus mismos hiperparámetros.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.comunes.config import RANDOM_STATE, TEST_SIZE
from src.parte2.modulo_evidencia import (
    DIR_TABLAS,
    ORDEN_MODELOS,
    cargar_artefactos,
    marcar_gemelos,
)


def ejecutar_comparacion():
    print("Comparación de modelos con y sin filas repetidas\n")

    modelos, datos = cargar_artefactos()
    X_test, y_test = datos["X_test_std"], datos["y_test"]
    gemelo = marcar_gemelos(X_test, datos["X_train_std"])

    # Escenario C: quitar duplicados antes de partir
    df = pd.read_csv(ROOT_DIR / "data" / "wine" / "winequality-red.csv", sep=";")
    df_unico = df.drop_duplicates()
    X = df_unico.drop(columns=["quality"])
    y = df_unico["quality"].to_numpy()
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(Xtr_c)
    Xtr_c, Xte_c = scaler.transform(Xtr_c), scaler.transform(Xte_c)
    print(f"Filas: {len(df)} originales, {len(df_unico)} sin repetir "
          f"({len(Xtr_c)} train / {len(Xte_c)} test)\n")

    filas = []
    for nombre in ORDEN_MODELOS:
        pred = modelos[nombre].predict(X_test)
        pred_c = clone(modelos[nombre]).fit(Xtr_c, ytr_c).predict(Xte_c)
        filas.append({
            "Modelo": nombre,
            "MSE_A_original": mean_squared_error(y_test, pred),
            "MAE_A_original": mean_absolute_error(y_test, pred),
            "MSE_B_sin_gemelo": mean_squared_error(y_test[~gemelo], pred[~gemelo]),
            "MAE_B_sin_gemelo": mean_absolute_error(y_test[~gemelo], pred[~gemelo]),
            "MSE_C_sin_duplicados": mean_squared_error(yte_c, pred_c),
            "MAE_C_sin_duplicados": mean_absolute_error(yte_c, pred_c),
        })

    tabla = pd.DataFrame(filas).round(4)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_TABLAS / "parte5_comparacion_sin_duplicados.csv"
    tabla.to_csv(ruta, index=False)

    print(tabla.to_string(index=False))
    print(f"\nTabla guardada en {ruta}")
    return tabla


if __name__ == "__main__":
    ejecutar_comparacion()