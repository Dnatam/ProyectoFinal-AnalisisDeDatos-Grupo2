import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# RUTAS E IMPORTACIONES COMPARTIDAS
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine
from src.comunes.config import RANDOM_STATE

TABLAS = RAIZ / "resultados" / "tablas"
GRAFICAS = RAIZ / "resultados" / "graficas"

TABLAS.mkdir(parents=True, exist_ok=True)
GRAFICAS.mkdir(parents=True, exist_ok=True)

RUTA_TABLA = TABLAS / "parte3_comparacion_oob.csv"
RUTA_GRAFICA = GRAFICAS / "parte3_comparacion_oob.png"

# CONFIGURACIONES DEL EXPERIMENTO
VALORES_HOJA = [1, 2, 5, 10, 20, 50]
N_ARBOLES = 300
MAX_FEATURES = 0.33

# ENTRENAMIENTO Y EVALUACION
def evaluar_bosques(X_train, X_test, y_train, y_test):
    resultados = []

    for minimo_hoja in VALORES_HOJA:
        print(
            f"\nEntrenando Random Forest: "
            f"min_samples_leaf={minimo_hoja}"
        )

        modelo = RandomForestRegressor(
            n_estimators=N_ARBOLES,
            min_samples_leaf=minimo_hoja,
            max_features=MAX_FEATURES,
            bootstrap=True,
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        modelo.fit(X_train, y_train)

        # PREDICCIONES DE ENTRENAMIENTO
        pred_train = modelo.predict(X_train)

        # PREDICCIONES DE PRUEBA
        pred_test = modelo.predict(X_test)

        # PREDICCIONES OUT-OF-BAG
        pred_oob = modelo.oob_prediction_

        if not np.all(np.isfinite(pred_oob)):
            raise ValueError(
                "Se encontraron predicciones OOB no validas."
            )

        mse_train = mean_squared_error(
            y_train, pred_train
        )

        mse_oob = mean_squared_error(
            y_train, pred_oob
        )

        mse_test = mean_squared_error(
            y_test, pred_test
        )

        resultados.append({
            "min_samples_leaf": minimo_hoja,
            "n_estimators": N_ARBOLES,
            "max_features": MAX_FEATURES,
            "mse_train": mse_train,
            "mse_oob": mse_oob,
            "mse_test": mse_test,
            "brecha_oob_train": mse_oob - mse_train,
            "brecha_test_train": mse_test - mse_train,
            "diferencia_oob_test": mse_oob - mse_test,
            "r2_oob": modelo.oob_score_
        })

        print(f"  MSE train: {mse_train:.6f}")
        print(f"  MSE OOB:   {mse_oob:.6f}")
        print(f"  MSE test:  {mse_test:.6f}")

    return pd.DataFrame(resultados)

# GRAFICA DE ERRORES
def graficar_errores(tabla):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tabla))

    ax.plot(
        x,
        tabla["mse_train"],
        marker="o",
        linewidth=2,
        label="MSE entrenamiento",
        color="#3274A1"
    )

    ax.plot(
        x,
        tabla["mse_oob"],
        marker="s",
        linewidth=2,
        label="MSE OOB",
        color="#E1812C"
    )

    ax.plot(
        x,
        tabla["mse_test"],
        marker="^",
        linewidth=2,
        label="MSE prueba",
        color="#55A868"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        tabla["min_samples_leaf"].astype(str)
    )

    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("Error cuadratico medio (MSE)")
    ax.set_title(
        "Random Forest: comparacion de errores train, OOB y test"
    )

    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "300 arboles, bootstrap=True, max_features=0.33. "
        "Mismo conjunto de prueba para todas las configuraciones.",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(RUTA_GRAFICA, dpi=300)
    plt.close()

# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=== RANDOM FOREST: TRAIN / OOB / TEST ===")

    datos = cargar_wine()

    X_train = datos["X_train_std"]
    X_test = datos["X_test_std"]
    y_train = datos["y_train"]
    y_test = datos["y_test"]

    print(f"\nObservaciones train: {len(y_train)}")
    print(f"Observaciones test:  {len(y_test)}")
    print(f"Cantidad de arboles: {N_ARBOLES}")

    tabla = evaluar_bosques(
        X_train,
        X_test,
        y_train,
        y_test
    )

    tabla.to_csv(RUTA_TABLA, index=False)

    graficar_errores(tabla)

    print("\n=== RESUMEN ===")

    columnas = [
        "min_samples_leaf",
        "mse_train",
        "mse_oob",
        "mse_test",
        "brecha_oob_train",
        "brecha_test_train"
    ]

    print(
        tabla[columnas].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(RUTA_TABLA)
    print(RUTA_GRAFICA)

    print("\nExperimento finalizado correctamente.")