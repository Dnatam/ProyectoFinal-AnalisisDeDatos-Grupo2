import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# PERMITIR IMPORTAR ARCHIVOS COMPARTIDOS
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine
from src.comunes.config import RANDOM_STATE
from src.parte3.grados_libertad import (
    df_ols,
    df_ridge,
    df_knn
)

# CONFIGURACIONES
VALORES_RIDGE = [0.1, 1, 10, 100, 1000, 10000]
VALORES_KNN = [1, 3, 5, 10, 20, 30, 50, 100]
PROFUNDIDADES_ARBOL = [2, 4, 6, 8, 10, 12, None]
MINIMOS_HOJA_ARBOL = [1, 5, 10, 20, 50]
MINIMOS_HOJA_RF = [1, 5, 10, 20, 50]

# REGISTRAR RESULTADOS
def registrar_resultado(
    registros,
    familia,
    configuracion,
    grados_libertad,
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)

    registros.append({
        "modelo": familia,
        "configuracion": configuracion,
        "grados_libertad": float(grados_libertad),
        "mse_train": float(mse_train),
        "mse_test": float(mse_test),
        "brecha_test_train": float(mse_test - mse_train)
    })

    print(
        f"{familia:<15} | "
        f"{configuracion:<26} | "
        f"df={grados_libertad:>8.2f} | "
        f"Train={mse_train:.4f} | "
        f"Test={mse_test:.4f}"
    )

# CALCULAR TRAZA DE RANDOM FOREST
def traza_random_forest(modelo, X_train):
    """
    Calcula tr(S) promediando las contribuciones de las hojas de cada arbol.

    Valido para el bosque utilizado aqui:
    bootstrap=False, con todas las observaciones presentes en cada arbol.
    """
    trazas = []

    for arbol in modelo.estimators_:
        hojas = arbol.apply(X_train)
        cantidad_hojas = len(np.unique(hojas))
        trazas.append(cantidad_hojas)

    return float(np.mean(trazas))

# GENERAR GRAFICA
def graficar_resultados(tabla, ruta):
    colores = {
        "OLS": "black",
        "Ridge": "tab:blue",
        "k-NN": "tab:orange",
        "Arbol": "tab:green",
        "Random Forest": "tab:red"
    }

    plt.figure(figsize=(12, 7))

    for familia, grupo in tabla.groupby("modelo"):
        grupo = grupo.sort_values("grados_libertad")
        color = colores[familia]

        plt.plot(
            grupo["grados_libertad"],
            grupo["mse_train"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{familia} - Train"
        )

        plt.plot(
            grupo["grados_libertad"],
            grupo["mse_test"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{familia} - Test"
        )

    plt.xscale("log")
    plt.xlabel("Complejidad estructural: tr(S)")
    plt.ylabel("Error cuadratico medio (MSE)")
    plt.title("Wine Quality - Error frente a complejidad")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(ruta, dpi=300)
    plt.close()

# EXPERIMENTO PRINCIPAL
if __name__ == "__main__":
    datos = cargar_wine()

    X_train = datos["X_train_std"]
    X_test = datos["X_test_std"]
    y_train = datos["y_train"]
    y_test = datos["y_test"]

    registros = []

    print("=== ERROR FRENTE A COMPLEJIDAD ===")
    print(f"Entrenamiento: {len(y_train)} observaciones")
    print(f"Prueba: {len(y_test)} observaciones")

    # OLS
    print("\nOLS")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    registrar_resultado(
        registros, "OLS", "sin regularizacion",
        df_ols(X_train), modelo,
        X_train, X_test, y_train, y_test
    )

    # RIDGE
    print("\nRidge")

    for alpha in VALORES_RIDGE:
        modelo = Ridge(alpha=alpha)
        modelo.fit(X_train, y_train)

        registrar_resultado(
            registros, "Ridge", f"alpha={alpha}",
            df_ridge(X_train, alpha), modelo,
            X_train, X_test, y_train, y_test
        )

    # K-NN
    print("\nk-NN")

    for k in VALORES_KNN:
        modelo = KNeighborsRegressor(
            n_neighbors=k,
            weights="uniform",
            algorithm="brute"
        )
        modelo.fit(X_train, y_train)

        registrar_resultado(
            registros, "k-NN", f"k={k}",
            df_knn(len(X_train), k), modelo,
            X_train, X_test, y_train, y_test
        )

    # ARBOLES: PROFUNDIDAD
    print("\nArboles - max_depth")

    for profundidad in PROFUNDIDADES_ARBOL:
        modelo = DecisionTreeRegressor(
            max_depth=profundidad,
            random_state=RANDOM_STATE
        )
        modelo.fit(X_train, y_train)

        registrar_resultado(
            registros, "Arbol",
            f"max_depth={profundidad}",
            modelo.get_n_leaves(),
            modelo,
            X_train, X_test, y_train, y_test
        )

    # ARBOLES: TAMANO MINIMO DE HOJA
    print("\nArboles - min_samples_leaf")

    for minimo in MINIMOS_HOJA_ARBOL:
        modelo = DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=minimo,
            random_state=RANDOM_STATE
        )
        modelo.fit(X_train, y_train)

        registrar_resultado(
            registros, "Arbol",
            f"depth=6, leaf={minimo}",
            modelo.get_n_leaves(),
            modelo,
            X_train, X_test, y_train, y_test
        )

    # RANDOM FOREST
    print("\nRandom Forest")

    for minimo in MINIMOS_HOJA_RF:
        modelo = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=minimo,
            max_features=0.33,
            bootstrap=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        modelo.fit(X_train, y_train)

        grados = traza_random_forest(modelo, X_train)

        registrar_resultado(
            registros, "Random Forest",
            f"min_samples_leaf={minimo}",
            grados, modelo,
            X_train, X_test, y_train, y_test
        )

    # GUARDAR TABLA
    carpeta_tablas = RAIZ / "resultados" / "tablas"
    carpeta_graficas = RAIZ / "resultados" / "graficas"

    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    carpeta_graficas.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame(registros)

    ruta_tabla = carpeta_tablas / "parte3_error_complejidad.csv"
    tabla.to_csv(ruta_tabla, index=False)

    # GUARDAR GRAFICA
    ruta_grafica = (
        carpeta_graficas / "parte3_error_complejidad.png"
    )
    graficar_resultados(tabla, ruta_grafica)

    print("\n=== RESUMEN ===")
    print(f"Configuraciones evaluadas: {len(tabla)}")
    print(f"Tabla guardada en: {ruta_tabla}")
    print(f"Grafica guardada en: {ruta_grafica}")
    print("\nExperimento finalizado correctamente.")
