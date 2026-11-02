from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte2.seleccion_consultas import (
    extraer_pesos_knn,
    extraer_pesos_ols,
    extraer_pesos_rf,
)


def ejecutar_analisis_pesos_distancia():
    print(" 2.2 Pesos Negativos en OLS y Peso vs Distancia")

    # Cargar modelos y datos guardados
    ruta_modelos = ROOT_DIR / "resultados" / "modelos_entrenados.joblib"
    if not ruta_modelos.exists():
        raise FileNotFoundError(
            "Primero ejecuta cv_modelos.py y seleccion_consultas.py"
        )

    artefactos = joblib.load(ruta_modelos)
    modelos = artefactos["modelos"]
    datos = artefactos["datos"]

    X_train_std = datos["X_train_std"]
    X_test_std = datos["X_test_std"]

    # Importar los índices de las consultas guardadas
    ruta_csv_consultas = (
        ROOT_DIR / "resultados" / "tablas" / "parte2_consultas_evidencia.csv"
    )
    df_consultas = pd.read_csv(ruta_csv_consultas)

    consultas_unicas = df_consultas[
        ["Consulta", "Idx_Test"]
    ].drop_duplicates()

    print("\n--- 1. Análisis de Pesos Negativos en OLS ---")
    for _, row in consultas_unicas.iterrows():
        nombre_c = row["Consulta"]
        idx_c = int(row["Idx_Test"])
        x_c_std = X_test_std[idx_c]

        w_ols = extraer_pesos_ols(X_train_std, x_c_std)

        n_neg = np.sum(w_ols < 0)
        porcentaje_neg = (n_neg / len(w_ols)) * 100
        min_peso = np.min(w_ols)

        print(f"• {nombre_c} (Índice Test = {idx_c}):")
        print(
            f"  - Total de pesos negativos: {n_neg} de {len(w_ols)} ({porcentaje_neg:.1f}%)"
        )
        print(f"  - Peso más negativo: {min_peso:.6f}")

    # --- 2. Graficar w_i(x) vs Distancia para la Consulta Mediana ---
    row_mediana = consultas_unicas[
        consultas_unicas["Consulta"] == "Nef Mediano"
    ].iloc[0]
    idx_mediana = int(row_mediana["Idx_Test"])
    x_mediana_std = X_test_std[idx_mediana]

    # Calcular distancias euclidianas desde cada x_i de entrenamiento a la consulta
    distancias = np.linalg.norm(X_train_std - x_mediana_std, axis=1)

    # Extraer pesos para OLS, k-NN y Random Forest
    w_ols = extraer_pesos_ols(X_train_std, x_mediana_std)
    w_knn = extraer_pesos_knn(
        modelos["k-NN"], X_train_std, x_mediana_std
    )
    w_rf = extraer_pesos_rf(
        modelos["Random Forest"], X_train_std, x_mediana_std
    )

    # Crear gráfica comparativa
    plt.figure(figsize=(10, 6))
    plt.scatter(
        distancias,
        w_ols,
        alpha=0.4,
        color="crimson",
        label="OLS",
        s=25,
    )
    plt.scatter(
        distancias,
        w_knn,
        alpha=0.6,
        color="dodgerblue",
        label="k-NN (k=29)",
        s=30,
    )
    plt.scatter(
        distancias,
        w_rf,
        alpha=0.6,
        color="forestgreen",
        label="Random Forest",
        s=30,
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.xlabel("Distancia Euclídeana $d(x_i, x_{consulta})$", fontsize=11)
    plt.ylabel("Peso Asignado $w_i(x)$", fontsize=11)
    plt.title(
        f"Comparación de Geometría de Pesos vs. Distancia\n(Consulta Nef Mediano - Test #[{idx_mediana}])",
        fontsize=12,
    )
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # Guardar gráfica en resultados/graficas/
    dir_graficas = ROOT_DIR / "resultados" / "graficas"
    dir_graficas.mkdir(parents=True, exist_ok=True)
    ruta_fig = dir_graficas / "parte2_pesos_vs_distancia.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.close()

    print(
        f"\nGráfica generada y guardada exitosamente en:\n   {ruta_fig}"
    )


if __name__ == "__main__":
    ejecutar_analisis_pesos_distancia()