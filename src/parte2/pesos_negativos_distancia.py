from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte2.modulo_evidencia import (
    DIR_GRAFICAS,
    DIR_TABLAS,
    cargar_artefactos,
    pesos_verificados,
    variables_importantes,
)

N_EXTREMOS = 5


def cargar_consultas():
    """{nombre: índice de prueba} guardado por seleccion_consultas.py (2.1)."""
    ruta = DIR_TABLAS / "parte2_consultas_evidencia.csv"
    if not ruta.exists():
        raise FileNotFoundError("Primero corre: python -m src.parte2.seleccion_consultas")
    tabla = pd.read_csv(ruta)[["Consulta", "Idx_Test"]].drop_duplicates()
    return dict(zip(tabla["Consulta"], tabla["Idx_Test"].astype(int)))


def ejecutar_analisis_pesos_distancia():
    print("2.2 Pesos negativos en OLS y peso contra distancia\n")

    modelos, datos = cargar_artefactos()
    X_train_std, X_test_std = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]
    columnas = datos["columnas"]
    X_train_orig = pd.DataFrame(datos["X_train"], columns=columnas)
    variables_clave = variables_importantes(modelos["Random Forest"], columnas, n=4)
    consultas = cargar_consultas()

    # --- 1. Pesos negativos de OLS para cada consulta ---
    resumen, extremos = [], []
    for nombre_c, idx_c in consultas.items():
        x_c = X_test_std[idx_c]
        w_ols, _ = pesos_verificados("OLS", modelos["OLS"], X_train_std, y_train, x_c)
        dist = np.linalg.norm(X_train_std - x_c, axis=1)

        negativos = w_ols < 0
        corr_abs = np.corrcoef(dist, np.abs(w_ols))[0, 1]
        resumen.append({
            "Consulta": nombre_c,
            "Idx_Test": idx_c,
            "pesos_negativos": int(negativos.sum()),
            "porcentaje_negativos": round(100 * negativos.mean(), 1),
            "suma_pesos_negativos": round(float(w_ols[negativos].sum()), 4),
            "suma_pesos_positivos": round(float(w_ols[~negativos].sum()), 4),
            "peso_mas_negativo": round(float(w_ols.min()), 5),
            "dist_mediana_negativos": round(float(np.median(dist[negativos])), 3),
            "dist_mediana_positivos": round(float(np.median(dist[~negativos])), 3),
            "corr(dist, |w|)": round(float(corr_abs), 3),
        })

        # Las 5 filas más negativas y, para contrastar, las 5 más positivas
        orden = np.argsort(w_ols, kind="stable")
        for tipo, idx_filas in [("más negativo", orden[:N_EXTREMOS]),
                                ("más positivo", orden[::-1][:N_EXTREMOS])]:
            for rango, i in enumerate(idx_filas, start=1):
                fila = {
                    "Consulta": nombre_c,
                    "Idx_Test": idx_c,
                    "tipo": tipo,
                    "rango": rango,
                    "fila_train": int(i),
                    "peso_w": round(float(w_ols[i]), 5),
                    "y_real": y_train[i],
                    "distancia": round(float(dist[i]), 3),
                    # % de filas de train que están MÁS CERCA de la consulta que esta
                    "percentil_distancia": round(100 * float(np.mean(dist < dist[i])), 1),
                }
                fila.update(X_train_orig.loc[i, variables_clave].round(4).to_dict())
                extremos.append(fila)

    df_resumen = pd.DataFrame(resumen)
    df_extremos = pd.DataFrame(extremos)

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta_resumen = DIR_TABLAS / "parte2_pesos_negativos_ols.csv"
    ruta_extremos = DIR_TABLAS / "parte2_pesos_extremos_ols.csv"
    df_resumen.to_csv(ruta_resumen, index=False)
    df_extremos.to_csv(ruta_extremos, index=False)

    print("--- Resumen de pesos negativos de OLS ---")
    print(df_resumen.to_string(index=False))
    for nombre_c, t in df_extremos.groupby("Consulta", sort=False):
        print(f"\n{nombre_c}: filas con los pesos más negativos y más positivos")
        print(t.drop(columns=["Consulta", "Idx_Test"]).to_string(index=False))

    # --- 2. Gráfica w_i(x) contra distancia (consulta de N_ef mediano) ---
    idx_med = consultas["Nef Mediano"]
    x_med = X_test_std[idx_med]
    dist = np.linalg.norm(X_train_std - x_med, axis=1)

    k = modelos["k-NN"].n_neighbors
    series = [
        ("OLS", "OLS", "crimson", 0.4),
        ("k-NN", f"k-NN (k={k})", "dodgerblue", 0.8),
        ("Random Forest", "Random Forest", "forestgreen", 0.7),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for nombre_m, etiqueta, color, alpha in series:
        w, _ = pesos_verificados(nombre_m, modelos[nombre_m], X_train_std, y_train, x_med)
        mostrar = np.abs(w) > 1e-12  # k-NN y bosque: solo los pesos distintos de cero
        ax.scatter(dist[mostrar], w[mostrar], s=22, alpha=alpha, color=color,
                   label=f"{etiqueta} ({mostrar.sum()} pesos ≠ 0)")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Distancia euclidiana (variables estandarizadas) entre $x_i$ y la consulta")
    ax.set_ylabel("Peso $w_i(x)$")
    ax.set_title(f"Peso contra distancia — consulta de $N_{{ef}}$ mediano (fila de prueba {idx_med})")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    ruta_fig = DIR_GRAFICAS / "parte2_pesos_vs_distancia.png"
    fig.savefig(ruta_fig, dpi=300)
    plt.close(fig)

    print(f"\nTablas guardadas en:\n  {ruta_resumen}\n  {ruta_extremos}")
    print(f"Gráfica guardada en:\n  {ruta_fig}")
    return df_resumen, df_extremos


if __name__ == "__main__":
    ejecutar_analisis_pesos_distancia()