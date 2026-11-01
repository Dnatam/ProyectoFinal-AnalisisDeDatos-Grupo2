import sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Permitir importar los archivos compartidos
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine

# GRADOS DE LIBERTAD DE OLS
def df_ols(X):
    """
    Calcula los grados de libertad de OLS.
    Considera el intercepto.
    """
    X_centrada = X - np.mean(X, axis=0)
    rango = np.linalg.matrix_rank(X_centrada)

    return 1 + rango

# GRADOS DE LIBERTAD DE RIDGE
def df_ridge(X, alpha):
    """
    Calcula los grados de libertad efectivos de Ridge.
    El intercepto no se penaliza.
    """
    X_centrada = X - np.mean(X, axis=0)

    valores_singulares = np.linalg.svd(
        X_centrada,
        compute_uv=False
    )

    cuadrados = valores_singulares ** 2

    return 1 + np.sum(
        cuadrados / (cuadrados + alpha)
    )

# GRADOS DE LIBERTAD DE K-NN
def df_knn(n, k):
    """
    Calcula los grados de libertad efectivos de k-NN
    con pesos uniformes y autoinclusion.
    """
    if not 1 <= k <= n:
        raise ValueError("k debe estar entre 1 y n.")

    return n / k

# GRADOS DE LIBERTAD DE ARBOLES
def df_arbol(X, y, max_depth=None, min_samples_leaf=1):
    """
    Entrena un arbol de regresion y calcula
    sus grados de libertad mediante el numero de hojas.
    """
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )

    modelo.fit(X, y)

    numero_hojas = modelo.get_n_leaves()

    return numero_hojas

# GRADOS DE LIBERTAD DE RANDOM FOREST
def df_random_forest(
    X,
    y,
    n_estimators=200,
    min_samples_leaf=5,
    max_features=0.33
):
    """
    Calcula la traza de la matriz de pesos
    de Random Forest sin bootstrap.

    Cada arbol asigna a una observacion un peso
    de 1 / numero de muestras en su hoja.
    """

    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=False,
        random_state=0,
        n_jobs=-1
    )

    modelo.fit(X, y)

    n = X.shape[0]
    diagonal = np.zeros(n)

    for arbol in modelo.estimators_:
        hojas = arbol.apply(X)

        _, indices, conteos = np.unique(
            hojas,
            return_inverse=True,
            return_counts=True
        )

        diagonal += 1.0 / conteos[indices]

    diagonal /= len(modelo.estimators_)

    return np.sum(diagonal)

# PRUEBA INICIAL
if __name__ == "__main__":

    datos = cargar_wine()
    X_train = datos["X_train_std"]

    print("=== GRADOS DE LIBERTAD EFECTIVOS ===")
    print("Observaciones:", X_train.shape[0])
    print("Variables:", X_train.shape[1])

    print("\nOLS:")
    print("df =", df_ols(X_train))

    print("\nRidge:")
    for alpha in [0.1, 1, 10, 100, 1000]:
        resultado = df_ridge(X_train, alpha)
        print(f"alpha={alpha:<6} | df={resultado:.4f}")
    
    print("\nk-NN:")
    for k in [1, 5, 10, 30, 50, 100]:
        resultado = df_knn(len(X_train), k)
        print(f"k={k:<4} | df={resultado:.4f}")
    
    y_train = datos["y_train"]

    print("\nArboles de regresion:")
    for profundidad in [2, 4, 6, 8, 10, None]:
        resultado = df_arbol(
            X_train,
            y_train,
            max_depth=profundidad
        )

        print(
            f"max_depth={str(profundidad):<5} | "
            f"df={resultado}"
        )

    print("\nArboles con diferentes tamanos de hoja:")
    for minimo_hoja in [1, 5, 10, 20, 50]:
        resultado = df_arbol(
            X_train,
            y_train,
            max_depth=6,
            min_samples_leaf=minimo_hoja
        )

        print(
            f"min_samples_leaf={minimo_hoja:<3} | "
            f"df={resultado}"
        )

    
    print("\nRandom Forest:")

    for minimo_hoja in [1, 5, 10, 20, 50]:
        resultado = df_random_forest(
            X_train,
            y_train,
            n_estimators=200,
            min_samples_leaf=minimo_hoja
        )

        print(
            f"min_samples_leaf={minimo_hoja:<3} | "
            f"df={resultado:.4f}"
        )

    
    # GRAFICAS DE GRADOS DE LIBERTAD EFECTIVOS
    carpeta_graficas = RAIZ / "resultados" / "graficas"
    carpeta_graficas.mkdir(parents=True, exist_ok=True)

    alphas = [0.1, 1, 10, 100, 1000]
    valores_k = [1, 5, 10, 30, 50, 100]
    profundidades = [2, 4, 6, 8, 10]
    tamanios_hoja = [1, 5, 10, 20, 50]

    df_ridge_valores = [
        df_ridge(X_train, alpha)
        for alpha in alphas
    ]

    df_knn_valores = [
        df_knn(len(X_train), k)
        for k in valores_k
    ]

    df_arbol_valores = [
        df_arbol(X_train, y_train, max_depth=p)
        for p in profundidades
    ]

    df_rf_valores = [
        df_random_forest(
            X_train,
            y_train,
            min_samples_leaf=m
        )
        for m in tamanios_hoja
    ]

    fig, ejes = plt.subplots(2, 2, figsize=(12, 9))

    # RIDGE
    ejes[0, 0].plot(
        alphas,
        df_ridge_valores,
        marker="o"
    )
    ejes[0, 0].set_xscale("log")
    ejes[0, 0].set_title("Ridge")
    ejes[0, 0].set_xlabel("Alpha")
    ejes[0, 0].set_ylabel("Grados de libertad efectivos")
    ejes[0, 0].grid(True, alpha=0.3)

    # K-NN
    ejes[0, 1].plot(
        valores_k,
        df_knn_valores,
        marker="o"
    )
    ejes[0, 1].set_title("k-NN")
    ejes[0, 1].set_xlabel("Numero de vecinos (k)")
    ejes[0, 1].set_ylabel("Grados de libertad efectivos")
    ejes[0, 1].grid(True, alpha=0.3)

    # ARBOLES
    ejes[1, 0].plot(
        profundidades,
        df_arbol_valores,
        marker="o"
    )
    ejes[1, 0].set_title("Arbol de regresion")
    ejes[1, 0].set_xlabel("Profundidad maxima")
    ejes[1, 0].set_ylabel("Numero de hojas")
    ejes[1, 0].grid(True, alpha=0.3)

    # RANDOM FOREST
    ejes[1, 1].plot(
        tamanios_hoja,
        df_rf_valores,
        marker="o"
    )
    ejes[1, 1].set_title("Random Forest")
    ejes[1, 1].set_xlabel("Minimo de muestras por hoja")
    ejes[1, 1].set_ylabel("Grados de libertad efectivos")
    ejes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Complejidad de los modelos mediante grados de libertad",
        fontsize=14
    )

    plt.tight_layout()

    ruta = carpeta_graficas / "parte3_grados_libertad.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGrafica guardada en: {ruta}")
    
    # GUARDAR TABLA DE GRADOS DE LIBERTAD
    carpeta_tablas = RAIZ / "resultados" / "tablas"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    registros = [
        {
            "modelo": "OLS",
            "hiperparametro": "ninguno",
            "valor": "N/A",
            "grados_libertad": df_ols(X_train)
        }
    ]

    for alpha in alphas:
        registros.append({
            "modelo": "Ridge",
            "hiperparametro": "alpha",
            "valor": alpha,
            "grados_libertad": df_ridge(X_train, alpha)
        })

    for k in valores_k:
        registros.append({
            "modelo": "k-NN",
            "hiperparametro": "k",
            "valor": k,
            "grados_libertad": df_knn(len(X_train), k)
        })

    for profundidad in profundidades:
        registros.append({
            "modelo": "Arbol",
            "hiperparametro": "max_depth",
            "valor": profundidad,
            "grados_libertad": df_arbol(
                X_train,
                y_train,
                max_depth=profundidad
            )
        })

    for minimo_hoja in tamanios_hoja:
        registros.append({
            "modelo": "Random Forest",
            "hiperparametro": "min_samples_leaf",
            "valor": minimo_hoja,
            "grados_libertad": df_random_forest(
                X_train,
                y_train,
                min_samples_leaf=minimo_hoja
            )
        })

    tabla = pd.DataFrame(registros)

    ruta_tabla = carpeta_tablas / "parte3_grados_libertad.csv"
    tabla.to_csv(ruta_tabla, index=False)

    print(f"\nTabla guardada en: {ruta_tabla}")
