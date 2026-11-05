import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine

VALORES_K = [1, 2, 3, 5, 10, 30, 50]


def matriz_knn_sklearn(modelo, X_train):
    """S con los vecinos que devuelve scikit-learn (sin forzar la autoinclusión)."""
    n, k = len(X_train), modelo.n_neighbors
    indices = modelo.kneighbors(X_train, return_distance=False)
    S = np.zeros((n, n))
    S[np.arange(n)[:, None], indices] = 1.0 / k
    return S, indices


if __name__ == "__main__":
    datos = cargar_wine()
    X_train, y_train = datos["X_train_std"], datos["y_train"]
    n = len(y_train)

    # Filas repetidas en entrenamiento
    df = pd.DataFrame(X_train)
    en_grupo_repetido = df.duplicated(keep=False)
    copias_extra = df.duplicated().sum()
    df["y"] = y_train
    notas_distintas = (df[en_grupo_repetido]
                       .groupby(list(range(X_train.shape[1])))["y"].nunique() > 1).sum()
    print("=== Filas repetidas en el entrenamiento de Wine ===")
    print(f"Filas que tienen al menos un gemelo exacto: {en_grupo_repetido.sum()}")
    print(f"Copias de más (filas que sobran si se quitan duplicados): {copias_extra}")
    print(f"Grupos de gemelos con notas distintas: {notas_distintas}\n")

    registros = []
    for k in VALORES_K:
        modelo = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
        S, indices = matriz_knn_sklearn(modelo, X_train)

        # S es la matriz real: reproduce .predict() en entrenamiento
        assert np.allclose(S @ y_train, modelo.predict(X_train))

        se_incluye = np.array([i in indices[i] for i in range(n)])
        registros.append({
            "k": k,
            "n_sobre_k": n / k,
            "traza_S_sklearn": np.trace(S),
            "filas_sin_si_misma": int((~se_incluye).sum()),
            "diferencia": n / k - np.trace(S),
        })

    tabla = pd.DataFrame(registros)
    print("=== tr(S) con los vecinos de scikit-learn contra n/k ===")
    print(tabla.round(4).to_string(index=False))
    print("\nCuando una fila no se elige a sí misma, su S_ii es 0 y la traza baja.")
    print("Las predicciones no cambian porque los gemelos tienen la misma nota,")
    print("así que con filas repetidas S no es única: depende del desempate.")

    carpeta = RAIZ / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "parte3_knn_duplicados.csv", index=False)
    print(f"\nTabla guardada en {carpeta / 'parte3_knn_duplicados.csv'}")