import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# PERMITIR IMPORTAR ARCHIVOS COMPARTIDOS
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine
from src.parte3.grados_libertad import (
    df_ols,
    df_ridge,
    df_knn,
    df_arbol,
    df_random_forest
)

# MATRIZ SUAVIZADORA DE OLS
def matriz_ols(X):
    """
    Construye la matriz suavizadora de OLS
    incluyendo el intercepto.
    """
    n = X.shape[0]

    X_diseno = np.column_stack([
        np.ones(n),
        X
    ])

    return X_diseno @ np.linalg.pinv(X_diseno)


# MATRIZ SUAVIZADORA DE RIDGE
def matriz_ridge(X, alpha):
    """
    Construye la matriz suavizadora de Ridge.
    El intercepto no se penaliza.
    """
    n = X.shape[0]

    X_centrada = X - X.mean(axis=0)

    U, valores, _ = np.linalg.svd(
        X_centrada,
        full_matrices=False
    )

    pesos = valores**2 / (valores**2 + alpha)

    return (
        np.ones((n, n)) / n
        + (U * pesos) @ U.T
    )


# MATRIZ SUAVIZADORA DE K-NN
def matriz_knn(X, k):
    """
    Construye la matriz suavizadora de k-NN
    con pesos uniformes y autoinclusion explicita.

    Cada observacion utiliza su propio valor
    y los k-1 vecinos restantes.
    """
    n = X.shape[0]

    if not 1 <= k <= n:
        raise ValueError("k debe estar entre 1 y n.")

    S = np.zeros((n, n))

    if k == 1:
        np.fill_diagonal(S, 1.0)
        return S

    vecinos = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute"
    )

    vecinos.fit(X)

    indices = vecinos.kneighbors(
        X,
        return_distance=False
    )

    for i in range(n):
        seleccionados = [i]

        for j in indices[i]:
            if j != i and len(seleccionados) < k:
                seleccionados.append(j)

        S[i, seleccionados] = 1.0 / k

    return S


# MATRIZ SUAVIZADORA DE ARBOLES
def matriz_arbol(X, y, max_depth=6, min_samples_leaf=1):
    """
    Construye la matriz suavizadora de un arbol
    utilizando las observaciones de cada hoja.
    """
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )

    modelo.fit(X, y)

    hojas = modelo.apply(X)
    n = X.shape[0]

    S = np.zeros((n, n))

    for hoja in np.unique(hojas):
        indices = np.where(hojas == hoja)[0]
        cantidad = len(indices)

        S[np.ix_(indices, indices)] = 1.0 / cantidad

    return S


# MATRIZ SUAVIZADORA DE RANDOM FOREST
def matriz_random_forest(
    X,
    y,
    n_estimators=200,
    min_samples_leaf=5,
    max_features=0.33
):
    """
    Construye la matriz suavizadora de Random Forest
    promediando las matrices de sus arboles.

    Se utiliza bootstrap=False.
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
    S = np.zeros((n, n))

    for arbol in modelo.estimators_:
        hojas = arbol.apply(X)

        for hoja in np.unique(hojas):
            indices = np.where(hojas == hoja)[0]
            cantidad = len(indices)

            S[np.ix_(indices, indices)] += 1.0 / cantidad

    S /= len(modelo.estimators_)

    return S


# REGISTRAR RESULTADOS
def registrar_resultado(
    registros,
    modelo,
    hiperparametro,
    valor,
    teorico,
    matricial
):
    """
    Almacena los resultados de una verificacion
    para generar posteriormente una tabla CSV.
    """
    registros.append({
        "modelo": modelo,
        "hiperparametro": hiperparametro,
        "valor": valor,
        "df_formula": float(teorico),
        "traza_S": float(matricial),
        "diferencia_absoluta": float(
            abs(teorico - matricial)
        )
    })


# VERIFICACION DE MATRICES
if __name__ == "__main__":

    datos = cargar_wine()

    X_train = datos["X_train_std"]
    y_train = datos["y_train"]

    registros = []

    print("=== VERIFICACION DE MATRICES S ===")

    # OLS
    print("\nOLS")

    S_ols = matriz_ols(X_train)

    teorico = df_ols(X_train)
    matricial = np.trace(S_ols)

    print(f"Formula={teorico:.4f}")
    print(f"Matriz S={matricial:.4f}")

    assert np.isclose(teorico, matricial)

    registrar_resultado(
        registros,
        "OLS",
        "ninguno",
        "N/A",
        teorico,
        matricial
    )

    # RIDGE
    print("\nRidge")

    for alpha in [1, 10, 100, 1000]:
        S_ridge = matriz_ridge(X_train, alpha)

        teorico = df_ridge(X_train, alpha)
        matricial = np.trace(S_ridge)

        print(
            f"alpha={alpha:<5} | "
            f"Formula={teorico:.4f} | "
            f"Matriz={matricial:.4f}"
        )

        assert np.isclose(teorico, matricial)

        registrar_resultado(
            registros,
            "Ridge",
            "alpha",
            alpha,
            teorico,
            matricial
        )

    # K-NN
    print("\nk-NN")

    for k in [1, 5, 10, 30, 50]:
        S_knn = matriz_knn(X_train, k)

        teorico = df_knn(len(X_train), k)
        matricial = np.trace(S_knn)

        print(
            f"k={k:<3} | "
            f"Formula={teorico:.4f} | "
            f"Matriz={matricial:.4f}"
        )

        assert np.isclose(teorico, matricial)

        registrar_resultado(
            registros,
            "k-NN",
            "k",
            k,
            teorico,
            matricial
        )

    # ARBOLES DE REGRESION
    print("\nArboles de regresion")

    for profundidad in [2, 4, 6, 8, 10]:
        S_arbol = matriz_arbol(
            X_train,
            y_train,
            max_depth=profundidad
        )

        teorico = df_arbol(
            X_train,
            y_train,
            max_depth=profundidad
        )

        matricial = np.trace(S_arbol)

        print(
            f"max_depth={profundidad:<3} | "
            f"Hojas={teorico:<4} | "
            f"Matriz={matricial:.4f}"
        )

        assert np.isclose(teorico, matricial)

        registrar_resultado(
            registros,
            "Arbol",
            "max_depth",
            profundidad,
            teorico,
            matricial
        )

    # RANDOM FOREST
    print("\nRandom Forest")

    for minimo_hoja in [1, 5, 10, 20, 50]:
        S_rf = matriz_random_forest(
            X_train,
            y_train,
            n_estimators=200,
            min_samples_leaf=minimo_hoja
        )

        teorico = df_random_forest(
            X_train,
            y_train,
            n_estimators=200,
            min_samples_leaf=minimo_hoja
        )

        matricial = np.trace(S_rf)

        print(
            f"min_samples_leaf={minimo_hoja:<3} | "
            f"Formula={teorico:.4f} | "
            f"Matriz={matricial:.4f}"
        )

        assert np.isclose(teorico, matricial)

        registrar_resultado(
            registros,
            "Random Forest",
            "min_samples_leaf",
            minimo_hoja,
            teorico,
            matricial
        )

    # GUARDAR TABLA DE VERIFICACIONES
    carpeta_tablas = RAIZ / "resultados" / "tablas"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame(registros)

    ruta_tabla = (
        carpeta_tablas / "parte3_verificacion_matrices.csv"
    )

    tabla.to_csv(ruta_tabla, index=False)

    print(f"\nTabla guardada en: {ruta_tabla}")
    print(f"Total de verificaciones: {len(tabla)}")

    print("\nTodas las verificaciones fueron correctas.")
