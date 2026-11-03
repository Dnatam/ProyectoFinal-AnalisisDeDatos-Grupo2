import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.comunes.datos import RAIZ
from src.parte1.pesos_bosque import pesos_bosque
from src.parte1.pesos_knn_arbol import pesos_arbol, pesos_knn
from src.parte1.pesos_ols import pesos_ols
from src.parte1.pesos_ridge import LAMBDA_PARTE1, pesos_ridge

TOLERANCIA = 1e-8
N_FILAS = 20
FILA_TABLA = 7


def verificar(w, y_train, pred, tol=TOLERANCIA):
    """Comprueba que el voto ponderado reproduce la predicción del modelo.

    Lanza AssertionError si |sum_i w_i y_i - pred| >= tol. Devuelve la diferencia.
    """
    diferencia = abs(np.dot(w, y_train) - pred)
    assert diferencia < tol, (
        f"Los pesos no reproducen la predicción: |w·y - pred| = {diferencia:.3e} >= {tol:.0e}"
    )
    return diferencia


def ajustar_modelos(X_train, y_train, random_state=0):
    """Ajusta los cinco modelos de la Parte 1 y devuelve, para cada uno, el modelo y
    una función que calcula sus pesos para un conjunto de consultas."""
    ols = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=LAMBDA_PARTE1).fit(X_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=30).fit(X_train, y_train)
    arbol = DecisionTreeRegressor(max_depth=6, random_state=random_state).fit(X_train, y_train)
    bosque = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_features=0.33,
        bootstrap=False, random_state=random_state, n_jobs=-1,
    ).fit(X_train, y_train)

    return {
        "OLS": (ols, lambda Xq: pesos_ols(X_train, Xq)),
        "Ridge": (ridge, lambda Xq: pesos_ridge(X_train, Xq, LAMBDA_PARTE1)),
        "k-NN": (knn, lambda Xq: pesos_knn(knn, Xq)),
        "Arbol": (arbol, lambda Xq: pesos_arbol(arbol, X_train, Xq)),
        "Random Forest": (bosque, lambda Xq: pesos_bosque(bosque, X_train, Xq)),
    }


def verificar_modelos(modelos, X_test, y_train, n_filas=N_FILAS):
    """Aplica verificar() a las primeras n_filas de prueba con cada modelo.

    Devuelve un DataFrame con la diferencia máxima por modelo y un diccionario con
    las matrices de pesos (para reutilizarlas en la tabla)."""
    consultas = X_test[:n_filas]
    filas, pesos = [], {}
    for nombre, (modelo, funcion_pesos) in modelos.items():
        W = funcion_pesos(consultas)
        pred = modelo.predict(consultas)
        difs = [verificar(W[j], y_train, pred[j]) for j in range(n_filas)]
        pesos[nombre] = (W, pred)
        filas.append({"modelo": nombre, "filas_verificadas": n_filas,
                      "max_diferencia": max(difs)})
    return pd.DataFrame(filas), pesos


def tabla_fila(pesos, fila=FILA_TABLA):
    """Tabla de la fila de prueba `fila`: predicción, suma de pesos,
    número de pesos distintos de cero y número de pesos negativos."""
    filas = []
    for nombre, (W, pred) in pesos.items():
        w = W[fila]
        filas.append({
            "modelo": nombre,
            "prediccion": pred[fila],
            "suma_pesos": w.sum(),
            "pesos_no_nulos": int(np.count_nonzero(w)),
            "pesos_negativos": int((w < 0).sum()),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train, y_test = datos["y_train"], datos["y_test"]

    modelos = ajustar_modelos(X_train, y_train)
    resumen, pesos = verificar_modelos(modelos, X_test, y_train)

    print(f"=== Verificación: primeras {N_FILAS} filas de prueba, tolerancia {TOLERANCIA:.0e} ===")
    print(resumen.to_string(index=False))
    print(f"\nLas {N_FILAS * len(modelos)} verificaciones pasaron.")

    tabla = tabla_fila(pesos)
    print(f"\n=== Fila de prueba {FILA_TABLA} (y real = {y_test[FILA_TABLA]:.4f}, n_train = {len(y_train)}) ===")
    print(tabla.to_string(index=False))

    carpeta = RAIZ / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(carpeta / "1_5_verificacion_california.csv", index=False)
    tabla.to_csv(carpeta / f"1_5_tabla_fila{FILA_TABLA}_california.csv", index=False)
    print(f"\nTablas guardadas en {carpeta}")