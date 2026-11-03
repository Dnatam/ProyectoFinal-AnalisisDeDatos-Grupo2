import numpy as np


def agregar_intercepto(X):
    """Agrega una columna de unos a la izquierda: X_tilde = [1, X]."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack([np.ones(X.shape[0]), X])


def pesos_ols(X_train, X_consulta):
    """Pesos de OLS para una o varias consultas.

    Parámetros
    ----------
    X_train    : (n, p) matriz de entrenamiento (sin columna de unos).
    X_consulta : (p,) una consulta, o (m, p) varias consultas.

    Devuelve
    --------
    W : (m, n). La fila j tiene los pesos w(x_j); la predicción es W @ y_train.
        Si X_consulta es un solo vector, devuelve un vector de largo n.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    Xt = agregar_intercepto(X_train)          # (n, p+1)
    xt = agregar_intercepto(X_consulta)       # (m, p+1)

    # A = (Xt^T Xt)^{-1} Xt^T, de forma (p+1, n).
    # Se resuelve el sistema (Xt^T Xt) A = Xt^T en lugar de invertir la matriz:
    # da el mismo resultado y es numéricamente más estable.
    A = np.linalg.solve(Xt.T @ Xt, Xt.T)

    W = xt @ A                                
    return W[0] if una_sola else W


def matriz_sombrero(X_train):
    """H = X_tilde (X_tilde^T X_tilde)^{-1} X_tilde^T: los pesos en los puntos de
    entrenamiento (la matriz suavizadora S de OLS). Ojo: es (n, n)."""
    return pesos_ols(X_train, X_train)


if __name__ == "__main__":
    # Comprobación con California (desde la raíz:  python -m src.parte1.pesos_ols)
    from sklearn.linear_model import LinearRegression
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]

    W = pesos_ols(X_train, X_test[:20])
    pred_pesos = W @ y_train
    pred_sklearn = LinearRegression().fit(X_train, y_train).predict(X_test[:20])

    print("Forma de W:", W.shape)
    print("Máx |W y - predict|:", np.max(np.abs(pred_pesos - pred_sklearn)))
    print("Suma de pesos (mín, máx):", W.sum(axis=1).min(), W.sum(axis=1).max())

    # Suma igual a 1 también lejos de los datos (cualquier x):
    x_lejano = np.full(X_train.shape[1], 50.0)
    print("Suma de pesos en un x muy lejano:", pesos_ols(X_train, x_lejano).sum())

    # Pista del enunciado: H 1 = 1 (en una submuestra para no formar H de 16 512 x 16 512)
    sub = X_train[:2000]
    H = matriz_sombrero(sub)
    print("H 1 = 1:", np.allclose(H @ np.ones(len(sub)), 1))