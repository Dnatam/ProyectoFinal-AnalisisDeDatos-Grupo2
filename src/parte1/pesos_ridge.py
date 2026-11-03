import numpy as np

from src.parte1.pesos_ols import agregar_intercepto

LAMBDA_PARTE1 = 100.0


def pesos_ridge(X_train, X_consulta, lam=LAMBDA_PARTE1):
    """Pesos de ridge con intercepto NO penalizado (equivale a Ridge(alpha=lam)).

    Devuelve W de forma (m, n) (o un vector de largo n si hay una sola consulta);
    la predicción es W @ y_train.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    X_train = np.asarray(X_train, dtype=float)
    X_consulta = np.atleast_2d(np.asarray(X_consulta, dtype=float))
    n, p = X_train.shape

    x_bar = X_train.mean(axis=0)
    Xc = X_train - x_bar                                  # (n, p) centrada

    # A = (Xc^T Xc + lambda I)^{-1} Xc^T, de forma (p, n), resolviendo el sistema
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T)

    W = np.full((X_consulta.shape[0], n), 1.0 / n) + (X_consulta - x_bar) @ A
    return W[0] if una_sola else W


def pesos_ridge_intercepto_penalizado(X_train, X_consulta, lam=LAMBDA_PARTE1):
    """Pesos de ridge si se agrega la columna de unos y se penaliza junto con
    los demás coeficientes. NO es lo que hace sklearn: sirve para el contraste."""
    una_sola = np.asarray(X_consulta).ndim == 1
    Xt = agregar_intercepto(X_train)                      # (n, p+1)
    xt = agregar_intercepto(X_consulta)                   # (m, p+1)

    A = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T)
    W = xt @ A
    return W[0] if una_sola else W


if __name__ == "__main__":
    # Comprobación con California
    from sklearn.linear_model import Ridge
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]
    n = len(y_train)
    lam = LAMBDA_PARTE1

    pred_sklearn = Ridge(alpha=lam).fit(X_train, y_train).predict(X_test[:20])

    print("=== Intercepto libre ===")
    W = pesos_ridge(X_train, X_test[:20], lam)
    print("Máx |W y - predict|:", np.max(np.abs(W @ y_train - pred_sklearn)))
    print("Suma de pesos (mín, máx):", W.sum(axis=1).min(), W.sum(axis=1).max())

    print("\n=== Intercepto penalizado ===")
    Wp = pesos_ridge_intercepto_penalizado(X_train, X_test[:20], lam)
    dif = Wp @ y_train - pred_sklearn
    print("Máx |W y - predict|:", np.max(np.abs(dif)))
    print("Diferencia media (W y - predict):", dif.mean())
    print("Suma de pesos (mín, máx):", Wp.sum(axis=1).min(), Wp.sum(axis=1).max())
    print("n / (n + lambda):", n / (n + lam))
    print("Sesgo esperado hacia cero, -y_bar * lambda/(n+lambda):",
          -y_train.mean() * lam / (n + lam))

    # Sanity check: el modelo penalizado sí es un Ridge sin intercepto sobre [1, X]
    X_tilde_train = agregar_intercepto(X_train)
    X_tilde_test = agregar_intercepto(X_test[:20])
    pred_pen = Ridge(alpha=lam, fit_intercept=False).fit(X_tilde_train, y_train).predict(X_tilde_test)
    print("Coincide con Ridge(fit_intercept=False) sobre [1, X]:",
          np.allclose(Wp @ y_train, pred_pen))