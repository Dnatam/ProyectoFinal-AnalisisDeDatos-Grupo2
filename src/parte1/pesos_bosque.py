import inspect

import numpy as np
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
_PIDE_PESOS = "sample_weight" in inspect.signature(_generate_sample_indices).parameters


def _n_bootstrap(n, max_samples):
    if _PIDE_PESOS:
        return _get_n_samples_bootstrap(n, max_samples, None)
    return _get_n_samples_bootstrap(n, max_samples)


def _indices_bootstrap(semilla, n, n_boot):
    if _PIDE_PESOS:
        return _generate_sample_indices(semilla, n, n_boot, None)
    return _generate_sample_indices(semilla, n, n_boot)


def conteos_bootstrap(bosque, n_train):
    """Matriz (T, n) con c_ti = veces que la fila i salió en la muestra del árbol t.

    Reproduce el sorteo que hace scikit-learn al ajustar (usa la semilla guardada en
    cada árbol). Si el bosque no usa bootstrap, todos los conteos son 1.
    """
    T = len(bosque.estimators_)
    if not bosque.bootstrap:
        return np.ones((T, n_train))

    n_boot = _n_bootstrap(n_train, bosque.max_samples)
    conteos = np.empty((T, n_train))
    for t, arbol in enumerate(bosque.estimators_):
        indices = _indices_bootstrap(arbol.random_state, n_train, n_boot)
        conteos[t] = np.bincount(indices, minlength=n_train)
    return conteos


def pesos_bosque(bosque, X_train, X_consulta, conteos=None):
    """Pesos de un random forest ya ajustado.

    conteos : None usa la fórmula del enunciado (cada fila de la hoja cuenta una vez),
              que es exacta solo sin bootstrap. Pasar conteos_bootstrap(...) para la
              versión que también es exacta con bootstrap.

    Devuelve W de forma (m, n_train), o un vector si hay una sola consulta.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    X_consulta = np.atleast_2d(X_consulta)

    hojas_train = bosque.apply(X_train)        # (n, T)
    hojas_consulta = bosque.apply(X_consulta)  # (m, T)
    n, T = hojas_train.shape
    if conteos is None:
        conteos = np.ones((T, n))

    W = np.zeros((X_consulta.shape[0], n))
    for t in range(T):
        # c_ti * 1{x_i comparte hoja con la consulta en el árbol t}
        misma = (hojas_consulta[:, t][:, None] == hojas_train[:, t][None, :]) * conteos[t]
        W += misma / misma.sum(axis=1, keepdims=True)
    W /= T
    return W[0] if una_sola else W


def bosque_parte1(bootstrap=False):
    """El bosque del enunciado (sin ajustar)."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_features=0.33,
        bootstrap=bootstrap, random_state=0, n_jobs=-1,
    )


if __name__ == "__main__":
    # Comprobación con California 
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]
    consultas = X_test[:20]

    print("=== Sin bootstrap ===")
    bosque = bosque_parte1(bootstrap=False).fit(X_train, y_train)
    W = pesos_bosque(bosque, X_train, consultas)
    print("Máx |W y - predict|:", np.max(np.abs(W @ y_train - bosque.predict(consultas))))
    print("Suma de pesos (mín, máx):", W.sum(axis=1).min(), W.sum(axis=1).max())
    print("Pesos distintos de cero por consulta (mín, máx):",
          (W > 0).sum(axis=1).min(), (W > 0).sum(axis=1).max())

    print("\n=== Con bootstrap, misma fórmula ===")
    bosque_b = bosque_parte1(bootstrap=True).fit(X_train, y_train)
    pred_b = bosque_b.predict(consultas)
    W_ingenua = pesos_bosque(bosque_b, X_train, consultas)
    dif = W_ingenua @ y_train - pred_b
    print("Máx |W y - predict|:", np.max(np.abs(dif)))
    print("Diferencia media:", dif.mean())
    print("Suma de pesos (mín, máx):", W_ingenua.sum(axis=1).min(), W_ingenua.sum(axis=1).max())

    print("\n=== Con bootstrap, pesando por los conteos del sorteo ===")
    conteos = conteos_bootstrap(bosque_b, len(y_train))
    W_corr = pesos_bosque(bosque_b, X_train, consultas, conteos=conteos)
    print("Máx |W y - predict|:", np.max(np.abs(W_corr @ y_train - pred_b)))
    print("Suma de pesos (mín, máx):", W_corr.sum(axis=1).min(), W_corr.sum(axis=1).max())
    print("Fracción media de filas fuera de la muestra de cada árbol:",
          (conteos == 0).mean(), " (teórica: (1 - 1/n)^n ≈ 1/e =", round(np.exp(-1), 4), ")")
    print("Máximo de veces que una fila se repite en una muestra:", int(conteos.max()))