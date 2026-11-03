import numpy as np


def pesos_knn(modelo, X_consulta):
    """Pesos de un KNeighborsRegressor ya ajustado (con weights='uniform').

    Devuelve W de forma (m, n_train), o un vector si hay una sola consulta.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    X_consulta = np.atleast_2d(X_consulta)

    k = modelo.n_neighbors
    n = modelo.n_samples_fit_

    # indices[j] = posiciones de los k vecinos de la consulta j en X_train
    _, indices = modelo.kneighbors(X_consulta)

    W = np.zeros((X_consulta.shape[0], n))
    filas = np.arange(X_consulta.shape[0])[:, None]  
    W[filas, indices] = 1.0 / k
    return W[0] if una_sola else W


def pesos_arbol(modelo, X_train, X_consulta):
    """Pesos de un árbol de regresión ya ajustado.

    Devuelve W de forma (m, n_train), o un vector si hay una sola consulta.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    X_consulta = np.atleast_2d(X_consulta)

    hojas_train = modelo.apply(X_train)        # (n,)  hoja de cada fila de entrenamiento
    hojas_consulta = modelo.apply(X_consulta)  # (m,)  hoja de cada consulta

    misma = hojas_consulta[:, None] == hojas_train[None, :]
    W = misma / misma.sum(axis=1, keepdims=True)
    return W[0] if una_sola else W


if __name__ == "__main__":
    # Comprobación con California
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]
    consultas = X_test[:20]

    print("=== k-NN (k = 30) ===")
    knn = KNeighborsRegressor(n_neighbors=30).fit(X_train, y_train)
    W = pesos_knn(knn, consultas)
    print("Máx |W y - predict|:", np.max(np.abs(W @ y_train - knn.predict(consultas))))
    print("Suma de pesos (mín, máx):", W.sum(axis=1).min(), W.sum(axis=1).max())
    print("Pesos distintos de cero por consulta:", np.unique((W > 0).sum(axis=1)))
    print("Valores distintos de los pesos no nulos:", np.unique(W[W > 0]))

    # Validación con vecinos: los pesos no nulos están justo en los índices de kneighbors
    _, indices = knn.kneighbors(consultas)
    coinciden = all(set(np.flatnonzero(W[j])) == set(indices[j]) for j in range(len(consultas)))
    print("Pesos no nulos = vecinos de kneighbors:", coinciden)

    print("\n=== Árbol (max_depth = 6) ===")
    arbol = DecisionTreeRegressor(max_depth=6, random_state=0).fit(X_train, y_train)
    W = pesos_arbol(arbol, X_train, consultas)
    print("Máx |W y - predict|:", np.max(np.abs(W @ y_train - arbol.predict(consultas))))
    print("Suma de pesos (mín, máx):", W.sum(axis=1).min(), W.sum(axis=1).max())
    print("Número de hojas:", arbol.get_n_leaves())

    # Validación con hojas: el número de pesos no nulos es el tamaño de la hoja
    # que sklearn guarda en tree_.n_node_samples
    hojas = arbol.apply(consultas)
    tam_sklearn = arbol.tree_.n_node_samples[hojas]
    print("Pesos no nulos = tamaño de la hoja:", np.array_equal((W > 0).sum(axis=1), tam_sklearn))
    print("Tamaños de hoja de las 20 consultas:", tam_sklearn)