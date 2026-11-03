import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from src.comunes.datos import RAIZ

K = 10
VARIANZA_PCA = 0.90


def espacios(datos, varianza=VARIANZA_PCA):
    """Devuelve las versiones de (X_train, X_test) y los dos PCA ajustados.

    Ambos PCA se ajustan solo con el entrenamiento. El principal es sobre las
    estandarizadas; el de las crudas se incluye como contraste, porque ahí la
    varianza total la acaparan las variables de unidades grandes.
    """
    pca = PCA(n_components=varianza).fit(datos["X_train_std"])
    pca_crudo = PCA(n_components=varianza).fit(datos["X_train"])
    pct = int(varianza * 100)
    return {
        "Crudas": (datos["X_train"], datos["X_test"]),
        "Estandarizadas": (datos["X_train_std"], datos["X_test_std"]),
        f"PCA {pct} % (estandarizadas)": (pca.transform(datos["X_train_std"]),
                                          pca.transform(datos["X_test_std"])),
        f"PCA {pct} % (crudas, contraste)": (pca_crudo.transform(datos["X_train"]),
                                             pca_crudo.transform(datos["X_test"])),
    }, pca, pca_crudo


def errores_knn(espacios_X, y_train, y_test, k=K):
    """Ajusta k-NN en cada espacio y reporta el error de prueba."""
    filas = []
    for nombre, (Xtr, Xte) in espacios_X.items():
        pred = KNeighborsRegressor(n_neighbors=k).fit(Xtr, y_train).predict(Xte)
        filas.append({
            "escala": nombre,
            "dimensiones": Xtr.shape[1],
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
        })
    return pd.DataFrame(filas)


def dominancia_distancia(desviaciones, columnas):
    """Participación de cada variable en la distancia euclidiana cruda.

    Para dos filas tomadas al azar, E[(x_j - x'_j)^2] = 2 sigma_j^2, así que la
    variable j aporta, en promedio, la fracción sigma_j^2 / sum_l sigma_l^2 de la
    distancia al cuadrado.
    """
    sigma = np.asarray(desviaciones, dtype=float)
    tabla = pd.DataFrame({
        "variable": columnas,
        "desv_std": sigma,
        "participacion_distancia2": sigma**2 / np.sum(sigma**2),
    }).sort_values("participacion_distancia2", ascending=False, ignore_index=True)
    tabla["acumulada"] = tabla["participacion_distancia2"].cumsum()
    return tabla


if __name__ == "__main__":
    from src.comunes.datos import cargar_wine

    datos = cargar_wine()
    y_train, y_test = datos["y_train"], datos["y_test"]

    esp, pca, pca_crudo = espacios(datos)
    errores = errores_knn(esp, y_train, y_test)

    print(f"=== k-NN (k = {K}) en tres escalas: error de prueba (Wine) ===")
    print(errores.round(4).to_string(index=False))
    print(f"\nPCA: {pca.n_components_} componentes explican "
          f"{pca.explained_variance_ratio_.sum():.1%} de la varianza")
    print("Varianza explicada por componente:", np.round(pca.explained_variance_ratio_, 3))
    print(f"PCA sobre crudas: {pca_crudo.n_components_} componente(s) explican "
          f"{pca_crudo.explained_variance_ratio_.sum():.1%}")
    carga = pd.Series(pca_crudo.components_[0], index=datos["columnas"])
    print("Cargas del primer componente crudo (las 3 mayores en valor absoluto):")
    print(carga.reindex(carga.abs().sort_values(ascending=False).index)[:3].round(3).to_string())
    print("Referencia, predecir siempre la media de train: RMSE =",
          round(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))), 4))

    dom = dominancia_distancia(datos["desviaciones_originales"], datos["columnas"])
    print("\n=== Participación en la distancia al cuadrado, escala cruda ===")
    print(dom.round(4).to_string(index=False))

    carpeta = RAIZ / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    errores.to_csv(carpeta / "4_1_knn_escalas_wine.csv", index=False)
    dom.to_csv(carpeta / "4_1_dominancia_distancia_wine.csv", index=False)
    print(f"\nTablas guardadas en {carpeta}")