import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.comunes.datos import RAIZ

K = 10
RUTA_CONSULTAS = RAIZ / "resultados" / "tablas" / "parte2_consultas_evidencia.csv"
VARIABLES_CLAVE = ["total sulfur dioxide", "free sulfur dioxide",
                   "alcohol", "volatile acidity", "sulphates"]


def cargar_consultas(ruta=RUTA_CONSULTAS):
    """Diccionario {nombre de la consulta: índice en X_test} de la Parte 2."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Primero corre: python -m src.parte2.seleccion_consultas"
        )
    tabla = pd.read_csv(ruta)[["Consulta", "Idx_Test"]].drop_duplicates()
    return dict(zip(tabla["Consulta"], tabla["Idx_Test"].astype(int)))


def vecinos(X_train, x, k=K):
    """Índices (en X_train) de los k vecinos más cercanos de x, del más cercano al más lejano."""
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    return nn.kneighbors(np.atleast_2d(x))[1][0]


def comparar_consulta(datos, idx, k=K):
    """Vecinos crudos y estandarizados de la fila de prueba idx y su comparación."""
    v_crudo = vecinos(datos["X_train"], datos["X_test"][idx], k)
    v_std = vecinos(datos["X_train_std"], datos["X_test_std"][idx], k)
    comunes = np.intersect1d(v_crudo, v_std)
    return v_crudo, v_std, comunes


def cercania_por_variable(datos, idx, indices, variables=VARIABLES_CLAVE):
    """Distancia media |x_ij - x_j| entre la consulta y un grupo de vecinos, por variable,
    en desviaciones estándar (para comparar variables con unidades distintas)."""
    cols = [datos["columnas"].index(v) for v in variables]
    Xs = datos["X_train_std"][np.ix_(indices, cols)]
    xs = datos["X_test_std"][idx, cols]
    return pd.Series(np.abs(Xs - xs).mean(axis=0), index=variables)


if __name__ == "__main__":
    from src.comunes.datos import cargar_wine

    datos = cargar_wine()
    y_train, y_test = datos["y_train"], datos["y_test"]
    consultas = cargar_consultas()

    resumen, cercania = [], []
    for nombre, idx in consultas.items():
        v_crudo, v_std, comunes = comparar_consulta(datos, idx)

        print(f"=== {nombre} (fila de prueba {idx}, y real = {y_test[idx]:.0f}) ===")
        print("Vecinos crudos:        ", v_crudo.tolist())
        print("Vecinos estandarizados:", v_std.tolist())
        print(f"Comparten {len(comunes)} de {K}:", comunes.tolist())
        print(f"y de los vecinos crudos:         {y_train[v_crudo].astype(int).tolist()}"
              f"  -> predicción {y_train[v_crudo].mean():.2f}")
        print(f"y de los vecinos estandarizados: {y_train[v_std].astype(int).tolist()}"
              f"  -> predicción {y_train[v_std].mean():.2f}")

        c_crudo = cercania_por_variable(datos, idx, v_crudo)
        c_std = cercania_por_variable(datos, idx, v_std)
        tabla_c = pd.DataFrame({"vecinos_crudos": c_crudo, "vecinos_estandarizados": c_std})
        print("Distancia media a la consulta por variable (en desv. estándar):")
        print(tabla_c.round(3).to_string())
        print()

        resumen.append({
            "consulta": nombre, "idx_test": idx, "y_real": y_test[idx],
            "comunes": len(comunes), "k": K,
            "pred_crudo": y_train[v_crudo].mean(), "pred_estandarizado": y_train[v_std].mean(),
            "vecinos_crudos": " ".join(map(str, v_crudo)),
            "vecinos_estandarizados": " ".join(map(str, v_std)),
        })
        for escala, c in (("crudos", c_crudo), ("estandarizados", c_std)):
            cercania.append({"consulta": nombre, "idx_test": idx, "vecinos": escala, **c.to_dict()})

    resumen = pd.DataFrame(resumen)
    print("=== Resumen ===")
    print(resumen[["consulta", "idx_test", "y_real", "comunes", "pred_crudo",
                   "pred_estandarizado"]].round(2).to_string(index=False))

    carpeta = RAIZ / "resultados" / "tablas"
    resumen.to_csv(carpeta / "4_2_vecinos_comunes_wine.csv", index=False)
    pd.DataFrame(cercania).to_csv(carpeta / "4_2_cercania_por_variable_wine.csv", index=False)
    print(f"\nTablas guardadas en {carpeta}")