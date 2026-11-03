from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.comunes.config import RANDOM_STATE, TEST_SIZE


# RUTAS DEL PROYECTO
RAIZ = Path(__file__).resolve().parents[2]
CARPETA_DATOS = RAIZ / "data"


# DIVISION Y ESTANDARIZACION
def preparar_datos(X, y):
    """
    Divide los datos en entrenamiento y prueba.
    Conserva las variables originales y estandarizadas.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return {
        "X_train": np.asarray(X_train),
        "X_test": np.asarray(X_test),
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "scaler": scaler,
        "desviaciones_originales": np.std(X_train, axis=0, ddof=0),
        # Indices originales de cada fila, para verificar que todos usan la misma particion
        "indices_train": np.asarray(X_train.index),
        "indices_test": np.asarray(X_test.index),
    }


# CALIFORNIA HOUSING
def cargar_california():
    # data_home: guarda la descarga en data/california/ y la reutiliza despues
    datos = fetch_california_housing(
        data_home=str(CARPETA_DATOS / "california"),
        as_frame=True
    )

    X = datos.data
    y = datos.target

    resultado = preparar_datos(X, y)
    resultado["columnas"] = X.columns.tolist()

    return resultado


# WINE QUALITY RED
def cargar_wine():
    ruta = CARPETA_DATOS / "wine" / "winequality-red.csv"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ruta}"
        )

    df = pd.read_csv(ruta, sep=";")

    X = df.drop(columns=["quality"])
    y = df["quality"]

    resultado = preparar_datos(X, y)
    resultado["columnas"] = X.columns.tolist()

    return resultado


# REPORTE DE LA PREPARACION (0.4)
def tabla_desviaciones(resultado):
    """Tabla con la desviacion estandar de cada variable antes de estandarizar (train)."""
    return pd.DataFrame({
        "variable": resultado["columnas"],
        "desv_std_train": np.asarray(resultado["desviaciones_originales"], dtype=float),
    })


def resumen(resultado, nombre=""):
    """Imprime tamanos, comprobaciones del escalado y desviaciones originales."""
    print(f"=== {nombre} ===")
    print(f"train: {resultado['X_train_std'].shape}   test: {resultado['X_test_std'].shape}")
    print("media de X_train_std ~ 0:", np.allclose(resultado["X_train_std"].mean(axis=0), 0))
    print("desv. de X_train_std ~ 1:", np.allclose(resultado["X_train_std"].std(axis=0), 1))
    print("\nDesviacion estandar antes de estandarizar (train):")
    print(tabla_desviaciones(resultado).round(4).to_string(index=False))
    print()


def guardar_particiones(resultado, nombre):
    """Guarda en data/<nombre>/particiones/ las particiones crudas con su indice
    original y la tabla de desviaciones."""
    carpeta = CARPETA_DATOS / nombre / "particiones"
    carpeta.mkdir(parents=True, exist_ok=True)

    for parte in ("train", "test"):
        df = pd.DataFrame(resultado[f"X_{parte}"], columns=resultado["columnas"])
        df.insert(0, "indice_original", resultado[f"indices_{parte}"])
        df["y"] = resultado[f"y_{parte}"]
        df.to_csv(carpeta / f"{nombre}_{parte}.csv", index=False)

    tabla_desviaciones(resultado).to_csv(carpeta / f"{nombre}_desviaciones.csv", index=False)


if __name__ == "__main__":
    # Ejecutar desde la raiz del repo:  python -m src.comunes.datos
    wine = cargar_wine()
    resumen(wine, "Wine Quality (tinto)")
    guardar_particiones(wine, "wine")

    california = cargar_california()
    resumen(california, "California Housing")
    guardar_particiones(california, "california")