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
        "desviaciones_originales": np.std(X_train, axis=0, ddof=0)
    }


# CALIFORNIA HOUSING
def cargar_california():
    datos = fetch_california_housing(as_frame=True)

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