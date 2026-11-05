from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.comunes.datos import cargar_wine
from src.parte1.pesos_bosque import pesos_bosque
from src.parte1.pesos_knn_arbol import pesos_arbol, pesos_knn
from src.parte1.pesos_ols import pesos_ols
from src.parte1.pesos_ridge import pesos_ridge
from src.parte1.verificacion import verificar

# Rutas de la Parte 2
RUTA_MODELOS = ROOT_DIR / "resultados" / "modelos_entrenados.joblib"
DIR_TABLAS = ROOT_DIR / "resultados" / "tablas"
DIR_GRAFICAS = ROOT_DIR / "resultados" / "graficas"
DIR_EXPLICACIONES = ROOT_DIR / "notebooks" / "explicaciones" / "parte 2"

ORDEN_MODELOS = ["OLS", "Ridge", "k-NN", "Árbol", "Random Forest"]


# ---------------------------------------------------------------------
# Métricas de evidencia
# ---------------------------------------------------------------------
def calcular_nef(w: np.ndarray) -> float:
    """Tamaño efectivo de muestra de Kish: N_ef = (sum w_i)^2 / sum w_i^2."""
    w = np.asarray(w, dtype=float)
    suma_cuadrados = np.sum(w**2)
    if suma_cuadrados == 0:
        return 0.0
    return float(np.sum(w) ** 2 / suma_cuadrados)


def calcular_concentracion(w: np.ndarray) -> float:
    """Concentración (condición de Hájek-Šidák): c(x) = max w_i^2 / sum w_j^2.

    Si c(x) es grande, UNA SOLA FILA de entrenamiento domina la predicción.
    """
    w = np.asarray(w, dtype=float)
    suma_cuadrados = np.sum(w**2)
    if suma_cuadrados == 0:
        return 0.0
    return float(np.max(w**2) / suma_cuadrados)


# ---------------------------------------------------------------------
# Duplicados: filas de prueba con un gemelo exacto en entrenamiento
# ---------------------------------------------------------------------
def indices_gemelos(x: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Posiciones en X_train de las filas idénticas a x (en todas las variables)."""
    return np.where(np.all(np.isclose(X_train, x), axis=1))[0]


def marcar_gemelos(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Vector booleano: True si la fila de prueba tiene al menos un gemelo en train."""
    return np.array([len(indices_gemelos(x, X_train)) > 0 for x in X_test])


# ---------------------------------------------------------------------
# Pesos de los cinco modelos (Parte 1) + verificación contra .predict()
# ---------------------------------------------------------------------
def pesos_modelo(nombre: str, modelo, X_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pesos w(x) de cada modelo, usando las funciones de la Parte 1."""
    if nombre == "OLS":
        return pesos_ols(X_train, x)
    if nombre == "Ridge":
        return pesos_ridge(X_train, x, lam=modelo.alpha)
    if nombre == "k-NN":
        return pesos_knn(modelo, x)
    if nombre == "Árbol":
        return pesos_arbol(modelo, X_train, x)
    if nombre == "Random Forest":
        return pesos_bosque(modelo, X_train, x)
    raise ValueError(f"Modelo desconocido: {nombre}")


def pesos_verificados(nombre, modelo, X_train, y_train, x):
    """Devuelve (w, pred) y comprueba que sum_i w_i y_i == modelo.predict(x)."""
    w = pesos_modelo(nombre, modelo, X_train, x)
    pred = float(modelo.predict(np.atleast_2d(x))[0])
    verificar(w, y_train, pred)
    return w, pred


# ---------------------------------------------------------------------
# Testigos
# ---------------------------------------------------------------------
def variables_importantes(modelo_rf, columnas, n=4) -> list:
    """Las n variables con mayor feature_importances_ del bosque."""
    imp = pd.Series(modelo_rf.feature_importances_, index=columnas)
    return imp.sort_values(ascending=False).index[:n].tolist()


def obtener_testigos(
    w: np.ndarray,
    X_train: pd.DataFrame,
    y_train,
    top_n: int = 5,
    variables_clave: list = None,
) -> pd.DataFrame:
    """Las top_n filas de entrenamiento con mayor |w_i|.

    X_train debe venir en unidades ORIGINALES para que la tabla se pueda leer.
    Devuelve: rango, fila_train (posición en X_train), peso_w, y_real y las variables clave.
    """
    w = np.asarray(w, dtype=float)
    y_train = np.asarray(y_train)
    idx_top = np.argsort(-np.abs(w), kind="stable")[:top_n]

    columnas = variables_clave if variables_clave is not None else list(X_train.columns)
    df = X_train.iloc[idx_top][columnas].reset_index(drop=True)
    df.insert(0, "rango", np.arange(1, len(idx_top) + 1))
    df.insert(1, "fila_train", idx_top)
    df.insert(2, "peso_w", w[idx_top])
    df.insert(3, "y_real", y_train[idx_top])
    return df


# ---------------------------------------------------------------------
# Carga de modelos y datos
# ---------------------------------------------------------------------
def cargar_artefactos():
    """Modelos ajustados por cv_modelos.py y datos Wine recargados desde el CSV."""
    if not RUTA_MODELOS.exists():
        raise FileNotFoundError(
            "No existe resultados/modelos_entrenados.joblib. "
            "Primero corre: python -m src.parte2.cv_modelos"
        )
    modelos = joblib.load(RUTA_MODELOS)["modelos"]
    datos = cargar_wine()
    return modelos, datos


# =====================================================================
# Prueba con pesos ficticios
# =====================================================================
if __name__ == "__main__":
    print("--- Probando modulo_evidencia.py con pesos ficticios ---")

    w_uniforme = np.array([0.1] * 10 + [0.0] * 90)
    print("\n[10 pesos iguales a 0.1]")
    print(f"N_ef esperado 10.0 -> {calcular_nef(w_uniforme):.4f}")
    print(f"c(x) esperado 0.10 -> {calcular_concentracion(w_uniforme):.4f}")

    w_concentrado = np.array([0.9, 0.05, 0.05] + [0.0] * 97)
    print("\n[Una fila con 90% del peso]")
    print(f"N_ef -> {calcular_nef(w_concentrado):.4f}")
    print(f"c(x) -> {calcular_concentracion(w_concentrado):.4f}")

    X_dummy = pd.DataFrame({"alcohol": [12.5, 13.0, 11.2, 14.1, 10.8],
                            "pH": [3.2, 3.4, 3.1, 3.5, 3.0]})
    y_dummy = np.array([6, 7, 5, 8, 5])
    w_dummy = np.array([0.05, 0.70, -0.15, 0.08, 0.02])
    print("\n[Top 3 testigos]")
    print(obtener_testigos(w_dummy, X_dummy, y_dummy, top_n=3))

    X_tr = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    print("\n[Gemelos de (1, 2)]:", indices_gemelos(np.array([1.0, 2.0]), X_tr))
    print("\nMódulo probado.")