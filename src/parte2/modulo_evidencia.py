import numpy as np
import pandas as pd


def calcular_nef(w: np.ndarray) -> float:
    """
    Calcula el Tamaño Efectivo de Muestra de Kish (N_ef).
    Formula: N_ef(x) = (sum(w_i)^2) / sum(w_i^2)
    """
    w = np.asarray(w, dtype=float)
    suma_pesos = np.sum(w)
    suma_cuadrados = np.sum(w**2)

    if suma_cuadrados == 0:
        return 0.0

    return float((suma_pesos**2) / suma_cuadrados)


def calcular_concentracion(w: np.ndarray) -> float:
    """
    Calcula la Concentración de Hájek-Šidák c(x).
    Formula: c(x) = max(w_i^2) / sum(w_j^2)
    """
    w = np.asarray(w, dtype=float)
    suma_cuadrados = np.sum(w**2)

    if suma_cuadrados == 0:
        return 0.0

    return float(np.max(w**2) / suma_cuadrados)


def obtener_testigos(
    w: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 5,
    variables_clave: list = None,
) -> pd.DataFrame:
    """
    Identifica las filas de entrenamiento con mayor peso absoluto |w_i|.

    Retorna un DataFrame con:
    - Índice original de la fila de entrenamiento
    - Peso w_i recibido
    - Valor y_real de esa fila
    - Las variables más importantes/clave pasadas en 'variables_clave'
    """
    w = np.asarray(w, dtype=float)
    idx_top = np.argsort(np.abs(w))[-top_n:][::-1]

    # Extraer subconjunto de características
    if variables_clave is not None:
        columnas_mostrar = [c for c in variables_clave if c in X_train.columns]
        df_sub = X_train.iloc[idx_top][columnas_mostrar].copy()
    else:
        df_sub = X_train.iloc[idx_top].copy()

    # Agregar peso e y real
    df_sub.insert(0, "peso_w", w[idx_top])
    df_sub.insert(1, "y_real", y_train.iloc[idx_top].values)

    return df_sub


def generar_tabla_resumen_consulta(
    nombre_consulta: str,
    y_pred: float,
    nef: float,
    concentracion: float,
    df_testigos: pd.DataFrame,
) -> dict:
    """
    Formatea la ficha de evidencia para una consulta según los requerimientos de 2.1.
    """
    return {
        "consulta": nombre_consulta,
        "y_prediccion": y_pred,
        "N_ef": nef,
        "concentracion_c": concentracion,
        "testigos": df_testigos,
    }


# =====================================================================
# Bloque de prueba con pesos ficticios (se ejecuta al correr este script)
# =====================================================================
if __name__ == "__main__":
    print("--- Probando modulo_evidencia.py con pesos ficticios ---")

    # 1. Pesos uniformes (Ej. k-NN con k=10)
    w_uniforme = np.array([0.1] * 10 + [0.0] * 90)
    print(f"\n[Pesos Uniformes (10 vecinos al 10%)]")
    print(f"N_ef esperado: 10.0  -> Calculado: {calcular_nef(w_uniforme):.4f}")
    print(
        f"c(x) esperado: 0.10  -> Calculado: {calcular_concentracion(w_uniforme):.4f}"
    )

    # 2. Pesos concentrados (Una fila domina)
    w_concentrado = np.array([0.9, 0.05, 0.05] + [0.0] * 97)
    print(f"\n[Pesos Concentrados (Un dato tiene el 90% del peso)]")
    print(f"N_ef Calculado: {calcular_nef(w_concentrado):.4f}")
    print(f"c(x) Calculado: {calcular_concentracion(w_concentrado):.4f}")

    # 3. Prueba de extracción de testigos
    np.random.seed(0)
    X_dummy = pd.DataFrame(
        {
            "alcohol": [12.5, 13.0, 11.2, 14.1, 10.8],
            "pH": [3.2, 3.4, 3.1, 3.5, 3.0],
            "sulphates": [0.6, 0.8, 0.5, 0.9, 0.4],
        }
    )
    y_dummy = pd.Series([6, 7, 5, 8, 5])
    w_dummy = np.array([0.05, 0.70, -0.15, 0.08, 0.02])

    print("\n[Prueba de extracción de Top 3 Testigos]")
    testigos_df = obtener_testigos(
        w_dummy, X_dummy, y_dummy, top_n=3, variables_clave=["alcohol", "pH"]
    )
    print(testigos_df)
    print("\n✅ Módulo probado exitosamente.")