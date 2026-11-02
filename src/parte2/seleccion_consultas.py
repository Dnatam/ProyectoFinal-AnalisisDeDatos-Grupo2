from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte2.modulo_evidencia import (
    calcular_concentracion,
    calcular_nef,
    obtener_testigos,
)


def extraer_pesos_rf(model_rf, X_train_std, x_consulta_std):
    """Calcula los pesos w_i(x) para Random Forest (sin bootstrap)."""
    n_train = len(X_train_std)
    w = np.zeros(n_train)
    n_estimators = len(model_rf.estimators_)

    hojas_train = model_rf.apply(X_train_std)
    hoja_consulta = model_rf.apply(x_consulta_std.reshape(1, -1))[0]

    for t in range(n_estimators):
        hoja_target = hoja_consulta[t]
        mismos_nodos = hojas_train[:, t] == hoja_target
        n_hoja = np.sum(mismos_nodos)

        if n_hoja > 0:
            w[mismos_nodos] += 1.0 / (n_estimators * n_hoja)

    return w


def extraer_pesos_knn(model_knn, X_train_std, x_consulta_std):
    """Calcula pesos w_i(x) para k-NN (1/k en los k vecinos)."""
    n_train = len(X_train_std)
    w = np.zeros(n_train)
    k = model_knn.n_neighbors
    distances, indices = model_knn.kneighbors(x_consulta_std.reshape(1, -1))
    w[indices[0]] = 1.0 / k
    return w


def extraer_pesos_arbol(model_tree, X_train_std, x_consulta_std):
    """Calcula pesos w_i(x) para Árbol de Decisión."""
    n_train = len(X_train_std)
    w = np.zeros(n_train)

    hojas_train = model_tree.apply(X_train_std)
    hoja_consulta = model_tree.apply(x_consulta_std.reshape(1, -1))[0]

    mismos_nodos = hojas_train == hoja_consulta
    n_hoja = np.sum(mismos_nodos)

    if n_hoja > 0:
        w[mismos_nodos] = 1.0 / n_hoja
    return w


def extraer_pesos_ols(X_train_std, x_consulta_std):
    """w(x)^T = x_tilde^T (X_tilde^T X_tilde)^(-1) X_tilde^T con columna de unos."""
    n_train = len(X_train_std)
    X_tilde = np.hstack([np.ones((n_train, 1)), X_train_std])
    x_tilde = np.hstack([1.0, x_consulta_std])

    inv_xtx = np.linalg.pinv(X_tilde.T @ X_tilde)
    w = x_tilde @ inv_xtx @ X_tilde.T
    return w


def extraer_pesos_ridge(model_ridge, X_train_std, x_consulta_std):
    """w(x)^T = 1/n * 1^T + x^T (X_std^T X_std + lambda I)^(-1) X_std^T."""
    n_train = len(X_train_std)
    lam = model_ridge.alpha

    inv_term = np.linalg.pinv(
        X_train_std.T @ X_train_std + lam * np.eye(X_train_std.shape[1])
    )
    w_centered = x_consulta_std @ inv_term @ X_train_std.T
    w = (1.0 / n_train) + w_centered
    return w


def ejecutar_seleccion_consultas():
    print("==================================================")
    print(" 2.1 Selección de Consultas y Ficha de Evidencia")
    print("==================================================")

    ruta_modelos = ROOT_DIR / "resultados" / "modelos_entrenados.joblib"
    if not ruta_modelos.exists():
        raise FileNotFoundError(
            "Primero ejecuta cv_modelos.py para generar los modelos."
        )

    artefactos = joblib.load(ruta_modelos)
    modelos = artefactos["modelos"]
    datos = artefactos["datos"]

    X_train_std = datos["X_train_std"]
    X_test_std = datos["X_test_std"]
    y_train = datos["y_train"]
    y_test = datos["y_test"]
    columnas = datos["columnas"]

    X_train_df = pd.DataFrame(X_train_std, columns=columnas)

    # 1. Calcular N_ef de Random Forest en el conjunto de prueba
    rf = modelos["Random Forest"]
    n_test = len(X_test_std)
    nef_rf_list = []
    pesos_rf_test = []

    print("--> Calculando N_ef (Random Forest) en todo el conjunto de prueba...")
    for idx in range(n_test):
        w_rf = extraer_pesos_rf(rf, X_train_std, X_test_std[idx])
        pesos_rf_test.append(w_rf)
        nef_rf_list.append(calcular_nef(w_rf))

    nef_rf = np.array(nef_rf_list)

    # 2. Selección de las 3 consultas
    idx_mediano = int(np.argsort(nef_rf)[len(nef_rf) // 2])
    idx_minimo = int(np.argmin(nef_rf))
    conc_rf_list = [calcular_concentracion(w) for w in pesos_rf_test]
    idx_interes = int(np.argmax(conc_rf_list))

    consultas = {
        "Nef Mediano": idx_mediano,
        "Nef Mínimo": idx_minimo,
        "Interés (Máx Concentración)": idx_interes,
    }

    print("\n--- Consultas Seleccionadas ---")
    for nombre, idx in consultas.items():
        print(
            f"• {nombre}: Índice Test = {idx} | N_ef RF = {nef_rf[idx]:.2f} | y_real = {y_test[idx]}"
        )

    # 3. Ficha de Evidencia
    resumen_filas = []
    variables_clave = ["alcohol", "volatile acidity", "sulphates", "citric acid"]

    for nombre_c, idx_c in consultas.items():
        x_c_std = X_test_std[idx_c]
        y_c_real = y_test[idx_c]

        for nombre_m, model in modelos.items():
            if nombre_m == "OLS":
                w = extraer_pesos_ols(X_train_std, x_c_std)
            elif nombre_m == "Ridge":
                w = extraer_pesos_ridge(model, X_train_std, x_c_std)
            elif nombre_m == "k-NN":
                w = extraer_pesos_knn(model, X_train_std, x_c_std)
            elif nombre_m == "Árbol":
                w = extraer_pesos_arbol(model, X_train_std, x_c_std)
            elif nombre_m == "Random Forest":
                w = pesos_rf_test[idx_c]

            y_pred = float(np.sum(w * y_train))
            nef = calcular_nef(w)
            conc = calcular_concentracion(w)

            obtener_testigos(
                w, X_train_df, pd.Series(y_train), top_n=5, variables_clave=variables_clave
            )

            resumen_filas.append(
                {
                    "Consulta": nombre_c,
                    "Idx_Test": idx_c,
                    "Modelo": nombre_m,
                    "y_real": y_c_real,
                    "y_pred": round(y_pred, 4),
                    "N_ef": round(nef, 2),
                    "c(x)": round(conc, 4),
                }
            )

    df_resumen = pd.DataFrame(resumen_filas)

    # Guardar en CSV
    dir_tablas = ROOT_DIR / "resultados" / "tablas"
    dir_tablas.mkdir(parents=True, exist_ok=True)
    ruta_csv = dir_tablas / "parte2_consultas_evidencia.csv"
    df_resumen.to_csv(ruta_csv, index=False)

    print(f"\n✅ Tabla guardada en: {ruta_csv}\n")
    print(df_resumen.to_string(index=False))

    return df_resumen


if __name__ == "__main__":
    ejecutar_seleccion_consultas()