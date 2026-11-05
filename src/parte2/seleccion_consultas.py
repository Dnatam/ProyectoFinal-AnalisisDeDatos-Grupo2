from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.parte1.pesos_bosque import pesos_bosque
from src.parte2.modulo_evidencia import (
    DIR_TABLAS,
    ORDEN_MODELOS,
    calcular_concentracion,
    calcular_nef,
    cargar_artefactos,
    indices_gemelos,
    obtener_testigos,
    pesos_verificados,
    variables_importantes,
)

N_TESTIGOS = 5
N_VARIABLES = 4


def ejecutar_seleccion_consultas():
    print("2.1 Selección de consultas y ficha de evidencia\n")

    modelos, datos = cargar_artefactos()
    X_train_std, X_test_std = datos["X_train_std"], datos["X_test_std"]
    y_train, y_test = datos["y_train"], datos["y_test"]
    columnas = datos["columnas"]

    # Train en unidades originales (para que los testigos se puedan leer)
    X_train_orig = pd.DataFrame(datos["X_train"], columns=columnas)
    X_test_orig = pd.DataFrame(datos["X_test"], columns=columnas)

    # 1. N_ef y c(x) del bosque en todo el conjunto de prueba
    rf = modelos["Random Forest"]
    W_rf = pesos_bosque(rf, X_train_std, X_test_std)          # (n_test, n_train)
    diferencia = np.max(np.abs(W_rf @ y_train - rf.predict(X_test_std)))
    assert diferencia < 1e-8, f"Los pesos del bosque no reproducen .predict(): {diferencia:.2e}"
    print(f"Pesos del bosque verificados en las {len(X_test_std)} filas de prueba "
          f"(máx |w·y - pred| = {diferencia:.1e})")

    nef_rf = np.array([calcular_nef(w) for w in W_rf])
    conc_rf = np.array([calcular_concentracion(w) for w in W_rf])

    # 2. Las tres consultas
    idx_mediano = int(np.argsort(nef_rf, kind="stable")[len(nef_rf) // 2])
    idx_minimo = int(np.argmin(nef_rf))
    idx_interes = int(np.argmax(conc_rf))  # la de mayor concentración c(x)
    consultas = {
        "Nef Mediano": idx_mediano,
        "Nef Mínimo": idx_minimo,
        "Interés (Máx Concentración)": idx_interes,
    }

    variables_clave = variables_importantes(rf, columnas, n=N_VARIABLES)
    print(f"Variables más importantes del bosque: {variables_clave}\n")

    print("--- Consultas seleccionadas ---")
    for nombre, idx in consultas.items():
        gemelos = indices_gemelos(X_test_std[idx], X_train_std)
        texto_gemelo = (f"SÍ, filas de train {gemelos.tolist()} con y = {y_train[gemelos].tolist()}"
                        if len(gemelos) else "no")
        print(f"• {nombre}: fila de prueba {idx} | N_ef RF = {nef_rf[idx]:.2f} | "
              f"c(x) RF = {conc_rf[idx]:.4f} | y real = {y_test[idx]} | gemelo en train: {texto_gemelo}")
        print("  Valores de la consulta:",
              X_test_orig.loc[idx, variables_clave].round(3).to_dict())

    # 3. Ficha de evidencia para cada consulta y cada modelo
    resumen, testigos = [], []
    for nombre_c, idx_c in consultas.items():
        x_c = X_test_std[idx_c]
        gemelos = indices_gemelos(x_c, X_train_std)

        for nombre_m in ORDEN_MODELOS:
            w, pred = pesos_verificados(nombre_m, modelos[nombre_m], X_train_std, y_train, x_c)

            resumen.append({
                "Consulta": nombre_c,
                "Idx_Test": idx_c,
                "Modelo": nombre_m,
                "y_real": y_test[idx_c],
                "y_pred": round(pred, 4),
                "N_ef": round(calcular_nef(w), 2),
                "c(x)": round(calcular_concentracion(w), 4),
                "suma_w": round(float(w.sum()), 6),
                "pesos_no_cero": int(np.sum(np.abs(w) > 1e-12)),
                "pesos_negativos": int(np.sum(w < -1e-12)),
                "peso_en_gemelos": round(float(w[gemelos].sum()), 4) if len(gemelos) else 0.0,
                "tiene_gemelo": len(gemelos) > 0,
            })

            t = obtener_testigos(w, X_train_orig, y_train, top_n=N_TESTIGOS,
                                 variables_clave=variables_clave)
            t["es_gemelo"] = t["fila_train"].isin(gemelos)
            t.insert(0, "Consulta", nombre_c)
            t.insert(1, "Idx_Test", idx_c)
            t.insert(2, "Modelo", nombre_m)
            testigos.append(t)

    df_resumen = pd.DataFrame(resumen)
    df_testigos = pd.concat(testigos, ignore_index=True)
    df_testigos["peso_w"] = df_testigos["peso_w"].round(5)

    # 4. Guardar
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta_resumen = DIR_TABLAS / "parte2_consultas_evidencia.csv"
    ruta_testigos = DIR_TABLAS / "parte2_testigos.csv"
    df_resumen.to_csv(ruta_resumen, index=False)
    df_testigos.to_csv(ruta_testigos, index=False)

    print("\n--- Resumen por consulta y modelo ---")
    print(df_resumen.to_string(index=False))
    print("\n--- Testigos (5 filas de train con mayor |w|, unidades originales) ---")
    for (nombre_c, nombre_m), t in df_testigos.groupby(["Consulta", "Modelo"], sort=False):
        print(f"\n{nombre_c} — {nombre_m}")
        print(t.drop(columns=["Consulta", "Idx_Test", "Modelo"]).to_string(index=False))

    print(f"\nTablas guardadas en:\n  {ruta_resumen}\n  {ruta_testigos}")
    return df_resumen, df_testigos


if __name__ == "__main__":
    ejecutar_seleccion_consultas()