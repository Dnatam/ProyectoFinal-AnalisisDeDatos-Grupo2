import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src.comunes.datos import RAIZ

PROFUNDIDADES = [3, 6, 10, None]
PROFUNDIDAD_DETALLE = 6


def ajustar_par(datos, max_depth, random_state=0):
    """El mismo árbol ajustado sobre crudas y sobre estandarizadas."""
    crudo = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    estand = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    crudo.fit(datos["X_train"], datos["y_train"])
    estand.fit(datos["X_train_std"], datos["y_train"])
    return crudo, estand


def comparar(datos, crudo, estand):
    """Compara estructura, umbrales, partición del entrenamiento y predicciones."""
    tc, te = crudo.tree_, estand.tree_
    misma_estructura = (tc.node_count == te.node_count
                        and np.array_equal(tc.feature, te.feature)
                        and np.array_equal(tc.children_left, te.children_left))

    resultado = {
        "nodos": tc.node_count,
        "hojas": crudo.get_n_leaves(),
        "misma_estructura": misma_estructura,
    }
    if misma_estructura:
        internos = tc.feature >= 0
        f = tc.feature[internos]
        sc = datos["scaler"]
        convertido = (tc.threshold[internos] - sc.mean_[f]) / sc.scale_[f]
        resultado["max_dif_umbral"] = np.abs(convertido - te.threshold[internos]).max()
        resultado["misma_particion_train"] = np.array_equal(
            crudo.apply(datos["X_train"]), estand.apply(datos["X_train_std"]))

    pc = crudo.predict(datos["X_test"])
    pe = estand.predict(datos["X_test_std"])
    resultado["filas_test_distintas"] = int((pc != pe).sum())
    resultado["max_dif_prediccion"] = np.abs(pc - pe).max()
    resultado["RMSE_crudo"] = np.sqrt(mean_squared_error(datos["y_test"], pc))
    resultado["RMSE_estandarizado"] = np.sqrt(mean_squared_error(datos["y_test"], pe))
    return resultado


def diagnosticar(datos, crudo, estand):
    """Para cada fila de prueba con predicción distinta: nodo donde se separan los
    caminos, variable, valor de la fila, umbral en cada escala y si la fila cae
    exactamente en el punto medio de los dos valores de entrenamiento del corte."""
    pc = crudo.predict(datos["X_test"])
    pe = estand.predict(datos["X_test_std"])
    camino_train = crudo.decision_path(datos["X_train"])
    filas = []
    for i in np.flatnonzero(pc != pe):
        ruta_c = crudo.decision_path(datos["X_test"][i:i + 1]).indices
        ruta_e = estand.decision_path(datos["X_test_std"][i:i + 1]).indices
        k = next(j for j in range(min(len(ruta_c), len(ruta_e))) if ruta_c[j] != ruta_e[j])
        nodo = ruta_c[k - 1]
        f = crudo.tree_.feature[nodo]
        umbral = crudo.tree_.threshold[nodo]

        # Valores de entrenamiento del nodo a cada lado del corte (en float64)
        en_nodo = camino_train[:, nodo].nonzero()[0]
        v = datos["X_train"][en_nodo, f]
        abajo, arriba = v[v <= umbral].max(), v[v > umbral].min()
        x = datos["X_test"][i, f]

        filas.append({
            "fila_test": i,
            "nodo": nodo,
            "variable": datos["columnas"][f],
            "valor_fila": x,
            "train_abajo": abajo,
            "train_arriba": arriba,
            "punto_medio_exacto": (abajo + arriba) / 2,
            "en_el_punto_medio": bool(np.isclose(x, (abajo + arriba) / 2, rtol=0, atol=1e-12)),
            "umbral_crudo_float32": umbral,
            # El árbol compara el valor convertido a float32 contra el umbral
            "va_a_crudo": "izquierda" if np.float32(x) <= umbral else "derecha",
            "va_a_estandarizado": ("izquierda" if np.float32(datos["X_test_std"][i, f])
                                   <= estand.tree_.threshold[nodo] else "derecha"),
            "pred_crudo": pc[i],
            "pred_estandarizado": pe[i],
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    from src.comunes.datos import cargar_wine

    datos = cargar_wine()

    print("=== El mismo árbol en escala cruda y estandarizada (Wine) ===")
    filas = []
    for prof in PROFUNDIDADES:
        crudo, estand = ajustar_par(datos, prof)
        filas.append({"max_depth": "None" if prof is None else prof, **comparar(datos, crudo, estand)})
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    print("\n=== Diagnóstico de las filas que difieren ===")
    diagnosticos = []
    for prof in PROFUNDIDADES:
        crudo, estand = ajustar_par(datos, prof)
        diag = diagnosticar(datos, crudo, estand)
        if len(diag):
            diag.insert(0, "max_depth", "None" if prof is None else prof)
            diagnosticos.append(diag)
    diagnosticos = pd.concat(diagnosticos, ignore_index=True)
    cols = ["max_depth", "fila_test", "variable", "valor_fila", "train_abajo", "train_arriba",
            "en_el_punto_medio", "va_a_crudo", "va_a_estandarizado", "pred_crudo", "pred_estandarizado"]
    print(diagnosticos[cols].to_string(index=False))
    print("\n¿Todas las filas que difieren caen exactamente en el punto medio del corte?",
          bool(diagnosticos["en_el_punto_medio"].all()))

    d6 = diagnosticos[diagnosticos["max_depth"] == PROFUNDIDAD_DETALLE]
    if len(d6):
        r = d6.iloc[0]
        print(f"\nDetalle (max_depth = {PROFUNDIDAD_DETALLE}, fila {r.fila_test}, nodo {r.nodo}):")
        print(f"  corte en '{r.variable}' entre {r.train_abajo} y {r.train_arriba} -> punto medio exacto {r.punto_medio_exacto}")
        print(f"  valor de la fila: {r.valor_fila}")
        print(f"  umbral guardado en escala cruda (float32): {r.umbral_crudo_float32!r}")
        print(f"  la fila va a la {r.va_a_crudo} en cruda y a la {r.va_a_estandarizado} en estandarizada")

    carpeta = RAIZ / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "4_3_arbol_escalas_wine.csv", index=False)
    diagnosticos.to_csv(carpeta / "4_3_arbol_diagnostico_wine.csv", index=False)
    print(f"\nTablas guardadas en {carpeta}")