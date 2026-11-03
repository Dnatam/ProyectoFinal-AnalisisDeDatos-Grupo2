import numpy as np
import pandas as pd

from src.comunes.datos import RAIZ
from src.parte1.pesos_bosque import bosque_parte1, pesos_bosque

FILA = 7


def proximidad_bosque(bosque, X_train, X_consulta):
    """Proximidad de cada fila de entrenamiento a la(s) consulta(s).

    Devuelve P de forma (m, n_train), o un vector si hay una sola consulta.
    """
    una_sola = np.asarray(X_consulta).ndim == 1
    X_consulta = np.atleast_2d(X_consulta)

    hojas_train = bosque.apply(X_train)        # (n, T)
    hojas_consulta = bosque.apply(X_consulta)  # (m, T)
    T = hojas_train.shape[1]

    P = np.zeros((X_consulta.shape[0], hojas_train.shape[0]))
    for t in range(T):
        P += hojas_consulta[:, t][:, None] == hojas_train[:, t][None, :]
    P /= T
    return P[0] if una_sola else P


def tamanos_hoja(bosque, X_train, x):
    """|L_t(x)|: número de filas de entrenamiento en la hoja de x, árbol por árbol."""
    hojas_train = bosque.apply(X_train)                  # (n, T)
    hojas_x = bosque.apply(np.atleast_2d(x))[0]          # (T,)
    return (hojas_train == hojas_x[None, :]).sum(axis=0)


if __name__ == "__main__":
    from src.comunes.datos import cargar_california

    datos = cargar_california()
    X_train, X_test = datos["X_train_std"], datos["X_test_std"]
    y_train = datos["y_train"]
    x = X_test[FILA]

    bosque = bosque_parte1(bootstrap=False).fit(X_train, y_train)
    pred = bosque.predict(x[None, :])[0]

    prox = proximidad_bosque(bosque, X_train, x)
    w = pesos_bosque(bosque, X_train, x)
    tam = tamanos_hoja(bosque, X_train, x)

    print(f"=== Proximidad del bosque, fila de prueba {FILA} ===")
    print("Suma de la proximidad:", prox.sum())
    print("Tamaño promedio de la hoja de x en los 200 árboles:", tam.mean())
    print("Tamaño de hoja (mín, máx):", tam.min(), tam.max())
    print("Proximidad máxima:", prox.max(),
          " filas con proximidad 1 (misma hoja en todos los árboles):", int((prox == 1).sum()))

    print("\n=== Comparación con los pesos ===")
    print("Suma de los pesos:", w.sum())
    print("Filas con proximidad > 0:", int((prox > 0).sum()), " filas con peso > 0:", int((w > 0).sum()))
    print("Mismo soporte (prox > 0 exactamente donde w > 0):", np.array_equal(prox > 0, w > 0))
    print("sum prox_i y_i (no es una predicción):", prox @ y_train, " predict:", pred)

    prox_norm = prox / prox.sum()
    print("\n=== Proximidad normalizada al final (prox / sum prox) ===")
    print("Suma:", prox_norm.sum())
    print("Máx |prox_norm - w|:", np.max(np.abs(prox_norm - w)))
    print("Predicción con prox_norm:", prox_norm @ y_train, " predict:", pred,
          " diferencia:", prox_norm @ y_train - pred)

    top = np.argsort(-w)[:5]
    tabla = pd.DataFrame({
        "fila_train": top,
        "y": y_train[top],
        "proximidad": prox[top],
        "peso": w[top],
        "prox_normalizada": prox_norm[top],
    })
    print("\nLas 5 filas con mayor peso:")
    print(tabla.to_string(index=False))

    resumen = pd.DataFrame([{
        "fila_prueba": FILA,
        "suma_proximidad": prox.sum(),
        "tamano_hoja_promedio": tam.mean(),
        "suma_pesos": w.sum(),
        "filas_no_nulas": int((w > 0).sum()),
        "pred_bosque": pred,
        "pred_prox_normalizada": prox_norm @ y_train,
        "max_dif_prox_norm_vs_peso": np.max(np.abs(prox_norm - w)),
    }])
    carpeta = RAIZ / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(carpeta / f"1_6_proximidad_fila{FILA}_california.csv", index=False)
    tabla.to_csv(carpeta / f"1_6_top5_fila{FILA}_california.csv", index=False)
    print(f"\nTablas guardadas en {carpeta}")