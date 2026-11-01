import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# PERMITIR IMPORTAR ARCHIVOS COMPARTIDOS
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.comunes.datos import cargar_wine
from src.comunes.config import RANDOM_STATE, N_SIMULACIONES
from src.parte3.grados_libertad import df_ridge, df_knn
from src.parte3.verificar_matrices import matriz_knn

# CONFIGURACIONES
ALPHA_RIDGE = 100
VALORES_KNN = [5, 30]
PROFUNDIDADES_ARBOL = [6, 10]

# MODELO DE REFERENCIA
def construir_referencia(X, y):
    """
    Estima f(X) con Random Forest.

    Utiliza bootstrap=True para obtener predicciones OOB y estimar la desviacion del ruido.
    """
    modelo = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        max_features=0.33,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(X, y)

    f_estimada = modelo.predict(X)
    pred_oob = modelo.oob_prediction_

    mascara = np.isfinite(pred_oob)

    if not np.all(mascara):
        raise ValueError(
            "Existen observaciones sin prediccion OOB valida."
        )

    rmse_oob = np.sqrt(
        mean_squared_error(y, pred_oob)
    )

    if not np.isfinite(rmse_oob) or rmse_oob <= 0:
        raise ValueError("La estimacion OOB no es valida.")

    return f_estimada, rmse_oob, modelo

# REGISTRAR RESULTADOS
def registrar(
    registros,
    repeticion,
    modelo,
    configuracion,
    mse_train,
    mse_new,
    grados_libertad,
    sigma,
    n_observaciones
):
    optimismo = mse_new - mse_train

    # Referencia de traza: exacta para suavizadores
    # fijos, no necesariamente para arboles adaptativos.
    optimismo_traza = (
        2 * sigma**2 * grados_libertad / n_observaciones
    )
    
    registros.append({
        "repeticion": repeticion,
        "modelo": modelo,
        "configuracion": configuracion,
        "grados_libertad": float(grados_libertad),
        "mse_train": float(mse_train),
        "mse_new": float(mse_new),
        "optimismo_empirico": float(optimismo),
        "optimismo_traza": float(optimismo_traza),
        "sigma": float(sigma)
    })

# EXPERIMENTO MONTE CARLO
if __name__ == "__main__":
    datos = cargar_wine()

    X = datos["X_train_std"]
    y_real = datos["y_train"]

    N_OBSERVACIONES = len(y_real)

    print("=== SIMULACION MONTE CARLO DEL OPTIMISMO ===")
    print(f"Observaciones fijas: {N_OBSERVACIONES}")
    print(f"Repeticiones: {N_SIMULACIONES}")

    # ESTIMAR FUNCION VERDADERA Y RUIDO
    print("\nConstruyendo Random Forest de referencia...")

    f_estimada, sigma, bosque = construir_referencia(
        X, y_real
    )

    print(f"RMSE OOB estimado: {sigma:.6f}")
    print(f"Varianza estimada: {sigma**2:.6f}")
    print(f"R2 OOB: {bosque.oob_score_:.6f}")

    # MATRICES K-NN FIJAS
    print("\nPreparando matrices k-NN...")

    matrices_knn = {
        k: matriz_knn(X, k)
        for k in VALORES_KNN
    }

    # GRADOS DE LIBERTAD FIJOS
    df_ridge_fijo = df_ridge(X, ALPHA_RIDGE)

    df_knn_fijos = {
        k: df_knn(N_OBSERVACIONES, k)
        for k in VALORES_KNN
    }

    # GENERADOR REPRODUCIBLE
    generador = np.random.default_rng(RANDOM_STATE)

    registros = []

    print("\nIniciando simulaciones...")

    for repeticion in range(1, N_SIMULACIONES + 1):

        # DOS RESPUESTAS INDEPENDIENTES
        ruido_train = generador.normal(
            0, 1, N_OBSERVACIONES
        )

        ruido_new = generador.normal(
            0, 1, N_OBSERVACIONES
        )

        y_sim = f_estimada + sigma * ruido_train
        y_new = f_estimada + sigma * ruido_new

        # K-NN: SUAVIZADOR FIJO CON AUTOINCLUSION
        for k in VALORES_KNN:
            pred = matrices_knn[k] @ y_sim

            mse_train = mean_squared_error(
                y_sim, pred
            )

            mse_new = mean_squared_error(
                y_new, pred
            )

            registrar(
                registros,
                repeticion,
                "k-NN",
                f"k={k}",
                mse_train,
                mse_new,
                df_knn_fijos[k],
                sigma,
                N_OBSERVACIONES
            )

        # RIDGE
        modelo_ridge = Ridge(alpha=ALPHA_RIDGE)
        modelo_ridge.fit(X, y_sim)

        pred = modelo_ridge.predict(X)

        registrar(
            registros,
            repeticion,
            "Ridge",
            f"alpha={ALPHA_RIDGE}",
            mean_squared_error(y_sim, pred),
            mean_squared_error(y_new, pred),
            df_ridge_fijo,
            sigma,
            N_OBSERVACIONES
        )

        # ARBOLES DE REGRESION
        for profundidad in PROFUNDIDADES_ARBOL:
            modelo_arbol = DecisionTreeRegressor(
                max_depth=profundidad,
                random_state=RANDOM_STATE
            )

            modelo_arbol.fit(X, y_sim)

            pred = modelo_arbol.predict(X)

            registrar(
                registros,
                repeticion,
                "Arbol",
                f"max_depth={profundidad}",
                mean_squared_error(y_sim, pred),
                mean_squared_error(y_new, pred),
                modelo_arbol.get_n_leaves(),
                sigma,
                N_OBSERVACIONES
            )

        if repeticion % 10 == 0:
            print(
                f"Repeticion {repeticion}/{N_SIMULACIONES} "
                "completada"
            )

    # GUARDAR RESULTADOS INDIVIDUALES
    carpeta_tablas = RAIZ / "resultados" / "tablas"
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame(registros)

    ruta = (
        carpeta_tablas /
        "parte3_montecarlo_repeticiones.csv"
    )

    tabla.to_csv(ruta, index=False)

    # RESUMEN PRELIMINAR
    print("\n=== RESUMEN PRELIMINAR ===")

    resumen = tabla.groupby(
        ["modelo", "configuracion"],
        as_index=False
    ).agg(
        optimismo_empirico=("optimismo_empirico", "mean"),
        optimismo_traza=("optimismo_traza", "mean"),
        mse_train=("mse_train", "mean"),
        mse_new=("mse_new", "mean")
    )

    print(resumen.to_string(index=False))

    print("\n=== ARCHIVOS ===")
    print(f"Tabla guardada en: {ruta}")
    print(f"Total de registros: {len(tabla)}")
    print("\nSimulacion finalizada correctamente.")