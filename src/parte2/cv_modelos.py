from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.comunes.config import CV_FOLDS, RANDOM_STATE
from src.comunes.datos import cargar_wine
from src.parte2.modulo_evidencia import DIR_TABLAS, RUTA_MODELOS

BUSQUEDAS = {
    "Ridge": (
        Ridge(),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]},
    ),
    "k-NN": (
        KNeighborsRegressor(),
        {"n_neighbors": list(range(3, 35, 2))},
    ),
    "Árbol": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"max_depth": list(range(2, 13)), "min_samples_leaf": [1, 2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestRegressor(n_estimators=200, bootstrap=False, random_state=RANDOM_STATE),
        {
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": [0.33, "sqrt", 1.0],
        },
    ),
}


def ejecutar_cv_y_guardar_modelos():
    print("Validación cruzada de los cinco modelos — Wine Quality (tinto)")
    datos = cargar_wine()
    X_train_std, y_train = datos["X_train_std"], datos["y_train"]
    print(f"Entrenamiento: {X_train_std.shape[0]} filas, {X_train_std.shape[1]} variables\n")

    modelos, filas = {}, []

    # OLS no tiene hiperparámetros: solo se reporta su error de CV
    ols = LinearRegression().fit(X_train_std, y_train)
    mse_ols = -cross_val_score(LinearRegression(), X_train_std, y_train,
                               cv=CV_FOLDS, scoring="neg_mean_squared_error").mean()
    modelos["OLS"] = ols
    filas.append({"Modelo": "OLS", "Hiperparámetros": "ninguno", "MSE_CV": mse_ols})
    print(f"OLS: MSE CV = {mse_ols:.4f}")

    for nombre, (estimador, rejilla) in BUSQUEDAS.items():
        busqueda = GridSearchCV(estimador, rejilla, cv=CV_FOLDS,
                                scoring="neg_mean_squared_error", n_jobs=-1)
        busqueda.fit(X_train_std, y_train)
        modelos[nombre] = busqueda.best_estimator_
        texto = ", ".join(f"{k}={v}" for k, v in busqueda.best_params_.items())
        filas.append({"Modelo": nombre, "Hiperparámetros": texto,
                      "MSE_CV": -busqueda.best_score_})
        print(f"{nombre}: {texto} | MSE CV = {-busqueda.best_score_:.4f}")

    # Tabla de hiperparámetros escogidos
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    tabla["MSE_CV"] = tabla["MSE_CV"].round(4)
    ruta_tabla = DIR_TABLAS / "parte2_hiperparametros.csv"
    tabla.to_csv(ruta_tabla, index=False)

    # Solo los modelos (sin el diccionario de datos)
    RUTA_MODELOS.parent.mkdir(exist_ok=True)
    joblib.dump({"modelos": modelos}, RUTA_MODELOS)

    print("\n" + tabla.to_string(index=False))
    print(f"\nTabla guardada en {ruta_tabla}")
    print(f"Modelos guardados en {RUTA_MODELOS}")
    return modelos, tabla


if __name__ == "__main__":
    ejecutar_cv_y_guardar_modelos()