from pathlib import Path
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Asegurar que el directorio raíz está en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.comunes.config import CV_FOLDS, RANDOM_STATE
from src.comunes.datos import cargar_wine


def ejecutar_cv_y_guardar_modelos():
    print(" Carga de Datos Reales: Wine Quality (Tinto)")

    # Cargar estructura de datos producida por el grupo
    datos = cargar_wine()
    X_train_std = datos["X_train_std"]
    X_test_std = datos["X_test_std"]
    y_train = datos["y_train"]
    y_test = datos["y_test"]

    print(
        f"✓ Entrenamiento: {X_train_std.shape[0]} muestras, {X_train_std.shape[1]} variables"
    )
    print(f"✓ Prueba: {X_test_std.shape[0]} muestras\n")

    modelos_optimos = {}

    # ---------------------------------------------------------
    # 1. OLS (Regresión Lineal Simple)
    # ---------------------------------------------------------
    print("--> 1/5 Ajustando OLS...")
    ols = LinearRegression()
    ols.fit(X_train_std, y_train)
    modelos_optimos["OLS"] = ols
    print("    [✓] OLS ajustado exitosamente.")

    # ---------------------------------------------------------
    # 2. Ridge Regression
    # ---------------------------------------------------------
    print(f"\n--> 2/5 Optimizando Ridge ({CV_FOLDS}-fold CV)...")
    ridge_grid = GridSearchCV(
        Ridge(random_state=RANDOM_STATE),
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]},
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    ridge_grid.fit(X_train_std, y_train)
    modelos_optimos["Ridge"] = ridge_grid.best_estimator_
    print(f"    [✓] Mejor alpha: {ridge_grid.best_params_['alpha']}")

    # ---------------------------------------------------------
    # 3. k-NN
    # ---------------------------------------------------------
    print(f"\n--> 3/5 Optimizando k-NN ({CV_FOLDS}-fold CV)...")
    knn_grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid={"n_neighbors": list(range(3, 35, 2))},
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    knn_grid.fit(X_train_std, y_train)
    modelos_optimos["k-NN"] = knn_grid.best_estimator_
    print(
        f"    [✓] Mejor n_neighbors: {knn_grid.best_params_['n_neighbors']}"
    )

    # ---------------------------------------------------------
    # 4. Árbol de Decisión
    # ---------------------------------------------------------
    print(f"\n--> 4/5 Optimizando Árbol de Decisión ({CV_FOLDS}-fold CV)...")
    tree_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid={
            "max_depth": list(range(2, 13)),
            "min_samples_leaf": [1, 2, 5, 10],
        },
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    tree_grid.fit(X_train_std, y_train)
    modelos_optimos["Árbol"] = tree_grid.best_estimator_
    print(
        f"    [✓] Mejores parámetros Árbol: {tree_grid.best_params_}"
    )

    # ---------------------------------------------------------
    # 5. Random Forest Regressor
    # ---------------------------------------------------------
    print(f"\n--> 5/5 Optimizando Random Forest ({CV_FOLDS}-fold CV)...")
    rf_grid = GridSearchCV(
        RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, bootstrap=False
        ),
        param_grid={
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": [0.33, "sqrt", 1.0],
        },
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rf_grid.fit(X_train_std, y_train)
    modelos_optimos["Random Forest"] = rf_grid.best_estimator_
    print(f"    [✓] Mejores parámetros RF: {rf_grid.best_params_}")

    # Guardar artefactos optimizados en resultados/
    dir_salida = ROOT_DIR / "resultados"
    dir_salida.mkdir(exist_ok=True)
    ruta_guardado = dir_salida / "modelos_entrenados.joblib"

    joblib.dump(
        {
            "modelos": modelos_optimos,
            "datos": datos,
        },
        ruta_guardado,
    )

    print("\n==================================================")
    print(f"Modelos guardados exitosamente en:\n   {ruta_guardado}")
    print("==================================================")

    return modelos_optimos, datos


if __name__ == "__main__":
    ejecutar_cv_y_guardar_modelos()