from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RUTAS
RAIZ = Path(__file__).resolve().parents[2]
TABLAS = RAIZ / "resultados" / "tablas"
GRAFICAS = RAIZ / "resultados" / "graficas"

GRAFICAS.mkdir(parents=True, exist_ok=True)

ARCHIVO_ERRORES = TABLAS / "parte3_error_complejidad.csv"
ARCHIVO_MONTECARLO = TABLAS / "parte3_montecarlo_resumen.csv"

ARCHIVO_SALIDA = TABLAS / "parte3_brecha_optimismo.csv"
GRAFICA_SALIDA = GRAFICAS / "parte3_brecha_optimismo.png"

# CONFIGURACIONES COMUNES A AMBOS EXPERIMENTOS
ORDEN = [
    ("Ridge", "alpha=100"),
    ("k-NN", "k=30"),
    ("k-NN", "k=5"),
    ("Arbol", "max_depth=6"),
    ("Arbol", "max_depth=10")
]

ETIQUETAS = [
    "Ridge\nα=100",
    "k-NN\nk=30",
    "k-NN\nk=5",
    "Árbol\nprof. 6",
    "Árbol\nprof. 10"
]

# CARGAR RESULTADOS
def cargar_datos():
    errores = pd.read_csv(ARCHIVO_ERRORES)
    montecarlo = pd.read_csv(ARCHIVO_MONTECARLO)

    columnas_errores = {
        "modelo",
        "configuracion",
        "grados_libertad",
        "mse_train",
        "mse_test",
        "brecha_test_train"
    }

    columnas_montecarlo = {
        "modelo",
        "configuracion",
        "optimismo_empirico",
        "optimismo_traza",
        "error_estandar"
    }

    if not columnas_errores.issubset(errores.columns):
        raise ValueError("Faltan columnas en error-complejidad.")

    if not columnas_montecarlo.issubset(montecarlo.columns):
        raise ValueError("Faltan columnas en Monte Carlo.")

    return errores, montecarlo

# COMPARAR AMBOS EXPERIMENTOS
def construir_comparacion(errores, montecarlo):
    configuraciones = pd.DataFrame(
        ORDEN,
        columns=["modelo", "configuracion"]
    )

    columnas_errores = [
        "modelo",
        "configuracion",
        "grados_libertad",
        "mse_train",
        "mse_test",
        "brecha_test_train"
    ]

    columnas_montecarlo = [
        "modelo",
        "configuracion",
        "optimismo_empirico",
        "optimismo_traza",
        "error_estandar"
    ]

    comparacion = configuraciones.merge(
        errores[columnas_errores],
        on=["modelo", "configuracion"],
        how="left",
        validate="one_to_one"
    )

    comparacion = comparacion.merge(
        montecarlo[columnas_montecarlo],
        on=["modelo", "configuracion"],
        how="left",
        validate="one_to_one"
    )

    if comparacion.isna().any().any():
        raise ValueError(
            "Faltan configuraciones al unir los experimentos."
        )

    brecha_calculada = (
        comparacion["mse_test"] -
        comparacion["mse_train"]
    )

    if not np.allclose(
        brecha_calculada,
        comparacion["brecha_test_train"]
    ):
        raise ValueError(
            "La brecha guardada no coincide con los MSE."
        )

    comparacion["diferencia_brecha_montecarlo"] = (
        comparacion["brecha_test_train"] -
        comparacion["optimismo_empirico"]
    )

    return comparacion

# GRAFICA COMPARATIVA
def graficar_comparacion(comparacion):
    x = np.arange(len(comparacion))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - ancho / 2,
        comparacion["brecha_test_train"],
        width=ancho,
        label="Brecha test - train (Wine real)",
        color="#3274A1"
    )

    ax.bar(
        x + ancho / 2,
        comparacion["optimismo_empirico"],
        width=ancho,
        yerr=comparacion["error_estandar"],
        capsize=4,
        label="Optimismo Monte Carlo (X fijo)",
        color="#E1812C"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.9
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS)

    ax.set_ylabel("Diferencia de errores cuadráticos medios")
    ax.set_title(
        "Brecha test/train frente al optimismo Monte Carlo"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "La brecha corresponde a una partición real; "
        "Monte Carlo utiliza respuestas simuladas sobre X fijo.",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(GRAFICA_SALIDA, dpi=300)
    plt.close()

# EJECUCION
if __name__ == "__main__":
    print("=== BRECHA TEST/TRAIN VS OPTIMISMO ===")

    errores, montecarlo = cargar_datos()

    comparacion = construir_comparacion(
        errores, montecarlo
    )

    comparacion.to_csv(
        ARCHIVO_SALIDA,
        index=False
    )

    print("\n=== RESULTADOS ===")

    columnas = [
        "modelo",
        "configuracion",
        "brecha_test_train",
        "optimismo_empirico",
        "diferencia_brecha_montecarlo"
    ]

    print(
        comparacion[columnas].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    graficar_comparacion(comparacion)

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ARCHIVO_SALIDA)
    print(GRAFICA_SALIDA)

    print("\nAnalisis finalizado correctamente.")