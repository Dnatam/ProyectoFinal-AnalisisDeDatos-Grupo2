from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RUTAS DEL PROYECTO
RAIZ = Path(__file__).resolve().parents[2]

CARPETA_TABLAS = RAIZ / "resultados" / "tablas"
CARPETA_GRAFICAS = RAIZ / "resultados" / "graficas"

CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

ARCHIVO_ENTRADA = (
    CARPETA_TABLAS / "parte3_montecarlo_repeticiones.csv"
)

ARCHIVO_RESUMEN = (
    CARPETA_TABLAS / "parte3_montecarlo_resumen.csv"
)

GRAFICA_COMPARACION = (
    CARPETA_GRAFICAS / "parte3_montecarlo_comparacion.png"
)

GRAFICA_DISTRIBUCION = (
    CARPETA_GRAFICAS / "parte3_montecarlo_distribucion.png"
)

# ORDEN DE PRESENTACION
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

# CARGAR Y VALIDAR RESULTADOS
def cargar_resultados():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(ARCHIVO_ENTRADA)

    columnas = [
        "repeticion",
        "modelo",
        "configuracion",
        "grados_libertad",
        "mse_train",
        "mse_new",
        "optimismo_empirico",
        "optimismo_traza",
        "sigma"
    ]

    if not set(columnas).issubset(df.columns):
        raise ValueError("Faltan columnas en el CSV.")

    if df[columnas].isna().any().any():
        raise ValueError("El CSV contiene valores faltantes.")

    if len(df) != 250:
        raise ValueError(
            f"Se esperaban 250 registros y hay {len(df)}."
        )

    if df.duplicated(
        subset=["repeticion", "modelo", "configuracion"]
    ).any():
        raise ValueError("Hay repeticiones duplicadas.")

    configuraciones = set(
        zip(df["modelo"], df["configuracion"])
    )

    if configuraciones != set(ORDEN):
        raise ValueError(
            "Las configuraciones no coinciden con las esperadas."
        )

    for modelo, configuracion in ORDEN:
        grupo = df[
            (df["modelo"] == modelo) &
            (df["configuracion"] == configuracion)
        ]

        if set(grupo["repeticion"]) != set(range(1, 51)):
            raise ValueError(
                f"Repeticiones incompletas: {modelo} {configuracion}"
            )

    diferencia = (
        df["mse_new"] - df["mse_train"]
    )

    if not np.allclose(
        diferencia, df["optimismo_empirico"]
    ):
        raise ValueError(
            "El optimismo no coincide con la diferencia de MSE."
        )

    return df

# CALCULAR ESTADISTICAS
def calcular_resumen(df):
    resumen = df.groupby(
        ["modelo", "configuracion"],
        as_index=False
    ).agg(
        repeticiones=("repeticion", "count"),
        df_promedio=("grados_libertad", "mean"),
        mse_train_promedio=("mse_train", "mean"),
        mse_new_promedio=("mse_new", "mean"),
        optimismo_empirico=("optimismo_empirico", "mean"),
        desviacion_estandar=("optimismo_empirico", "std"),
        optimismo_traza=("optimismo_traza", "mean")
    )

    resumen["error_estandar"] = (
        resumen["desviacion_estandar"] /
        np.sqrt(resumen["repeticiones"])
    )

    resumen["diferencia"] = (
        resumen["optimismo_empirico"] -
        resumen["optimismo_traza"]
    )

    resumen["ic95_inferior"] = (
        resumen["optimismo_empirico"] -
        1.96 * resumen["error_estandar"]
    )

    resumen["ic95_superior"] = (
        resumen["optimismo_empirico"] +
        1.96 * resumen["error_estandar"]
    )

    resumen["orden"] = resumen.apply(
        lambda fila: ORDEN.index(
            (fila["modelo"], fila["configuracion"])
        ),
        axis=1
    )

    resumen = (
        resumen.sort_values("orden")
        .drop(columns="orden")
        .reset_index(drop=True)
    )

    return resumen

# GRAFICA 1: EMPIRICO VS REFERENCIA
def graficar_comparacion(resumen):
    x = np.arange(len(resumen))
    ancho = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - ancho / 2,
        resumen["optimismo_empirico"],
        width=ancho,
        yerr=resumen["error_estandar"],
        capsize=5,
        label="Optimismo empírico",
        color="#3274A1"
    )

    ax.bar(
        x + ancho / 2,
        resumen["optimismo_traza"],
        width=ancho,
        label="Referencia basada en tr(S)",
        color="#E1812C"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS)

    ax.set_ylabel("Optimismo (diferencia de MSE)")
    ax.set_title(
        "Monte Carlo: optimismo empírico frente a referencia de traza"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "Barras de error: ±1 error estándar. "
        "La referencia no es exacta para árboles adaptativos.",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(GRAFICA_COMPARACION, dpi=300)
    plt.close()

# GRAFICA 2: DISTRIBUCION DE LAS REPETICIONES
def graficar_distribucion(df):
    grupos = []

    for modelo, configuracion in ORDEN:
        valores = df.loc[
            (df["modelo"] == modelo) &
            (df["configuracion"] == configuracion),
            "optimismo_empirico"
        ].to_numpy()

        grupos.append(valores)

    fig, ax = plt.subplots(figsize=(11, 6))

    box = ax.boxplot(
        grupos,
        tick_labels=ETIQUETAS,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black"
        }
    )

    colores = [
        "#4C72B0",
        "#55A868",
        "#64B5CD",
        "#C44E52",
        "#8172B3"
    ]

    for parche, color in zip(box["boxes"], colores):
        parche.set_facecolor(color)
        parche.set_alpha(0.75)

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax.set_ylabel("Optimismo empírico")
    ax.set_title(
        "Distribución del optimismo en 50 simulaciones Monte Carlo"
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "La línea central indica la mediana; "
        "el rombo blanco indica la media.",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(GRAFICA_DISTRIBUCION, dpi=300)
    plt.close()

# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=== ANALISIS DE MONTE CARLO ===")

    df = cargar_resultados()

    print(f"\nRegistros cargados: {len(df)}")
    print(f"Repeticiones: {df['repeticion'].nunique()}")

    resumen = calcular_resumen(df)

    resumen.to_csv(
        ARCHIVO_RESUMEN,
        index=False
    )

    print("\n=== RESUMEN ESTADISTICO ===")

    columnas_mostrar = [
        "modelo",
        "configuracion",
        "df_promedio",
        "optimismo_empirico",
        "error_estandar",
        "optimismo_traza",
        "diferencia"
    ]

    print(
        resumen[columnas_mostrar].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nGenerando graficas...")

    graficar_comparacion(resumen)
    graficar_distribucion(df)

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ARCHIVO_RESUMEN)
    print(GRAFICA_COMPARACION)
    print(GRAFICA_DISTRIBUCION)

    print("\nAnalisis finalizado correctamente.")
