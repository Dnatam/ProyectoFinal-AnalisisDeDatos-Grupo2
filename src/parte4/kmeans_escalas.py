import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.comunes.datos import RAIZ

K_RANGO = range(2, 9)
K_ELEGIDO = 3
SEMILLA = 0


def kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)


def barrido_k(espacios_X, k_rango=K_RANGO):
    """Inercia y silueta para cada k en cada escala."""
    filas = []
    for escala, X in espacios_X.items():
        for k in k_rango:
            km = kmeans(X, k)
            filas.append({"escala": escala, "k": k, "inercia": km.inertia_,
                          "silueta": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas)


def eta2(X, etiquetas, columnas):
    """Proporción de la varianza de cada variable explicada por los grupos.

    eta^2_j = sum_g (n_g/n) (media_gj - media_j)^2 / var_j, entre 0 y 1. No depende
    de la escala de la variable, así que permite comparar variables entre sí.
    """
    X = np.asarray(X, dtype=float)
    media = X.mean(axis=0)
    entre = np.zeros(X.shape[1])
    for g in np.unique(etiquetas):
        m = etiquetas == g
        entre += m.mean() * (X[m].mean(axis=0) - media) ** 2
    return pd.Series(entre / X.var(axis=0), index=columnas)


def perfil_grupos(X_crudo, etiquetas, y, columnas):
    """Centros en unidades originales, tamaño y nota media de cada grupo."""
    df = pd.DataFrame(X_crudo, columns=columnas)
    df["grupo"] = etiquetas
    perfil = df.groupby("grupo").mean().T
    perfil.loc["n_vinos"] = np.bincount(etiquetas)
    perfil.loc["nota_media (no usada)"] = [y[etiquetas == g].mean() for g in perfil.columns]
    return perfil


def grafica_barrido(barrido, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for escala, sub in barrido.groupby("escala", sort=False):
        ejes[0].plot(sub["k"], sub["inercia"] / sub["inercia"].iloc[0], "o-", label=escala)
        ejes[1].plot(sub["k"], sub["silueta"], "o-", label=escala)
    ejes[0].set(title="Inercia (relativa a k = 2)", xlabel="k", ylabel="inercia / inercia(k=2)")
    ejes[1].set(title="Silueta", xlabel="k", ylabel="silueta media")
    for eje in ejes:
        eje.axvline(K_ELEGIDO, color="gray", ls="--", lw=1)
        eje.legend()
        eje.grid(alpha=0.3)
    fig.suptitle("k-means en Wine: elección de k")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from src.comunes.datos import cargar_wine

    datos = cargar_wine()
    columnas, y = datos["columnas"], datos["y_train"]
    espacios_X = {"Crudas": datos["X_train"], "Estandarizadas": datos["X_train_std"]}

    print("=== Elección de k (entrenamiento) ===")
    barrido = barrido_k(espacios_X)
    print(barrido.pivot(index="k", columns="escala", values=["inercia", "silueta"]).round(3).to_string())

    km_crudo = kmeans(datos["X_train"], K_ELEGIDO)
    km_std = kmeans(datos["X_train_std"], K_ELEGIDO)
    et_crudo, et_std = km_crudo.labels_, km_std.labels_

    print(f"\n=== Tabla cruzada de asignaciones (k = {K_ELEGIDO}) ===")
    cruzada = pd.crosstab(pd.Series(et_crudo, name="grupo_crudo"),
                          pd.Series(et_std, name="grupo_estandarizado"), margins=True)
    print(cruzada.to_string())
    print("Índice de Rand ajustado (0 = como al azar, 1 = idénticas):",
          round(adjusted_rand_score(et_crudo, et_std), 3))

    print("\n=== eta^2: qué variable separa los grupos ===")
    separacion = pd.DataFrame({
        "eta2_grupos_crudos": eta2(datos["X_train"], et_crudo, columnas),
        "eta2_grupos_estandarizados": eta2(datos["X_train"], et_std, columnas),
    })
    print(separacion.sort_values("eta2_grupos_crudos", ascending=False).round(3).to_string())

    print("\n=== Perfil de los grupos crudos (unidades originales) ===")
    perfil_crudo = perfil_grupos(datos["X_train"], et_crudo, y, columnas)
    print(perfil_crudo.round(3).to_string())

    print("\n=== Perfil de los grupos estandarizados (unidades originales) ===")
    perfil_std = perfil_grupos(datos["X_train"], et_std, y, columnas)
    print(perfil_std.round(3).to_string())

    tablas = RAIZ / "resultados" / "tablas"
    graficas = RAIZ / "resultados" / "graficas"
    tablas.mkdir(parents=True, exist_ok=True)
    graficas.mkdir(parents=True, exist_ok=True)
    barrido.to_csv(tablas / "4_4_kmeans_barrido_k_wine.csv", index=False)
    cruzada.to_csv(tablas / "4_4_kmeans_tabla_cruzada_wine.csv")
    separacion.to_csv(tablas / "4_4_kmeans_eta2_wine.csv")
    perfil_crudo.to_csv(tablas / "4_4_kmeans_perfil_crudo_wine.csv")
    perfil_std.to_csv(tablas / "4_4_kmeans_perfil_estandarizado_wine.csv")
    grafica_barrido(barrido, graficas / "4_4_kmeans_eleccion_k_wine.png")
    print(f"\nTablas en {tablas}\nFigura en {graficas / '4_4_kmeans_eleccion_k_wine.png'}")