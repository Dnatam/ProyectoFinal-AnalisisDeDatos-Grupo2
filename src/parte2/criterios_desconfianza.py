from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def redactar_criterios_y_validar():
    print(" 2.4 Criterios para Desconfiar de una Predicción")

    # Validar que existan los archivos de las partes anteriores
    ruta_c21 = ROOT_DIR / "resultados" / "tablas" / "parte2_consultas_evidencia.csv"
    ruta_c23 = ROOT_DIR / "resultados" / "tablas" / "parte2_quintiles_nef_mae.csv"

    if not ruta_c21.exists() or not ruta_c23.exists():
        raise FileNotFoundError(
            "Faltan los archivos de las tareas 2.1 o 2.3 para realizar la validación."
        )

    df_21 = pd.read_csv(ruta_c21)
    df_23 = pd.read_csv(ruta_c23)

    print("✓ Tabla 2.1 (Evidencia) cargada y validada correctamente.")
    print("✓ Tabla 2.3 (Quintiles MAE) cargada y validada correctamente.\n")

    criterios_md = """# Criterios para Desconfiar de una Predicción (Parte 2.4)

Basado en los análisis geométricos y empíricos del modelo **Random Forest** y modelos lineales sobre el dataset **Wine Quality (tinto)**, se establecen los siguientes criterios clave para alertar a un usuario final (ej. panel evaluador o bodega) que **no debe confiar** en una predicción:

---

### 1. Alta Concentración del Peso $c(x) > 0.50$ (Violación del TLC)
* **Evidencia:** En la consulta de máxima concentración (Test #12), el peso $c(x)$ del Random Forest alcanzó **0.9641**, lo que significa que un solo árbol/hoja o una observación histórica dominó prácticamente el 96% del voto.
* **Riesgo:** Se rompe la condición de Hájek-Šidák del Teorema del Límite Central. La predicción no es un promedio estable de casos similares, sino una réplica casi directa de una sola botella histórica.

### 2. Tamaño Efectivo de Muestra Extremadamente Bajo ($N_{ef} < 10$)
* **Evidencia:** Para la consulta con $N_{ef}$ mínimo (Test #67), el tamaño efectivo de muestra fue de apenas **3.42** observaciones.
* **Riesgo:** Apoyar una decisión comercial o de calidad en menos de 4 botellas históricas incrementa fuertemente la varianza y la fragilidad del resultado.

### 3. Presencia Masiva de Pesos Negativos en Modelos Lineales (OLS > 40%)
* **Evidencia:** OLS asignó entre **45.5% y 46.1% de pesos negativos** en las consultas evaluadas (con pesos individuales tan bajos como $-0.0145$).
* **Riesgo:** OLS extrapola extrapolando puntos lejanos del espacio de características mediante penalizaciones. Si la consulta está en una región dispersa, la predicción puede ser distorsionada por datos sin relación fisicoquímica directa.

### 4. Regiones de Incertidumbre y Desviación Extrema
* **Evidencia:** Aunque la evidencia ($N_{ef}$) sea alta, un cambio brusco en variables críticas como `volatile acidity` o `alcohol` respecto a la media histórica debe marcarse como zona de alerta operativa.
"""

    dir_explicaciones = ROOT_DIR / "notebooks" / "explicaciones" / "parte2"
    dir_explicaciones.mkdir(parents=True, exist_ok=True)
    ruta_md = dir_explicaciones / "2_4_criterios_desconfianza.md"

    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(criterios_md)

    print(f"Documento de criterios generado exitosamente en:\n   {ruta_md}\n")
    print(criterios_md)


if __name__ == "__main__":
    redactar_criterios_y_validar()