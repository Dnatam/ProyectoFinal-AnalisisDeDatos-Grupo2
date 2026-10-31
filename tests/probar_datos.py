from src.comunes.datos import cargar_california, cargar_wine

california = cargar_california()
wine = cargar_wine()

print("=== CALIFORNIA HOUSING ===")
print("Entrenamiento:", california["X_train"].shape)
print("Prueba:", california["X_test"].shape)

print("\n=== WINE QUALITY RED ===")
print("Entrenamiento:", wine["X_train"].shape)
print("Prueba:", wine["X_test"].shape)

print("\nDesviaciones originales:")
print(wine["desviaciones_originales"])

print("\nPromedio de variables estandarizadas:")
print(wine["X_train_std"].mean(axis=0))