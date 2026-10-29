
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuración reproducible
SEMILLA = 42

origen = Path(
    "experimentos/preparacion_dataset/etiquetas_convertidas.csv"
)

destino = Path(
    "experimentos/preparacion_dataset/divisiones"
)

destino.mkdir(parents=True, exist_ok=True)

# Cargar etiquetas
df = pd.read_csv(origen)

# Comprobar integridad
assert df["filename"].is_unique
assert set(df["label"].unique()) == {0, 1}

# 80 % entrenamiento, 20 % temporal
train, temporal = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=SEMILLA
)

# Dividir el 20 % restante en validación y prueba
val, test = train_test_split(
    temporal,
    test_size=0.50,
    stratify=temporal["label"],
    random_state=SEMILLA
)

# Verificar que no haya grabaciones repetidas
assert set(train["filename"]).isdisjoint(val["filename"])
assert set(train["filename"]).isdisjoint(test["filename"])
assert set(val["filename"]).isdisjoint(test["filename"])

assert len(train) + len(val) + len(test) == len(df)

# Guardar resultados
for nombre, datos in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    ruta = destino / f"{nombre}.csv"
    datos.to_csv(ruta, index=False)

    print(f"\n{nombre.upper()}: {len(datos)} grabaciones")
    print(datos["label"].value_counts().sort_index())

print("\nDivisión completada correctamente")
print("Semilla utilizada:", SEMILLA)
