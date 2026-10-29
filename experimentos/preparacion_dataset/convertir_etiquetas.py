
from pathlib import Path
import pandas as pd

# Archivos de entrada y salida
origen = Path("data/physionet/training-a/REFERENCE.csv")
destino = Path("experimentos/preparacion_dataset/etiquetas_convertidas.csv")

# Leer etiquetas originales de PhysioNet
df = pd.read_csv(
    origen,
    header=None,
    names=["filename", "label_original"]
)

# Verificar las etiquetas encontradas
etiquetas_validas = {-1, 1}

if not set(df["label_original"].unique()).issubset(etiquetas_validas):
    raise ValueError("Se encontraron etiquetas inesperadas")

# Convertir al formato de PyTorch
conversion = {
    -1: 0,  # Normal
     1: 1   # Anormal
}

df["label"] = df["label_original"].map(conversion)

# Agregar extensión a los nombres
df["filename"] = df["filename"] + ".wav"

# Crear directorio de salida
destino.parent.mkdir(parents=True, exist_ok=True)

# Guardar sin modificar el archivo original
df.to_csv(destino, index=False)

print("Conversión completada")
print("Total de registros:", len(df))
print("Distribución de clases:")
print(df["label"].value_counts().sort_index())
print("Archivo generado:", destino)
