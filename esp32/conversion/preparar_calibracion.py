
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.data_loaders import HeartSoundDataSet


# ======================================
# CONFIGURACION
# ======================================

TRAIN_CSV = (
    "experimentos/preparacion_dataset/"
    "divisiones/train.csv"
)

H5_PATH = (
    "experimentos/resultados/"
    "mel_128_training_a.h5"
)

SALIDA = Path(
    "experimentos/resultados/"
    "calibracion_esp32.npz"
)


# ======================================
# SELECCIONAR GRABACIONES
# ======================================

df = pd.read_csv(TRAIN_CSV)

# Muestra reproducible y aproximadamente
# proporcional a las clases del entrenamiento.

normales = df[df["label"] == 0].sample(
    n=9,
    random_state=42
)

anormales = df[df["label"] == 1].sample(
    n=23,
    random_state=42
)

calibracion = pd.concat(
    [normales, anormales],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)


# ======================================
# PREPARAR ENTRADAS DE LA CNN
# ======================================

dataset = HeartSoundDataSet(
    fea_path=H5_PATH,
    labels=calibracion,
    duration=5,
    training=False,
    delta=True,
    norm=True
)

entradas = []

for i in range(len(dataset)):
    caracteristicas, etiqueta = dataset[i]

    entrada = np.asarray(
        caracteristicas,
        dtype=np.float32
    )

    assert entrada.shape == (384, 333)
    assert np.all(np.isfinite(entrada))

    entradas.append(entrada)

entradas = np.stack(entradas)


# ======================================
# GUARDAR CALIBRACION
# ======================================

SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.savez_compressed(
    SALIDA,
    entradas=entradas
)

print("Calibracion preparada")
print("Grabaciones:", len(entradas))
print("Dimensiones:", entradas.shape)
print("Formato:", entradas.dtype)
print("Archivo:", SALIDA)
