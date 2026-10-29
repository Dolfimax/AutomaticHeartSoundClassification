
import numpy as np
import h5py
import librosa
import torch

from model.model import simple_cnn

# Cargar características Log-Mel
archivo = "experimentos/resultados/logmel_a0001.h5"

with h5py.File(archivo, "r") as h5:
    feature = h5["a0001"][()]

print("Log-Mel original:", feature.shape)

# Normalización original del DataLoader
media = np.mean(feature)
desviacion = np.std(feature)

feature = (feature - media) / desviacion

# Calcular derivadas temporales
delta = librosa.feature.delta(feature)
delta_2 = librosa.feature.delta(delta)

# Concatenar tres grupos de características
feature = np.concatenate(
    (feature, delta, delta_2),
    axis=0
)

# Seleccionar segmento central de 5 segundos
cycle_len = int(5 * 1000 / 15)
num_frames = feature.shape[1]

if num_frames >= cycle_len:
    inicio = (num_frames - cycle_len) // 2
    feature = feature[:, inicio:inicio + cycle_len]
else:
    feature = np.pad(
        feature,
        ((0, 0), (0, cycle_len - num_frames)),
        mode="wrap"
    )

print("Características preparadas:", feature.shape)

# Convertir a tensor PyTorch
entrada = torch.tensor(
    feature,
    dtype=torch.float32
).unsqueeze(0)

print("Tensor de entrada:", entrada.shape)

# Instanciar arquitectura CNN original
modelo = simple_cnn(
    num_classes=2,
    in_channel=3
)

modelo.eval()

# Prueba de propagación hacia adelante
with torch.no_grad():
    salida = modelo(entrada)

print("Dimensiones de salida:", salida.shape)
print("Prueba de arquitectura completada")
print("ATENCION: modelo sin entrenamiento.")
print("Los valores de salida NO son diagnósticos.")
