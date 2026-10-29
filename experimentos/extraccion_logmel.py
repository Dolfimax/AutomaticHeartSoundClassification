
import os
import sys
import numpy as np
import h5py

# Acceder al directorio principal del proyecto
sys.path.insert(0, os.path.abspath("."))

# Importar funciones originales del repositorio
from utils.util import read_audio
from utils.audio_feature_extractor import LogMelExtractor

# Archivo de prueba
archivo = "data/wav_example/a0001.wav"

# Leer y filtrar audio con el método original
audio, fs = read_audio(archivo, filter=True)

# Extraer características Log-Mel
caracteristicas = LogMelExtractor(
    audio,
    fs,
    mel_bins=128,
    hoplen=15,
    winlen=25,
    log=True,
    snv=False
)

# Crear carpeta para resultados
os.makedirs("experimentos/resultados", exist_ok=True)

# Guardar características en HDF5
salida = "experimentos/resultados/logmel_a0001.h5"

with h5py.File(salida, "w") as archivo_h5:
    archivo_h5.create_dataset(
        "a0001",
        data=caracteristicas
    )

print("Extracción Log-Mel completada")
print("Frecuencia de muestreo:", fs, "Hz")
print("Dimensiones:", caracteristicas.shape)
print("Archivo generado:", salida)
