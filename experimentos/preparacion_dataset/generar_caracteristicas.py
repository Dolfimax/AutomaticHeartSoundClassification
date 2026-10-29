
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from utils.util import read_audio
from utils.audio_feature_extractor import LogMelExtractor

# Rutas de entrada y salida
directorio_audio = Path("data/physionet/training-a")

archivo_etiquetas = Path(
    "experimentos/preparacion_dataset/etiquetas_convertidas.csv"
)

archivo_salida = Path(
    "experimentos/resultados/mel_128_training_a.h5"
)

# Leer etiquetas convertidas
df = pd.read_csv(archivo_etiquetas)

# Validar datos
assert len(df) == 409, "Cantidad inesperada de registros"
assert df["filename"].is_unique, "Hay nombres duplicados"
assert set(df["label"].unique()) == {0, 1}, "Etiquetas incorrectas"

archivo_salida.parent.mkdir(parents=True, exist_ok=True)

# Generar características
with h5py.File(archivo_salida, "w") as h5:
    for filename in tqdm(df["filename"], desc="Procesando audios"):

        ruta_audio = directorio_audio / filename

        if not ruta_audio.exists():
            raise FileNotFoundError(ruta_audio)

        # Lectura y filtrado original (25-400 Hz)
        audio, fs = read_audio(str(ruta_audio), filter=True)

        # Extracción Log-Mel original
        caracteristicas = LogMelExtractor(
            audio,
            fs,
            mel_bins=128,
            hoplen=15,
            winlen=25,
            log=True,
            snv=False
        )

        # Validar resultado
        if not np.isfinite(caracteristicas).all():
            raise ValueError(
                f"Valores no válidos en {filename}"
            )

        # Guardar con el nombre del archivo
        h5.create_dataset(
            filename,
            data=caracteristicas.astype(np.float32)
        )

print("Extracción completada")
print("Grabaciones procesadas:", len(df))
print("Archivo generado:", archivo_salida)
