
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx


# ======================================
# CONFIGURACION
# ======================================

MODELO_ONNX = (
    "experimentos/resultados/cnn_cardiaca.onnx"
)

CALIBRACION = (
    "experimentos/resultados/calibracion_esp32.npz"
)

MODELO_ESPDL = (
    "experimentos/resultados/cnn_cardiaca_esp32s3.espdl"
)

TARGET = "esp32s3"
DEVICE = "cpu"


# ======================================
# CARGAR DATOS DE CALIBRACION
# ======================================

datos = np.load(CALIBRACION)

entradas = datos["entradas"]

assert entradas.shape == (32, 384, 333)
assert entradas.dtype == np.float32
assert np.isfinite(entradas).all()

tensor = torch.from_numpy(
    entradas.copy()
)

dataset = TensorDataset(tensor)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


def collate_fn(batch):
    return batch[0].to(DEVICE)


# ======================================
# CUANTIZACION PARA ESP32-S3
# ======================================

print("\nCONVERSION CNN CARDIACA A ESP-DL")
print("Modelo original:", MODELO_ONNX)
print("Plataforma:", TARGET)
print("Cuantizacion: INT8")
print("Entradas de calibracion:", len(dataset))

espdl_quantize_onnx(
    onnx_import_file=MODELO_ONNX,
    espdl_export_file=MODELO_ESPDL,
    calib_dataloader=loader,
    calib_steps=32,
    input_shape=[1, 384, 333],
    target=TARGET,
    num_of_bits=8,
    collate_fn=collate_fn,
    device=DEVICE,
    error_report=True,
    skip_export=False,
    export_test_values=False,
    verbose=1
)


# ======================================
# VERIFICAR ARCHIVO GENERADO
# ======================================

archivo = Path(MODELO_ESPDL)

if not archivo.is_file():
    raise RuntimeError(
        "La conversion termino sin generar el archivo ESP-DL"
    )

print("\nConversion completada")
print("Archivo:", archivo)
print(
    "Tamano:",
    round(archivo.stat().st_size / 1024, 2),
    "KiB"
)
