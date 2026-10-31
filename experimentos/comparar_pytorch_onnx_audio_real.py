
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from torch.utils.data import DataLoader
from model.model import simple_cnn
from data_loader.data_loaders import HeartSoundDataSet


# ======================================
# CONFIGURACION
# ======================================

CHECKPOINT = (
    "experimentos/resultados/"
    "cnn_experimento_con_ponderacion.pth"
)

ARCHIVO_ONNX = (
    "experimentos/resultados/cnn_cardiaca.onnx"
)

H5_PATH = (
    "experimentos/resultados/mel_128_training_a.h5"
)

VAL_CSV = (
    "experimentos/preparacion_dataset/divisiones/val.csv"
)

torch.set_num_threads(1)


# ======================================
# CARGAR GRABACIONES DE VALIDACION
# ======================================

df = pd.read_csv(VAL_CSV)

dataset = HeartSoundDataSet(
    fea_path=H5_PATH,
    labels=df,
    duration=5,
    training=False,
    delta=True,
    norm=True
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Grabaciones a evaluar:", len(dataset))


# ======================================
# CARGAR MODELO PYTORCH
# ======================================

modelo = simple_cnn(
    num_classes=2,
    in_channel=3
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

modelo.load_state_dict(
    checkpoint["state_dict"]
)

modelo.eval()


# ======================================
# CARGAR MODELO ONNX
# ======================================

sesion = ort.InferenceSession(
    ARCHIVO_ONNX,
    providers=["CPUExecutionProvider"]
)

nombre_entrada = sesion.get_inputs()[0].name


# ======================================
# COMPARAR PREDICCIONES
# ======================================

diferencias = []
coincidencias = 0
total = 0

with torch.no_grad():

    for features, labels in loader:

        entrada = features.float().cpu()

        salida_torch = modelo(entrada).numpy()

        salida_onnx = sesion.run(
            None,
            {
                nombre_entrada: entrada.numpy()
            }
        )[0]

        diferencia = np.max(
            np.abs(salida_torch - salida_onnx)
        )

        diferencias.append(float(diferencia))

        clase_torch = int(
            np.argmax(salida_torch, axis=1)[0]
        )

        clase_onnx = int(
            np.argmax(salida_onnx, axis=1)[0]
        )

        if clase_torch == clase_onnx:
            coincidencias += 1

        total += 1


# ======================================
# RESULTADOS
# ======================================

print("\nCOMPARACION PYTORCH - ONNX")
print("Entradas reales evaluadas:", total)

print(
    "Diferencia absoluta maxima:",
    max(diferencias)
)

print(
    "Diferencia maxima media:",
    np.mean(diferencias)
)

print(
    "Predicciones coincidentes:",
    coincidencias,
    "de",
    total
)

print(
    "Concordancia:",
    round(100 * coincidencias / total, 2),
    "%"
)

print(
    "Equivalencia numerica (atol=1e-5):",
    max(diferencias) <= 1e-5
)

print("\nComparacion completada")
