
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

from model.model import simple_cnn
from data_loader.data_loaders import HeartSoundDataSet

# =====================================
# CONFIGURACION
# =====================================

H5_PATH = "experimentos/resultados/mel_128_training_a.h5"

VAL_CSV = (
    "experimentos/preparacion_dataset/divisiones/val.csv"
)

CHECKPOINT = (
    "experimentos/resultados/cnn_mejor_modelo.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================
# CARGAR DATOS DE VALIDACION
# =====================================

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
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# =====================================
# CARGAR MODELO ENTRENADO
# =====================================

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

modelo = modelo.to(device)
modelo.eval()

# =====================================
# REALIZAR PREDICCIONES
# =====================================

etiquetas_reales = []
predicciones = []

with torch.no_grad():
    for features, labels in loader:

        features = features.float().to(device)

        salida = modelo(features)
        pred = salida.argmax(dim=1).cpu().numpy()

        etiquetas_reales.extend(labels.numpy())
        predicciones.extend(pred)

# =====================================
# CALCULAR METRICAS
# =====================================

matriz = confusion_matrix(
    etiquetas_reales,
    predicciones,
    labels=[0, 1]
)

print("\nMATRIZ DE CONFUSION")
print("Filas: clase real | Columnas: prediccion")
print("Clases: 0=Normal, 1=Anormal")
print(matriz)

print("\nREPORTE DE CLASIFICACION")

print(
    classification_report(
        etiquetas_reales,
        predicciones,
        labels=[0, 1],
        target_names=["Normal", "Anormal"],
        zero_division=0
    )
)

print(
    "Exactitud balanceada:",
    round(
        balanced_accuracy_score(
            etiquetas_reales,
            predicciones
        ),
        4
    )
)

print("\nDistribucion de predicciones:")
print(pd.Series(predicciones).value_counts().sort_index())

print("\nEvaluacion completada")
