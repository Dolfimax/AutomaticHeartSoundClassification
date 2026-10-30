
import os
import numpy as np
import pandas as pd
import torch
import matplotlib

# Permite guardar figuras dentro de Docker sin pantalla.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from model.model import simple_cnn
from data_loader.data_loaders import HeartSoundDataSet


# ======================================
# CONFIGURACION
# ======================================

H5_PATH = "experimentos/resultados/mel_128_training_a.h5"
VAL_CSV = "experimentos/preparacion_dataset/divisiones/val.csv"

CHECKPOINT = (
    "experimentos/resultados/"
    "cnn_experimento_con_ponderacion.pth"
)

RESULTADOS = "experimentos/resultados"
os.makedirs(RESULTADOS, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo:", device)


# ======================================
# CARGAR DATOS
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
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ======================================
# CARGAR MODELO
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

modelo = modelo.to(device)
modelo.eval()


# ======================================
# EXTRAER PUNTUACIONES
# ======================================

etiquetas = []
puntuaciones = []

with torch.no_grad():

    for features, labels in loader:

        features = features.float().to(device)

        salida = modelo(features)

        # Columna 1: clase anormal.
        valores = salida[:, 1].cpu().numpy()

        etiquetas.extend(labels.cpu().numpy())
        puntuaciones.extend(valores)


etiquetas = np.asarray(etiquetas, dtype=int)
puntuaciones = np.asarray(puntuaciones)

assert len(etiquetas) == 41
assert np.all(np.isfinite(puntuaciones))


# ======================================
# METRICAS GENERALES
# ======================================

roc_auc = roc_auc_score(
    etiquetas,
    puntuaciones
)

average_precision = average_precision_score(
    etiquetas,
    puntuaciones
)

predicciones_05 = (
    puntuaciones >= 0.50
).astype(int)

matriz_05 = confusion_matrix(
    etiquetas,
    predicciones_05,
    labels=[0, 1]
)

balanced_05 = balanced_accuracy_score(
    etiquetas,
    predicciones_05
)

print("\nRESULTADOS GENERALES")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))
print("Exactitud balanceada (0.50):", round(balanced_05, 4))

print("\nMatriz de confusion (umbral 0.50):")
print(matriz_05)


# ======================================
# GRAFICO 1: HISTOGRAMA
# ======================================

plt.figure(figsize=(9, 5))

plt.hist(
    puntuaciones[etiquetas == 0],
    bins=np.linspace(0, 1, 16),
    alpha=0.6,
    label="Normal (real)"
)

plt.hist(
    puntuaciones[etiquetas == 1],
    bins=np.linspace(0, 1, 16),
    alpha=0.6,
    label="Anormal (real)"
)

plt.axvline(
    0.50,
    linestyle="--",
    label="Umbral 0.50"
)

plt.xlabel("Puntuacion asignada a la clase anormal")
plt.ylabel("Numero de grabaciones")
plt.title("Distribucion de puntuaciones - CNN ponderada")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    f"{RESULTADOS}/histograma_probabilidades.png",
    dpi=200
)

plt.close()


# ======================================
# GRAFICO 2: CURVA ROC
# ======================================

fpr, tpr, _ = roc_curve(
    etiquetas,
    puntuaciones
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Referencia aleatoria"
)

plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC - CNN ponderada")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    f"{RESULTADOS}/curva_roc.png",
    dpi=200
)

plt.close()


# ======================================
# GRAFICO 3: PRECISION-RECALL
# ======================================

precision, recall, _ = precision_recall_curve(
    etiquetas,
    puntuaciones
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.axhline(
    np.mean(etiquetas),
    linestyle="--",
    label="Prevalencia de clase anormal"
)

plt.xlabel("Sensibilidad (Recall)")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall - CNN ponderada")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    f"{RESULTADOS}/curva_precision_recall.png",
    dpi=200
)

plt.close()


# ======================================
# GRAFICO 4: METRICAS POR UMBRAL
# ======================================

umbrales = np.linspace(0, 1, 101)

sensibilidades = []
especificidades = []
balanced_accuracies = []

for umbral in umbrales:

    pred = (puntuaciones >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        etiquetas,
        pred,
        labels=[0, 1]
    ).ravel()

    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)

    sensibilidades.append(sensibilidad)
    especificidades.append(especificidad)
    balanced_accuracies.append(
        (sensibilidad + especificidad) / 2
    )

plt.figure(figsize=(9, 5))

plt.plot(
    umbrales,
    sensibilidades,
    label="Sensibilidad"
)

plt.plot(
    umbrales,
    especificidades,
    label="Especificidad"
)

plt.plot(
    umbrales,
    balanced_accuracies,
    label="Exactitud balanceada",
    linewidth=2
)

plt.axvline(
    0.50,
    linestyle="--",
    label="Umbral actual 0.50"
)

plt.xlabel("Umbral de clasificacion")
plt.ylabel("Valor de la metrica")
plt.title("Sensibilidad y especificidad segun umbral")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    f"{RESULTADOS}/metricas_por_umbral.png",
    dpi=200
)

plt.close()



# ======================================
# COMPARACION DE UMBRALES
# ======================================

print("\nCOMPARACION DE UMBRALES")
print(
    "Umbral | Sensibilidad | Especificidad | "
    "Balanced Acc | FN | FP"
)

for umbral in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:

    pred = (puntuaciones >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        etiquetas,
        pred,
        labels=[0, 1]
    ).ravel()

    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)
    balanced = (sensibilidad + especificidad) / 2

    print(
        f"{umbral:.2f}    | "
        f"{sensibilidad:.4f}       | "
        f"{especificidad:.4f}        | "
        f"{balanced:.4f}       | "
        f"{fn:2d} | {fp:2d}"
    )


# ======================================
# FINALIZACION
# ======================================

print("\nGraficos guardados en:", RESULTADOS)

print("1. histograma_probabilidades.png")
print("2. curva_roc.png")
print("3. curva_precision_recall.png")
print("4. metricas_por_umbral.png")

print("\nAnalisis completado")
