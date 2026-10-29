
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score


# Componentes originales del repositorio
from model.model import simple_cnn
from data_loader.data_loaders import HeartSoundDataSet

# ==========================================
# CONFIGURACION EXPERIMENTAL
# ==========================================

SEMILLA = 42
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0002
PACIENCIA = 10

H5_PATH = "experimentos/resultados/mel_128_training_a.h5"

TRAIN_CSV = (
    "experimentos/preparacion_dataset/divisiones/train.csv"
)

VAL_CSV = (
    "experimentos/preparacion_dataset/divisiones/val.csv"
)

MODELO_SALIDA = Path(
    "experimentos/resultados/cnn_mejor_modelo.pth"
)

# ==========================================
# REPRODUCIBILIDAD
# ==========================================

random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# DISPOSITIVO
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo utilizado:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# CARGAR DIVISIONES EXISTENTES
# ==========================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

assert set(train_df["filename"]).isdisjoint(
    set(val_df["filename"])
)

print("Grabaciones entrenamiento:", len(train_df))
print("Grabaciones validacion:", len(val_df))

# ==========================================
# DATASETS ORIGINALES
# ==========================================

train_dataset = HeartSoundDataSet(
    fea_path=H5_PATH,
    labels=train_df,
    duration=5,
    training=True,
    delta=True,
    norm=True
)

val_dataset = HeartSoundDataSet(
    fea_path=H5_PATH,
    labels=val_df,
    duration=5,
    training=False,
    delta=True,
    norm=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# MODELO CNN ORIGINAL
# ==========================================

modelo = simple_cnn(
    num_classes=2,
    in_channel=3
).to(device)

# El modelo original ya aplica Softmax.
# NLLLoss recibe log-probabilidades.
criterio = nn.NLLLoss()

optimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=LEARNING_RATE,
    amsgrad=True
)

# ==========================================
# ENTRENAMIENTO Y VALIDACION
# ==========================================

mejor_loss = float("inf")
mejor_val_balanced_acc = -1.0
epocas_sin_mejora = 0

MODELO_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

for epoch in range(EPOCHS):

    modelo.train()
    perdida_train = 0.0
    correctos_train = 0
    total_train = 0

    for features, labels in train_loader:

        features = features.float().to(device)
        labels = labels.long().to(device)

        optimizador.zero_grad()

        predicciones = modelo(features)

        loss = criterio(
            torch.log(predicciones.clamp_min(1e-8)),
            labels
        )

        loss.backward()
        optimizador.step()

        n = labels.size(0)
        perdida_train += loss.item() * n

        correctos_train += (
            predicciones.argmax(dim=1) == labels
        ).sum().item()

        total_train += n

    # ======================================
    # VALIDACION
    # ======================================

    modelo.eval()
    perdida_val = 0.0
    correctos_val = 0
    total_val = 0
    
    etiquetas_val = []
    predicciones_val = []


    with torch.no_grad():

        for features, labels in val_loader:

            features = features.float().to(device)
            labels = labels.long().to(device)

            predicciones = modelo(features)

            loss = criterio(
                torch.log(predicciones.clamp_min(1e-8)),
                labels
            )

            n = labels.size(0)
            perdida_val += loss.item() * n

            correctos_val += (
                predicciones.argmax(dim=1) == labels
            ).sum().item()

            total_val += n
            
            etiquetas_val.extend(
                labels.cpu().numpy()
            )

            predicciones_val.extend(
                predicciones.argmax(dim=1).cpu().numpy()
            )


    train_loss = perdida_train / total_train
    val_loss = perdida_val / total_val

    train_acc = correctos_train / total_train
    val_acc = correctos_val / total_val

    
    val_balanced_acc = balanced_accuracy_score(
        etiquetas_val,
        predicciones_val
    )


    
    print(
        f"Epoca {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Balanced Acc: {val_balanced_acc:.4f}",
        flush=True
    )


    # ======================================
    # GUARDAR MEJOR MODELO
    # ======================================

    
    if (
        val_balanced_acc > mejor_val_balanced_acc
        or (
            val_balanced_acc == mejor_val_balanced_acc
            and val_loss < mejor_loss
        )
    ):
        mejor_val_balanced_acc = val_balanced_acc
        mejor_loss = val_loss
        epocas_sin_mejora = 0


        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": modelo.state_dict(),
                "optimizer": optimizador.state_dict(),
                "val_loss": val_loss,
                "val_balanced_acc": val_balanced_acc,
                "seed": SEMILLA
            },
            MODELO_SALIDA
        )

        print("Mejor modelo guardado", flush=True)

    else:
        epocas_sin_mejora += 1

    if epocas_sin_mejora >= PACIENCIA:
        print("Detencion temprana activada")
        break

print("Entrenamiento finalizado")
print("Modelo guardado en:", MODELO_SALIDA)
