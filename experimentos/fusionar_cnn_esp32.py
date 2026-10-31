
import copy

import numpy as np
import torch

from torch.nn.utils.fusion import fuse_conv_bn_eval
from model.model import simple_cnn


# ======================================
# CONFIGURACION
# ======================================

CHECKPOINT = (
    "experimentos/resultados/"
    "cnn_experimento_con_ponderacion.pth"
)

SALIDA = (
    "experimentos/resultados/"
    "cnn_cardiaca_fusionada.pth"
)

torch.manual_seed(42)


# ======================================
# CARGAR MODELO ENTRENADO
# ======================================

modelo_original = simple_cnn(
    num_classes=2,
    in_channel=3
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

modelo_original.load_state_dict(
    checkpoint["state_dict"]
)

modelo_original.eval()

# Crear copia para conservar el original
modelo_fusionado = copy.deepcopy(modelo_original)


# ======================================
# FUSIONAR CONVOLUCIONES Y BATCHNORM
# ======================================

for conv, bn in [
    ("conv1", "bn1"),
    ("conv2", "bn2"),
    ("conv3", "bn3")
]:
    convolucion = getattr(modelo_fusionado, conv)
    normalizacion = getattr(modelo_fusionado, bn)

    convolucion_fusionada = fuse_conv_bn_eval(
        convolucion,
        normalizacion
    )

    setattr(
        modelo_fusionado,
        conv,
        convolucion_fusionada
    )

    setattr(
        modelo_fusionado,
        bn,
        torch.nn.Identity()
    )

modelo_fusionado.eval()


# ======================================
# COMPARAR SALIDAS
# ======================================

entrada = torch.randn(
    1, 384, 333,
    dtype=torch.float32
)

with torch.no_grad():

    salida_original = modelo_original(entrada)
    salida_fusionada = modelo_fusionado(entrada)

original = salida_original.numpy()
fusionada = salida_fusionada.numpy()

diferencia = np.abs(original - fusionada)

equivalentes = np.allclose(
    original,
    fusionada,
    rtol=1e-4,
    atol=1e-5
)

print("\nRESULTADOS DE FUSION")

print("Salida original:", original)
print("Salida fusionada:", fusionada)

print(
    "Diferencia absoluta maxima:",
    float(np.max(diferencia))
)

print("Equivalencia numerica:", equivalentes)

print(
    "Clase original:",
    int(np.argmax(original, axis=1)[0])
)

print(
    "Clase fusionada:",
    int(np.argmax(fusionada, axis=1)[0])
)


# ======================================
# GUARDAR MODELO FUSIONADO
# ======================================

if equivalentes:

    torch.save(
        {
            "state_dict": modelo_fusionado.state_dict(),
            "modelo": modelo_fusionado,
            "origen": CHECKPOINT,
            "fusion": "Conv2D + BatchNorm2D",
            "semilla": 42
        },
        SALIDA
    )

    print("\nModelo fusionado guardado:", SALIDA)

else:

    print(
        "\nNo se guarda el modelo: "
        "la equivalencia numerica no fue confirmada."
    )

print("\nProceso finalizado")
