
from pathlib import Path

import torch
import onnx

from model.model import simple_cnn


# ======================================
# CONFIGURACION
# ======================================

CHECKPOINT = Path(
    "experimentos/resultados/"
    "cnn_experimento_con_ponderacion.pth"
)

SALIDA = Path(
    "experimentos/resultados/"
    "cnn_cardiaca.onnx"
)

SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================
# CARGAR CNN ENTRENADA
# ======================================

modelo = simple_cnn(
    num_classes=2,
    in_channel=3
)

checkpoint = torch.load(
    str(CHECKPOINT),
    map_location="cpu"
)

modelo.load_state_dict(
    checkpoint["state_dict"]
)

modelo.eval()

print("Modelo entrenado cargado")
print("Epoca:", checkpoint["epoch"])


# ======================================
# PREPARAR ENTRADA
# ======================================

entrada = torch.randn(
    1, 384, 333,
    dtype=torch.float32
)


# ======================================
# EXPORTAR A ONNX
# ======================================

torch.onnx.export(
    modelo,
    entrada,
    str(SALIDA),
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["caracteristicas_audio"],
    output_names=["clasificacion"]
)

print("Exportacion completada:", SALIDA)


# ======================================
# VERIFICAR ESTRUCTURA ONNX
# ======================================

modelo_onnx = onnx.load(str(SALIDA))
onnx.checker.check_model(modelo_onnx)

print("Estructura ONNX verificada")
print(
    "Tamano del archivo:",
    round(SALIDA.stat().st_size / 1024, 2),
    "KiB"
)
