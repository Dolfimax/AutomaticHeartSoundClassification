
import numpy as np
import torch
import onnxruntime as ort

from model.model import simple_cnn

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

torch.manual_seed(42)

# ======================================
# MODELO PYTORCH
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

# Entrada sintetica compartida
entrada = torch.randn(
    1, 384, 333,
    dtype=torch.float32
)

with torch.no_grad():
    salida_pytorch = modelo(entrada).numpy()

# ======================================
# MODELO ONNX
# ======================================

sesion = ort.InferenceSession(
    ARCHIVO_ONNX,
    providers=["CPUExecutionProvider"]
)

nombre_entrada = sesion.get_inputs()[0].name

salida_onnx = sesion.run(
    None,
    {nombre_entrada: entrada.numpy()}
)[0]

# ======================================
# COMPARACION
# ======================================

diferencia = np.abs(
    salida_pytorch - salida_onnx
)

print("\nCOMPARACION PYTORCH - ONNX")

print("Salida PyTorch:", salida_pytorch)
print("Salida ONNX:", salida_onnx)

print(
    "Diferencia absoluta maxima:",
    float(np.max(diferencia))
)

print(
    "Diferencia absoluta media:",
    float(np.mean(diferencia))
)

coinciden = np.allclose(
    salida_pytorch,
    salida_onnx,
    rtol=1e-4,
    atol=1e-5
)

print("Salidas numericamente equivalentes:", coinciden)

print(
    "Clase predicha PyTorch:",
    int(np.argmax(salida_pytorch, axis=1)[0])
)

print(
    "Clase predicha ONNX:",
    int(np.argmax(salida_onnx, axis=1)[0])
)

print("\nComparacion finalizada")
