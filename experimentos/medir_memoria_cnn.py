
import torch

from model.model import simple_cnn

# Crear el modelo para evaluar su arquitectura
modelo = simple_cnn(
    num_classes=2,
    in_channel=3
)

modelo.eval()

# Entrada con las dimensiones reales
entrada = torch.randn(1, 384, 333)

print("\nANALISIS DE MEMORIA DE LA CNN")
print("Entrada:", tuple(entrada.shape))

def medir_memoria(nombre):

    def hook(modulo, entradas, salida):

        if isinstance(salida, torch.Tensor):

            memoria = (
                salida.numel()
                * salida.element_size()
                / (1024 ** 2)
            )

            print(
                f"{nombre}: "
                f"{tuple(salida.shape)} | "
                f"{memoria:.3f} MiB"
            )

    return hook


# Registrar mediciones por capa
for nombre, modulo in modelo.named_children():

    modulo.register_forward_hook(
        medir_memoria(nombre)
    )


# Ejecutar inferencia sin entrenamiento
with torch.no_grad():
    modelo(entrada)

print("\nAnalisis completado")
