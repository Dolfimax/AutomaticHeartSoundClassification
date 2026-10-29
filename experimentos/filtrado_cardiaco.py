
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cargar grabación cardíaca
archivo = "data/wav_example/a0001.wav"
audio, fs = sf.read(archivo)

# Convertir a mono si fuera necesario
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Reproducir el filtro original
def butter_bandpass(lowcut, hightcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = hightcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def butter_bandpass_filter(data, lowcut, hightcut, fs, order=5):
    b, a = butter_bandpass(lowcut, hightcut, fs, order)
    return lfilter(b, a, data)

# Aplicar filtrado de 25 a 400 Hz
audio_filtrado = butter_bandpass_filter(
    audio, 25, 400, fs, order=5
)

# Seleccionar 5 segundos para visualizar
muestras = int(5 * fs)
tiempo = np.arange(muestras) / fs

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(tiempo, audio[:muestras])
axes[0].set_title("Señal cardíaca original")
axes[0].set_ylabel("Amplitud digital")
axes[0].grid(True)

axes[1].plot(tiempo, audio_filtrado[:muestras])
axes[1].set_title("Señal filtrada - Butterworth 25-400 Hz")
axes[1].set_xlabel("Tiempo (segundos)")
axes[1].set_ylabel("Amplitud digital")
axes[1].grid(True)

plt.tight_layout()

salida = "experimentos/resultados/comparacion_filtrado.png"
plt.savefig(salida, dpi=150)
plt.close()

print("Frecuencia de muestreo:", fs, "Hz")
print("RMS original:", np.sqrt(np.mean(audio**2)))
print("RMS filtrado:", np.sqrt(np.mean(audio_filtrado**2)))
print("Gráfico guardado en:", salida)
