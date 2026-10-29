
import os
import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Archivo de audio cardíaco
archivo = "data/wav_example/a0001.wav"

# Cargar señal sin alterar el muestreo original
audio, sr = librosa.load(archivo, sr=None)

# Vector temporal
tiempo = np.arange(len(audio)) / sr
# Seleccionar los primeros 5 segundos
duracion_segmento = 5
audio = audio[:int(duracion_segmento * sr)]
tiempo = np.arange(len(audio)) / sr

# Crear figura
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Gráfico 1: Forma de onda
axes[0].plot(tiempo, audio, linewidth=0.6)
axes[0].set_title("Fonocardiograma - a0001.wav")
axes[0].set_xlabel("Tiempo (segundos)")
axes[0].set_ylabel("Amplitud digital")
axes[0].grid(True)

# Gráfico 2: Espectrograma STFT
stft = librosa.stft(audio, n_fft=1024, hop_length=128)
espectrograma = librosa.amplitude_to_db(
    np.abs(stft), ref=np.max
)

img = librosa.display.specshow(
    espectrograma,
    sr=sr,
    hop_length=128,
    x_axis="time",
    y_axis="linear",
    ax=axes[1]
)

axes[1].set_title("Espectrograma del sonido cardíaco")
axes[1].set_ylim(0, 1000)
fig.colorbar(img, ax=axes[1], format="%+d dB")

# Guardar resultados
os.makedirs("experimentos/resultados", exist_ok=True)

salida = "experimentos/resultados/analisis_a0001.png"

plt.tight_layout()
plt.savefig(salida, dpi=150)
plt.close()

print("Análisis completado")
print("Frecuencia de muestreo:", sr, "Hz")
print("Gráfico guardado en:", salida)
