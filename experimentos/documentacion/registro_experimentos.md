# Registro experimental — Clasificación de sonidos cardíacos mediante CNN

**Proyecto:** Investigación sobre detección de anomalías cardíacas mediante inteligencia artificial  
**Investigadores:** Adolfo Darío Saavedra y Luciano Pescio  
**Estado:** Investigación exploratoria

## 1. Dataset

- Fuente: PhysioNet/CinC Challenge 2016, subconjunto training-a.
- Total: 409 grabaciones.
- Entrenamiento: 327 grabaciones.
- Validación: 41 grabaciones.
- Prueba reservada: 41 grabaciones.
- Clases: 0 = Normal; 1 = Anormal.
- División estratificada, semilla 42.
- Limitación: no se ha verificado independencia entre pacientes.

## 2. Procesamiento

- Filtrado de señales cardíacas.
- Extracción de características Log-Mel con 128 bandas.
- Arquitectura CNN del repositorio base AutomaticHeartSoundClassification.
- Entrenamiento mediante PyTorch en contenedor Docker y GPU NVIDIA GTX 1650.

## 3. Experimento E01 — CNN sin ponderación

- Mejor época: 19.
- Pérdida de validación: 0,5659.
- Exactitud general: 75,61 %.
- Exactitud balanceada: 54,55 %.
- Sensibilidad clase anormal: 100 % (30/30).
- Especificidad clase normal: 9,09 % (1/11).
- Matriz de confusión: [[1, 10], [0, 30]].
- Checkpoint: cnn_experimento_sin_ponderacion.pth.

## 4. Experimento E02 — CNN con ponderación

- Mejor época: 27.
- Pérdida de validación: 0,6746.
- Exactitud general: 70,73 %.
- Exactitud balanceada: 68,48 %.
- Sensibilidad clase anormal: 73,33 % (22/30).
- Especificidad clase normal: 63,64 % (7/11).
- Matriz de confusión: [[7, 4], [8, 22]].
- Pesos de clase: normal = 327/(2×94); anormal = 327/(2×233).
- Checkpoint: cnn_experimento_con_ponderacion.pth.

## 5. Interpretación preliminar

La ponderación de clases incrementó la exactitud balanceada en 13,93 puntos porcentuales respecto al entrenamiento sin ponderación, pero redujo la sensibilidad para identificar grabaciones anormales.

Los resultados corresponden a un único conjunto pequeño de validación y no constituyen evidencia de desempeño diagnóstico.

## 6. Limitaciones y próximas investigaciones

- Verificar que la separación de datos sea independiente por paciente.
- Evaluar la estabilidad mediante múltiples semillas y particiones.
- Investigar el impacto de los pesos de clase y los umbrales de decisión.
- Incorporar métricas ROC-AUC y PR-AUC.
- Reservar el conjunto de prueba para una evaluación final definida previamente.
- Validar la generalización en conjuntos externos.
- Investigar la relación entre anomalías acústicas y valvulopatías específicas mediante etiquetas clínicas apropiadas.