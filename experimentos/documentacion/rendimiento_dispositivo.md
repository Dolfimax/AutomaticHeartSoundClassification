# Caracterización computacional para un dispositivo de análisis de sonidos cardíacos

**Proyecto:** Estudio de factibilidad de un dispositivo de bajo costo para el análisis de sonidos cardíacos mediante inteligencia artificial.

**Investigadores:** Adolfo Darío Saavedra y Luciano Pescio.

**Estado:** Caracterización computacional preliminar.

## 1. Objetivo

Evaluar los requisitos computacionales de una arquitectura de red neuronal convolucional existente para determinar su viabilidad de implementación en un dispositivo económico capaz de analizar sonidos cardíacos.

Esta etapa se concentra en la implementación del modelo y en su compatibilidad con plataformas de procesamiento de bajo costo. No contempla el desarrollo de una nueva arquitectura de clasificación.

## 2. Modelo utilizado

- Repositorio de referencia: SiyuLou/AutomaticHeartSoundClassification.
- Arquitectura: `simple_cnn`.
- Clases de salida: normal y anormal.
- Número de parámetros: 25.954.
- Precisión de los pesos: FP32.
- Memoria estimada de pesos: 0,099 MiB.
- Dimensiones de entrada: 384 × 333.
- Tipo de entrada: float32.
- Memoria de entrada: 0,488 MiB.

La estimación de memoria de los pesos no incluye las activaciones intermedias, las estructuras internas del entorno de inferencia ni los buffers del procesamiento de señales.

## 3. Entorno experimental

- Sistema anfitrión: computadora personal con Windows.
- Contenedor Docker: `heart-ai-lab`.
- Imagen Docker: `heart-ai-research:0.3`.
- Framework: PyTorch 1.10.0.
- Modalidad de ejecución evaluada: CPU.
- Número de hilos: 1.
- GPU disponible durante el desarrollo: NVIDIA GeForce GTX 1650.
- La GPU no se utilizó en las mediciones de inferencia en CPU.
- Procesador anfitrión: pendiente de documentar.

## 4. Metodología de medición

Se ejecutaron dos pruebas de inferencia utilizando la arquitectura `simple_cnn`:

**Prueba R01:** modelo inicializado con pesos aleatorios.

**Prueba R02:** modelo cargado desde el checkpoint entrenado `cnn_experimento_con_ponderacion.pth`.

Para ambas pruebas se utilizaron las siguientes condiciones:

- Entrada sintética de dimensiones (1, 384, 333).
- Diez ejecuciones de calentamiento.
- Cien ejecuciones cronometradas.
- Un hilo de CPU.
- Inferencia sin cálculo de gradientes.
- Medición mediante `time.perf_counter()`.

Las mediciones contemplan la ejecución de la CNN, pero no la adquisición de audio ni el procesamiento previo de las señales.

## 5. Resultados

| Indicador | R01: pesos aleatorios | R02: pesos entrenados |
|---|---:|---:|
| Latencia media (ms) | 15,06 | 14,91 |
| Mediana (ms) | 15,01 | 14,33 |
| Percentil 95 (ms) | 16,67 | 19,21 |
| Inferencias medidas | 100 | 100 |
| Hilos de CPU | 1 | 1 |

## 6. Interpretación preliminar

La arquitectura evaluada presenta una cantidad reducida de parámetros y logró tiempos medios de inferencia cercanos a 15 ms en la CPU del equipo anfitrión.

Estos resultados indican que la ejecución del modelo es computacionalmente viable en el entorno evaluado, sin necesidad de una GPU dedicada durante la inferencia.

No obstante, no permiten establecer todavía su rendimiento en plataformas embebidas, debido a diferencias de arquitectura del procesador, memoria disponible, bibliotecas y capacidad de procesamiento.

Tampoco representan la latencia completa del dispositivo, porque se excluyeron la captura y el procesamiento del audio.

Las diferencias entre las mediciones R01 y R02 no deben interpretarse como evidencia de una ventaja de rendimiento de los pesos entrenados.

## 7. Limitaciones

- Las mediciones se realizaron en una computadora personal y no en hardware embebido.
- Se emplearon entradas sintéticas.
- No se midió el consumo máximo de memoria durante la inferencia.
- No se midió el consumo energético.
- No se incluyó la adquisición de la señal cardíaca.
- No se incluyó el filtrado ni la extracción Log-Mel.
- No se evaluó todavía el tiempo total desde la adquisición hasta la presentación del resultado.
- No se cuantificó la variabilidad entre ejecuciones independientes del experimento.

## 8. Próximas mediciones

1. Identificar y documentar el procesador anfitrión.
2. Medir el tiempo de filtrado y extracción de características Log-Mel.
3. Medir la latencia completa del procesamiento de una grabación real.
4. Determinar el consumo de memoria durante la ejecución.
5. Evaluar compatibilidad con plataformas embebidas de bajo costo.
6. Medir latencia y consumo energético sobre el hardware seleccionado.
7. Documentar características del sensor, adquisición de audio y costo total del prototipo.

## 9. Alcance clínico

La investigación se encuentra en una etapa de factibilidad tecnológica. Las mediciones computacionales no permiten establecer desempeño diagnóstico ni capacidad de detección de valvulopatías específicas.

Cualquier aplicación clínica futura requerirá validación con etiquetas diagnósticas apropiadas, datos independientes y estudios específicos de seguridad y desempeño.

## 10. Caracterización de memoria de las activaciones de la CNN

Se realizó una inspección de las salidas intermedias de la arquitectura `simple_cnn` utilizando PyTorch en CPU, con una entrada sintética de dimensiones (1, 384, 333) y precisión FP32.

| Capa | Memoria de salida (MiB) |
|---|---:|
| Entrada | 0,488 |
| Conv1 | 2,546 |
| BatchNorm1 | 0,634 |
| Conv2 | 1,214 |
| BatchNorm2 | 0,297 |
| Conv3 | 0,540 |
| BatchNorm3 | 0,133 |
| Pooling final | Menos de 0,001 |

La mayor salida intermedia registrada corresponde a la primera capa convolucional (`Conv1`), con aproximadamente 2,546 MiB en FP32.

Una representación INT8 del mismo tensor requeriría aproximadamente 0,636 MiB, sin considerar buffers adicionales ni alineamiento de memoria.

Estas mediciones representan los tamaños individuales de los tensores de salida. No equivalen al consumo máximo real de RAM durante la ejecución.

### Implicaciones para el dispositivo

La arquitectura presenta un número reducido de parámetros, pero sus activaciones intermedias constituyen un requisito de memoria relevante.

Se considera técnicamente justificado investigar su cuantización y conversión a un formato de inferencia compatible con microcontroladores, especialmente el ESP32-S3 con PSRAM.

La viabilidad definitiva deberá verificarse mediante ejecución real sobre el hardware seleccionado, incluyendo memoria máxima, latencia, consumo energético y conservación del rendimiento de clasificación después de la cuantización.

**Estado:** medición exploratoria completada; validación sobre hardware embebido pendiente.

## 11. Exportación a ONNX y verificación de equivalencia numérica

### Objetivo

Evaluar la posibilidad de utilizar la CNN cardíaca previamente entrenada en un motor de inferencia diferente de PyTorch, como etapa preliminar para su eventual implementación en hardware embebido.

### Modelo utilizado

- Arquitectura: `simple_cnn`.
- Parámetros: 25.954.
- Checkpoint: `cnn_experimento_con_ponderacion.pth`.
- Época seleccionada: 27.
- Formato exportado: ONNX.
- Opset: 11.
- Tamaño del modelo exportado: 104,64 KiB.

### Entorno de comparación

- Python 3.9.
- PyTorch 1.10.0+cu102.
- NumPy 1.20.3.
- ONNX 1.13.1.
- ONNX Runtime 1.10.0.
- Imagen Docker: `heart-ai-research:0.5`.
- Motor de inferencia: CPU.

### Operaciones identificadas en ONNX

| Operación | Cantidad |
|---|---:|
| Conv | 3 |
| BatchNormalization | 3 |
| Gemm | 2 |
| MaxPool | 4 |
| Relu | 3 |
| Reshape | 2 |
| Softmax | 1 |
| Total | 18 |

### Metodología

Se cargó el checkpoint entrenado y se compararon las salidas de PyTorch y ONNX Runtime utilizando una misma entrada sintética de dimensiones (1, 384, 333), con semilla 42.

Se calcularon las diferencias absolutas máxima y media entre ambas salidas y se comprobó su equivalencia mediante `numpy.allclose`, utilizando tolerancias `rtol=1e-4` y `atol=1e-5`.

### Resultados

| Indicador | Resultado |
|---|---|
| Diferencia absoluta máxima | 3,05 × 10⁻¹¹ |
| Diferencia absoluta media | 1,52 × 10⁻¹¹ |
| Equivalencia numérica | Verdadera |
| Clase predicha por PyTorch | 1 — Anormal |
| Clase predicha por ONNX Runtime | 1 — Anormal |

### Interpretación

La exportación a ONNX conservó las salidas numéricas de la CNN entrenada para la entrada sintética utilizada en esta prueba.

El tamaño del archivo ONNX y la estructura de la arquitectura justifican continuar investigando su adaptación a plataformas embebidas de bajo costo.

Sin embargo, estos resultados no demuestran todavía compatibilidad ni rendimiento en ESP32-S3.

### Limitaciones

- Comparación realizada con una única entrada sintética.
- No se utilizaron señales cardíacas reales en esta prueba.
- No se evaluó todavía el efecto de la cuantización.
- No se verificó compatibilidad de todos los operadores con ESP-DL.
- No se midieron memoria ni latencia en un microcontrolador real.
- La equivalencia computacional no implica validez diagnóstica.

### Próxima etapa

Verificar la equivalencia PyTorch–ONNX sobre entradas cardíacas reales y estudiar la conversión del modelo a un formato compatible con ESP32-S3.

## 12. Verificación de equivalencia PyTorch–ONNX con grabaciones cardíacas reales

### Objetivo

Determinar si la exportación de la CNN entrenada a ONNX conserva las puntuaciones y decisiones de clasificación cuando se utilizan características acústicas procedentes de grabaciones cardíacas reales.

### Metodología

Se utilizaron las 41 grabaciones del conjunto de validación PhysioNet/CinC 2016, subconjunto training-a.

Las características Log-Mel previamente extraídas se cargaron desde un archivo HDF5 y se prepararon mediante el procedimiento existente en `HeartSoundDataSet`, incluyendo normalización y características delta y delta-delta.

Se compararon las salidas del mismo checkpoint entrenado mediante PyTorch y ONNX Runtime, ejecutados en CPU.

No se modificaron los pesos ni se realizó un nuevo entrenamiento.

### Resultados

| Indicador | Resultado |
|---|---:|
| Entradas cardíacas evaluadas | 41 |
| Predicciones coincidentes | 41/41 |
| Concordancia de clasificación | 100 % |
| Diferencia absoluta máxima | 1,4305 × 10⁻⁶ |
| Media de diferencias máximas por entrada | 5,3026 × 10⁻⁷ |
| Equivalencia numérica con tolerancia absoluta 10⁻⁵ | Verdadera |

### Interpretación

La conversión del modelo PyTorch a ONNX conservó las decisiones de clasificación para todas las entradas evaluadas, con diferencias numéricas pequeñas.

Este resultado aporta evidencia de portabilidad computacional del modelo entre los dos motores de inferencia utilizados.

La concordancia del 100 % corresponde a la equivalencia entre implementaciones y no representa exactitud diagnóstica.

### Limitaciones

- La comparación utilizó únicamente 41 entradas del conjunto de validación.
- Las características acústicas fueron preprocesadas previamente y cargadas desde HDF5.
- No se evaluó aún la adquisición directa desde un sensor.
- No se aplicó cuantización del modelo.
- No se ejecutó todavía el modelo en ESP32-S3.
- No se verificó la equivalencia después de la conversión a un motor de inferencia embebido.
- No se realizó validación clínica independiente.

### Próxima etapa

Investigar la conversión y cuantización del modelo ONNX para su eventual ejecución en un ESP32-S3 con PSRAM, verificando compatibilidad de operadores, memoria, latencia y conservación de las predicciones.

## 13. Conversión y cuantización de la CNN para ESP32-S3

### 13.1. Objetivo

Investigar la factibilidad de trasladar una CNN previamente entrenada a un microcontrolador ESP32-S3 con PSRAM, conservando la arquitectura original y evitando nuevas etapas de entrenamiento.

Se utilizó el modelo exportado previamente desde PyTorch a ONNX, cuya concordancia de clasificación con PyTorch fue del 100 % sobre las 41 entradas de validación evaluadas.

Esta concordancia corresponde a equivalencia entre motores de inferencia y no implica precisión diagnóstica.

### 13.2. Herramientas utilizadas

- Modelo de partida: `cnn_cardiaca.onnx`.
- Arquitectura: `simple_cnn`.
- Parámetros entrenables originales: 25.954.
- Herramienta de conversión: ESP-PPQ 1.3.11.
- Imagen Docker: `heart-ai-esp32:0.1`.
- Plataforma de destino configurada: ESP32-S3.
- Estrategia: cuantización posterior al entrenamiento (PTQ).
- Precisión objetivo: INT8.
- Dispositivo utilizado para la conversión: CPU de la computadora anfitriona.
- Datos de calibración: 32 entradas derivadas de grabaciones cardíacas reales del conjunto de entrenamiento.

### 13.3. Caracterización de las entradas de calibración

| Indicador | Resultado |
|---|---:|
| Entradas | 32 |
| Dimensiones | (32, 384, 333) |
| Tipo de dato original | float32 |
| Valores finitos | Sí |
| Valor mínimo | −0,9632 |
| Valor máximo | 3,4592 |
| Media | −0,00038 |
| Desviación estándar | 0,5780 |

Las entradas se obtuvieron mediante el mismo procedimiento de preparación de características utilizado por la CNN durante la evaluación.

No se emplearon grabaciones del conjunto de prueba reservado.

### 13.4. Resultados de la conversión

La función `espdl_quantize_onnx` logró generar un archivo en formato ESP-DL orientado al ESP32-S3.

| Indicador | Resultado |
|---|---|
| Modelo original | `cnn_cardiaca.onnx` |
| Tamaño ONNX | 104,64 KiB |
| Modelo generado | `cnn_cardiaca_esp32s3.espdl` |
| Tamaño ESP-DL, última ejecución | 37,75 KiB |
| Reducción aproximada del tamaño | 64 % |
| Operaciones procesadas reportadas | 18 |
| Operaciones cuantizadas reportadas | 18 |
| Variables cuantizadas reportadas | 37 |
| Entrenamiento adicional | No |
| Fusión manual de capas | No |

También se generaron archivos auxiliares `.info` y `.json` que permiten inspeccionar información técnica de la conversión.

### 13.5. Transformación automática de operaciones

La inspección del informe `.info` permitió observar que las tres operaciones originales `BatchNormalization` fueron representadas como convoluciones agrupadas con kernels de 1 × 1.

Las operaciones resultantes utilizan grupos de 16, 32 y 64 canales, respectivamente.

Esto constituye evidencia de que ESP-PPQ puede transformar automáticamente estas operaciones durante el procedimiento utilizado, sin requerir una fusión manual previa.

El informe también muestra operaciones de convolución, pooling, capas densas y Softmax en la representación exportada.

### 13.6. Informe preliminar de errores de cuantización

Se repitió la conversión con `error_report=True`.

Entre las operaciones visibles en la salida se registraron:

| Operación | Error reportado |
|---|---:|
| Conv_1 | 0,035 % |
| Gemm_15 | 0,011 % |
| Conv_5 | 0,010 % |
| Gemm_16 | 0,003 % |
| Conv_9 | 0,002 % |

Estos valores son resultados parciales del informe de ESP-PPQ. Su definición exacta y su interpretación deberán verificarse con el encabezado completo del reporte.

No representan tasas de error diagnóstico ni permiten determinar por sí solos la equivalencia de las predicciones del modelo cuantizado.

### 13.7. Advertencias observadas durante la exportación

ESP-PPQ emitió mensajes indicando que ciertos elementos no eran exportables, incluidos `62`, `65` y `classification`.

Aunque se generó el archivo `.espdl`, todavía es necesario confirmar que la representación exportada sea funcional y que sus salidas puedan recuperarse correctamente durante la inferencia.

No se considera validada la ejecución del modelo en ESP32-S3.

### 13.8. Interpretación preliminar

Los resultados demuestran la factibilidad de generar una representación cuantizada de la CNN original mediante ESP-PPQ, sin modificar manualmente su arquitectura ni realizar un nuevo entrenamiento.

El tamaño obtenido resulta favorable para investigar su almacenamiento en un microcontrolador.

Sin embargo, el tamaño del archivo no representa el consumo total de RAM durante la ejecución, especialmente debido a las activaciones intermedias y al procesamiento de audio.

La generación del archivo ESP-DL no garantiza todavía compatibilidad operativa, precisión equivalente ni rendimiento temporal adecuado en el dispositivo físico.

### 13.9. Próximas verificaciones

1. Interpretar completamente el informe de errores de cuantización.
2. Comprobar la equivalencia de predicciones entre el modelo ONNX y la versión cuantizada.
3. Verificar que todas las operaciones y la salida del modelo puedan ejecutarse mediante ESP-DL.
4. Probar la carga y la inferencia en un ESP32-S3 con PSRAM.
5. Medir memoria máxima y tiempo de inferencia en el microcontrolador.
6. Incorporar la captura acústica y el procesamiento Log-Mel.
7. Evaluar el consumo energético y el costo total del prototipo.

**Estado del experimento:** conversión y cuantización preliminares completadas; validación funcional y física pendientes.

