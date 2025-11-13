# Parte 4: Benchmarking y Analisis

## Objetivo general
El bloque final del laboratorio proporciona herramientas para medir y comparar modelos entrenados en vision. Se busca cuantificar aspectos de coste computacional (FLOPs, latencia, memoria) junto con metricas de precision para facilitar experimentos reproducibles. La implementacion se apoya unicamente en PyTorch, evitando dependencias externas.

## ModelProfiler
`ModelProfiler` encapsula metodos de medicion para un unico modelo.

### Conteo de parametros
El metodo `count_parameters` suma `p.numel()` de todos los tensores entrenables. Esto corresponde al numero total de grados de libertad del modelo. El resultado se devuelve como `int`, lo que permite reportarlo en miles o millones segun convenga.

### Estimacion de FLOPs
- Se registran *forward hooks* en cada `Conv2d` y `Linear`.
- Para convoluciones, el coste se aproxima como:
  \[
  \text{FLOPs} = B \cdot C_{out} \cdot H_{out} \cdot W_{out} \cdot \left( \frac{C_{in}}{\text{groups}} \cdot K_h \cdot K_w + \mathbf{1}_{\text{bias}} \right)
  \]
- Para capas lineales se asume multiplicacion matriz-vector mas sesgo:
  \[
  B \cdot (C_{in} \cdot C_{out} + \mathbf{1}_{\text{bias}} \cdot C_{out})
  \]
- Tras ejecutar un *dummy forward* se eliminan los hooks y se restaura el modo de entrenamiento original.

### Perfil de memoria
- Si no hay GPU disponible se devuelve `{forward: 0.0, backward: 0.0, peak: 0.0}` como indicador.
- En GPU:
  1. Se limpia la cache y se reinician las estadisticas de memoria.
  2. Se registra la memoria maximo asignada durante una pasada en modo evaluacion (sin gradientes).
  3. Para la pasada de entrenamiento se generan etiquetas aleatorias y se calcula `CrossEntropyLoss`, haciendo `backward()` para contabilizar buffers de gradiente.
  4. Se calcula `backward_mem = max(peak - forward, 0)` para aislar el coste adicional de retropropagacion.

### Latencia
- Realiza varias ejecuciones de calentamiento para estabilizar caches.
- Usa `time.perf_counter()` y sincroniza la GPU si aplica.
- Devuelve el tiempo medio por pasada en segundos (latencia = `elapsed / num_runs`).

### Benchmark de entrenamiento
- Ejecuta SGD con `lr=0.01` y `momentum=0.9` durante el numero de epochs indicado.
- Cuenta muestras procesadas para calcular `samples_per_second` y promedio de tiempo por epoch.
- Restaura el modo de entrenamiento original del modelo al finalizar.

## ArchitectureComparator
Permite comparar un conjunto de modelos sobre las mismas condiciones.

### Accuracy
- Pone cada modelo en modo `eval()` y recorre el `test_loader` acumulando aciertos.
- Guarda el modo original (`train()` o `eval()`) de cada modelo para restaurarlo posteriormente.
- Resultados almacenados en `self.results['accuracy']` para reutilizacion.

### Eficiencia
- Para cada modelo crea un `ModelProfiler` y registra un diccionario con:
  - `parameters`: total de pesos entrenables.
  - `flops`: coste estimado por inferencia.
  - `latency`: tiempo medio de forward en segundos.
  - `memory`: memoria maxima en MB durante forward/backward (0 en CPU).
- Los valores quedan en `self.results['efficiency']`.

### Visualizacion
`plot_comparison` dibuja tres graficas (accuracy, parametros y latencia). Requiere que previamente se hayan ejecutado los metodos de precision y eficiencia. El grafico puede guardarse en disco (argumento `save_path`) o mostrarse en pantalla.

## Utilidades de datos y entrenamiento
- `create_data_loaders` configura transformaciones tipicas de CIFAR10/100 e incluye opcion de submuestrear el conjunto de entrenamiento.
- `train_model` implementa un bucle clasico de entrenamiento con seguimiento de perdidas y exactitud para train/val, asi como un scheduler `StepLR` opcional.

## Flujo sugerido de uso
1. Crear arquitecturas a comparar (p.ej., `resnet18`, `seresnet18`, `vit_tiny`).
2. Generar `train_loader` y `test_loader` con `create_data_loaders` (posiblemente con `subset_size` para experimentos rapidos).
3. Entrenar cada modelo con `train_model` u otro pipeline propio.
4. Instanciar `ArchitectureComparator` y ejecutar `compare_accuracy` + `compare_efficiency`.
5. Llamar a `plot_comparison` para obtener una vista conjunta de rendimiento vs coste.

## Consideraciones practicas
- Las estimaciones de FLOPs/latencia dependen del dispositivo y de la precision numerica; para comparaciones justas mantener configuraciones identicas.
- Las mediciones de memoria solo aparecen si existe GPU; en CPU se reportan ceros para indicar que la metrica no es aplicable.
- Se recomienda fijar `torch.manual_seed` antes de pruebas cuantitativas para garantizar reproducibilidad del `train_model` y del muestreo de subconjuntos.
- Para modelos muy grandes, reducir `num_runs` en `measure_latency` evita tiempos excesivos sin perder representatividad.

## Extensiones posibles
- Integrar `torch.cuda.amp` para medir el impacto de precision mixta.
- Agregar soporte para perfiles detallados por capa exportando los datos de hooks.
- Guardar los resultados en CSV/JSON a partir de `self.results` para historicos de benchmarking.
- Implementar barras apiladas o diagramas radar en `plot_comparison` para incluir memoria y FLOPs en la visualizacion.
