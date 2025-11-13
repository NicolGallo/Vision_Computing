# Parte 2: Mecanismos de atencion

## Resumen general
Los modulos de atencion afinan la representacion interna de una CNN sin cambiar la topologia convolucional de base. La idea central es aprender mascaras multiplicativas que refuerzan senales relevantes y atenuan ruido. En esta practica se implementan dos mecanismos complementarios:

- `SEModule`: aplica atencion solo sobre el eje de canales.
- `CBAM`: combina atencion de canal y espacial de forma secuencial.

A continuacion se detalla el funcionamiento matematico y la traduccion al codigo de `lab03_student.py`.

## SEModule (Squeeze-and-Excitation)

### Flujo matematico
1. **Squeeze**: promedio global por canal
   \[
   s_c = \frac{1}{H W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c, i, j}, \quad s \in \mathbb{R}^C
   \]
   donde `x` es el tensor de entrada `(B, C, H, W)` y `s` condensa cada canal en un escalar.

2. **Excitation**: MLP de dos capas con ratio de reduccion `r`
   \[
   z = W_2\,\delta(W_1 s) \quad\text{con}\quad W_1 \in \mathbb{R}^{\frac{C}{r} \times C},\; W_2 \in \mathbb{R}^{C \times \frac{C}{r}}
   \]
   La funcion `\delta` es `ReLU` y la salida pasa por `sigmoid` para obtener pesos en `[0, 1]`:
   \[
   a = \sigma(z)
   \]

3. **Scale**: reexpande `a` a `(B, C, 1, 1)` y la multiplica elemento a elemento con `x`:
   \[
   y_{c, i, j} = a_c \cdot x_{c, i, j}
   \]

### Aspectos de implementacion
- `AdaptiveAvgPool2d(1)` produce directamente tensores `(B, C, 1, 1)` sin importar el tamano espacial.
- `reduced_channels = max(channels // reduction, 1)` evita quedar en cero cuando `C < reduction`.
- Uso de `F.relu(..., inplace=True)` preserva memoria durante entrenamiento.
- Los pesos se guardan en `block.se` y se evaluan dentro del `forward` de cada bloque residual.

## CBAM (Convolutional Block Attention Module)

CBAM concatena dos etapas: atencion de canal (similar a SE) y atencion espacial que discrimina ubicaciones.

### Atencion de canal
Se reutiliza `SEModule`, por lo que la mascara `M_c(x)` se calcula exactamente como en la seccion anterior.

### Atencion espacial
1. Calculo de mapas estadisticos por canal:
   \[
   F_{avg}(x) = \frac{1}{C} \sum_{c=1}^{C} x_{c,:,:}, \quad F_{max}(x) = \max_{c \in [1,C]} x_{c,:,:}
   \]
2. Concatenacion `U = [F_{avg}; F_{max}]` da un tensor `(B, 2, H, W)`.
3. Convolucion 2D `conv( U )` con kernel impar (`kernel_size=7` por defecto) captura contexto local amplio.
4. `sigmoid` genera la mascara espacial `M_s(x)`.

### Combinacion final
El bloque completo produce
\[
F'(x) = x \odot M_c(x), \qquad F''(x) = F'(x) \odot M_s(F'(x))
\]
La multiplicacion se realiza in-place en codigo para mantener eficiencia.

## Integracion en ResNet

- `SEResNet` y `CBAMResNet` heredan de `ResNet` y redefinen un metodo privado (`_inject_se` o `_inject_cbam`).
- Durante la creacion de cada bloque residual se detecta dinamicamente su numero de canales con `bn2` (bloques basicos) o `bn3` (bottleneck).
- El atributo `block.se` se aprovecha en el `forward` de los bloques existentes: si no es `None`, se aplica la mascara tras la segunda/tercera normalizacion.
- Como el `forward` original suma luego con la identidad, el bloque completo queda:
  \[
  y = \mathrm{ReLU}\big(\mathcal{F}(x) \odot M(x) + \mathcal{S}(x)\big)
  \]
  donde `M` es `SEModule` o `CBAM`, y `\mathcal{S}` denota la ruta de atajo (posiblemente con `downsample`).

## Factorias disponibles
Para facilitar comparaciones experimentales, se exponen interfaces rapidas:

- `seresnet18`, `seresnet34`, `seresnet50`, `seresnet101`
- `cbam_resnet18`, `cbam_resnet34`, `cbam_resnet50`, `cbam_resnet101`

Cada funcion acepta `reduction` (y `kernel_size` en CBAM) para explorar distintas configuraciones sin reescribir el grafo.

## Consideraciones practicas
- **Eleccion de `reduction`**: valores en `\{4, 8, 16\}` equilibran coste y capacidad; un `reduction` demasiado grande limita la expresividad del MLP.
- **Inicializacion**: al multiplicar la salida por una mascara `sigmoid`, conviene usar inicializaciones estandar (Kaiming/He) para las convoluciones y dejar que la mascara se acerque a 1 al inicio del entrenamiento.
- **Impacto en FLOPs**: SE introduce solo operaciones lineales globales; CBAM anade una convolucion 2D extra pero de bajo coste (solo 2 canales de entrada).
- **Compatibilidad**: al incrustar la referencia del modulo en `block.se`, se pueden seguir utilizando checkpoints de ResNet siempre que la nueva clase cargue pesos con `strict=False` o se inicialicen de cero los parametros del modulo de atencion.

## Relacion con la seccion de pruebas
En `test_attention_implementation` se generan tensores aleatorios para verificar:

- Formas de salida de `SEModule` y `CBAM` aislados.
- Que `seresnet18` y `cbam_resnet18` realizan un forward completo y devuelven logits `(B, 10)`.

Estas pruebas actuan como sanidad minima antes de realizar entrenamientos prolongados.
