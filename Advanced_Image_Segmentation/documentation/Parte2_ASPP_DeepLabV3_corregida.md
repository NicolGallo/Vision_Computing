# Documentación de DeepLabV3+ y ASPP 

Este código implementa DeepLabV3+, una de las arquitecturas de segmentación semántica más influyentes y de mayor rendimiento. Su componente central, que lo distingue de modelos anteriores, es el módulo **ASPP** (Atrous Spatial Pyramid Pooling).

## 1. ASPP (Atrous Spatial Pyramid Pooling)

El módulo ASPP o convolución Atrous es el core de la familia DeepLab. Resuelve uno de los mayores desafíos en la segmentación: la captura de contexto multi-escala. En la segmentación, se producen o se necesitan dos aspectos contradictorios:

- **Información Semántica (El "Qué"):** Para saber que un grupo de píxeles es un "coche", necesitas un gran campo receptivo (ver una gran parte de la imagen). Esto se logra con redes profundas y downsampling (pooling).

- **Información Espacial (El "Dónde"):** Para dibujar el borde exacto del coche, necesitas una alta resolución espacial. El downsampling anterior destruye dicha información.

ASPP resuelve la gran problemática de cómo obtener un gran campo receptivo sin perder resolución.

La convolución Atrous (o dilatada) se caracteriza por ser una convolución con "agujeros". Introduce un parámetro llamado dilation rate (tasa de dilatación) que separa los pesos del kernel. Por ejemplo:

- **dilation=1:** Es una convolución estándar.

- **dilation=2:** El kernel (ej. 3x3) se expande, cubriendo un área de 5x5, pero solo usa los mismos 9 pesos, saltando un píxel entre cada peso.

- **dilation=6:** El kernel se expande aún más, cubriendo un área mucho mayor.

Una de las ventajas de los ejemplos anteriores es que aumenta exponencialmente el campo receptivo sin añadir parámetros y, lo más importante, sin realizar downsampling (el stride sigue siendo 1).

### 1.1. Implementación Apartado 2.1

El módulo ASPP se compone de 5 ramas paralelas con múltiples tasas de dilatación y que se procesan simultáneamente y luego se fusionan, capturando así los objetos. Estas ramas se detallan a continuación:

- **Rama 1 (*Convolución 1x1*):** Es la rama más simple. Es una convolución 1x1 estándar seguida de Batch Norm y ReLU. Captura características locales y finas y actúa como una proyección simple de las características de entrada.

- **Ramas 2, 3, 4 (*Convoluciones Atrous*):** Son el centro del ASPP. Se crean 3 convoluciones 3x3 con dilataciones iguales a 6, 12 y 18. Esto significa que el kernel de 3x3 "ve" un campo receptivo más amplio, capturando contexto a diferentes escalas.
El padding se establece igual al rate (`padding=rate`) para asegurar que el tamaño de salida (Alto x Ancho) sea el mismo que el de entrada (el tamaño espacial del mapa de características no cambia).

- **Rama 5 (*Image Pooling*):** Esta rama captura el contexto global de toda la imagen. Primero, `nn.AdaptiveAvgPool2d(1)` reduce el feature map a `1x1xC` (donde C es `in_channels`). De esta manera, se colapsa todo el mapa de características en un solo vector 1x1 (vector promedio). Posteriormente, se aplica una convolución 1x1 lo proyecta a `out_channels` (256).

En la capa de fusión (`conv_out`), las 5 ramas (1 + 3 atrous + 1 global) producen 5 feature maps, cada uno con `out_channels`. Al concatenarlos en la dimensión de canales (`dim=1`), se tiene un feature map con `5 * out_channels`.

**Nota:** `conv_out` es una convolución 1x1 que reduce esta dimensionalidad de vuelta a `out_channels`, fusionando toda la información.

### 1.2. Implementación Apartado 2.2

- En la implementación del `forward` del ASPP, se debe tomar la entrada x y pasarla por las 5 ramas para luego fusionar los resultados. Hay que destacar el guardado del tamaño de entrada `size = x.shape[2:]` (Alto y Ancho) para usarlo en el upsampling de la rama 5.

- Se calculan las primeras 4 ramas (las cuales mantienen el tamaño espacial de `x`). Por último, se calcula la rama global o 5 la cual se reduce a 1x1. 

- El paso clave reside en que se aumenta la rama global (que es `BxCx1x1`) de vuelta al tamaño original size usando `F.interpolate`. Esto transmite la información de contexto global a cada ubicación de píxel y permite la concatenación con las otras ramas.

- A continuación, se aplica la concatenación de las ramas en que se apila todas las características de todas las ramas a lo largo de la dimensión del canal. Si cada rama produce 256 canales por ejemplo, el resultado es un tensor con `5 * 256 = 1280 canales`.

- Por último, el resultado concatenado se pasa por la capa de fusión (que correspondía a una `Conv2d` 1x1) en que fusiona estos 1280 canales y los reduce nuevamente a 256 canales (`out_channels`).

De esta forma, el resultado es un mapa de características del mismo tamaño que la entrada, pero ahora cada píxel contiene información semántica rica de múltiples escalas.

## 2. DeepLabV3+

DeepLabV3+ toma el potente módulo ASPP y lo integra en una arquitectura Encoder-Decoder explícita. 

La versión mejorada de DeepLabV3 surge debido a que los modelos FCN y DeepLabV3 (sin plus) tenían un problema: reconstruían la segmentación a partir de características muy profundas (ej. stride 16 o 32). Esto resultaba en bordes de objetos "borrosos" o imprecisos.

DeepLabV3+ ("Plus") soluciona este problema añadiendo una ruta de decodificación simple pero efectiva que reutiliza las características de bajo nivel (alta resolución) de las primeras capas del encoder.

### 2.1. Implementación Apartado 2.3

Hay dos bloques diferenciados de construcción, como son la sección encoder y la sección decoder:

#### a) Sección Encoder (Backbone + ASPP)
    
El encoder tiene como propósito generar características semánticas ricas, aunque a baja resolución. Como backbone se emplea una ResNet50 preentrenada, en que a diferencia del FCN, no solo se extraen las capas, sino que se guardan como bloques. Del backbone utilizado, se necesitan dos salidas:

**1. Características de bajo nivel:** Se corresponde con la salida de la `layer1`. Tiene una alta resolución (stride 4) pero semántica simple. Es buena para definir bordes ya que la red aún no sabe qué son los objetos. La `layer1` de ResNet50 produce 256 canales.

**2. Características de alto nivel:** Se corresponde con la salida de la `layer4`. Tiene baja resolución (stride 32) pero una semántica muy rica (sabe "qué" hay). La `layer4` de ResNet50 produce 2048 canales.

Seguidamente, se conecta el módulo ASPP implementado al final de `layer4`. Por ello, tomará los 2048 canales de entrada y los comprimirá en 256 canales de salida, ricos en contexto multi-escala o semántica gracias al efecto del ASPP.

**Aspecto clave:** Se produce una proyección de bajo nivel, debido a que las características de la `layer1` (256 canales) no se pueden concatenar directamente con las del ASPP. El paper de DeepLabV3+ las proyecta primero con una convolución 1x1 a 48 canales. Esto reduce la dimensionalidad y permite al decodificador funcionar.

#### b) Sección Decoder (Fusión de características)  
    
El decoder tiene como propósito combinar la información semántica (del encoder) con la información espacial para refinar los bordes: Este módulo recibe la concatenación de:

**1.** La salida del ASPP (256 canales), aumentada (upsampled) 8x (de stride 32 a stride 4).
**2.** La salida de la proyección de bajo nivel (48 canales), que ya está en stride 4.

El resultado es la concatenación de las anteriores dos salidas, siendo esta la innovación de la versión plus del DeepLabV3 (total de canales de entrada: 256 + 48 = 304).

El decodificador utiliza `dos convoluciones` 3x3 para refinar estas características fusionadas y produce un tensor refinado de 256 canales de salida (con valor de stride 4).

Por último, se realiza la clasificación final mediante una `convolución` 1x1 final que toma los 256 canales refinados del decodificador y los mapea a las `n_classes` finales, dando lugar al mapa de segmentación final.

### 2.2. Implementación Apartado 2.4

La implementación del método `forward` debe poder conectar todas las partes del anterior apartado:

**1.** Se guarda el tamaño original de la imagen de entrada (`size = x.shape[2:]`).

**2.** Encoder (Parte 1): Se pasa la imagen por la capa inicial (`conv1...maxpool`).

**3.** Encoder (Parte 2): Se guarda la salida de la `layer1`.

**4.** Encoder (Parte 3): Se procede a pasar el feature map obtenido en la `layer1` por `layer2`, `layer3` y `layer4` para obtener las características semánticas profundas (`x`). En este punto, `x` tiene stride 32 (es pequeño) y la salida de la `layer1` tiene stride 4 (es grande).

**5.** Se pasa el output `x` (stride 32) por `self.aspp` para aplicar el efecto del ASPP. La salida sigue teniendo stride 32 pero ahora tiene 256 canales.

**6.** Decoder (Fusión): Se sobremuestrea el output `x` (con stride 32) para que coincida con el tamaño de la salida almacenada de la `layer1` (con stride 4). Esto se corresponde con un upsampling de 8x.

**7.** Seguidamente, se proyecta la salida almacenada de la `layer1` para obtener una salida con stride 4 pero ahora con 48 canales.

**8.** Se concatenan las dos salidas en la dimensión de canal (`dim=1`) con `torch.cat()`.

**9.** Se procede a pasar el resultado concatenado por `self.decoder`.

**10.** Se pasa la salida del decodificador por `self.classifier` para efectuar la clasificación final.

**11.** Finalmente, se sobremuestrea el resultado (que está en stride 4) al tamaño original de la imagen size.


A nivel gráfico o visual, se podría representar la DeepLabV3+ de la forma siguiente:

                                    ┌────────────────────────────────────────────┐
                                    │                INPUT IMAGE                 │
                                    │                (H x W x 3)                 │
                                    └────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                             conv1 → bn1 → relu → maxpool
                                                          │
                                                          ▼
                                             ResNet layer1  (stride 4)
                                             (low-level features, 256 ch)
                                                          │
                                                          ▼
                                             ResNet layer2  (stride 8)
                                                          │
                                                          ▼
                                             ResNet layer3  (stride 16)
                                                          │
                                                          ▼
                                             ResNet layer4  (stride 32)
                                             (high-level features, 2048 ch)
                                                          │
                                                          ▼
                                    ┌──────────────────────────────────────────┐
                                    │                ASPP (256 ch)             │
                                    │ (1x1 conv, atrous convs, global pool)    │
                                    └──────────────────────────────────────────┘
                                                          │
                                                          ▼
                            F.interpolate(ASPP output) → Upsample to low-level size (stride 4)
                                                          │
                                                          ▼
                                       low_level_conv (1x1 proj: 256 → 48 ch)
                                                          │
                                                          ▼
                                 torch.cat([ASPP_upsampled, low_level_proj]) → 304 channels
                                                          │
                                                          ▼
                                      decoder (Conv3x3 → 256 → Conv3x3 → 256)
                                                          │
                                                          ▼
                                        classifier (1×1 conv → n_classes)
                                                          │
                                                          ▼
                                F.interpolate(..., size=input_size) → Upsample to H×W
                                                          │
                                                          ▼
                                    ┌─────────────────────────────────────────┐
                                    │              FINAL OUTPUT               │
                                    │          (H × W × n_classes)            │
                                    └─────────────────────────────────────────┘
