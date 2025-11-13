# TAREA 2.1
    El módulo ASPP se compone de 5 ramas paralelas que se procesan simultáneamente y luego se fusionan.

    Rama 1 (Convolución 1x1): Es la rama más simple. Actúa como un bypass o una proyección lineal simple. Es una convolución 1x1 estándar seguida de Batch Norm y ReLU.

    Ramas 2, 3, 4 (Convoluciones Atrous): Son el corazón del ASPP.

        Usamos nn.Conv2d pero con el parámetro dilation=rate.

        Una dilatación de 6, 12 o 18 significa que el kernel de 3x3 "ve" un campo receptivo más amplio, capturando contexto a diferentes escalas.

        El padding se establece igual al rate para asegurar que el tamaño de salida (Alto x Ancho) sea el mismo que el de entrada.

        Como tenemos 3 rates, crearemos un nn.ModuleList para almacenar estos 3 nn.Sequential.

    Rama 5 (Image Pooling): Esta rama captura el contexto global de toda la imagen.

        Primero, nn.AdaptiveAvgPool2d(1) reduce el feature map a 1x1xC (donde C es in_channels).

        Luego, una convolución 1x1 lo proyecta a out_channels.

        Finalmente, en el forward, tendremos que aumentar (upsample) este feature de 1x1 a la dimensión original del feature map de entrada para poder concatenarlo con las otras ramas.

    Capa de Fusión (conv_out):

        Las 5 ramas (1 + 3 atrous + 1 global) producen 5 feature maps, cada uno con out_channels.

        Al concatenarlos en la dimensión de canales (dim=1), tendremos un feature map con 5 * out_channels.

        conv_out es una convolución 1x1 que reduce esta dimensionalidad de vuelta a out_channels, fusionando toda la información.

# TAREA 2.2
    El forward debe tomar la entrada x y pasarla por las 5 ramas, luego fusionar los resultados.

    Guardamos el tamaño de entrada size = x.shape[2:] (Alto y Ancho) para usarlo en el upsampling de la rama 5.

    Calculamos feat1 = self.conv1x1(x).

    Calculamos las 3 ramas atrous usando una list comprehension: feat_atrous = [conv(x) for conv in self.atrous_convs].

    Calculamos la rama global: feat_global = self.global_avg_pool(x).

    Paso clave: Aumentamos feat_global (que es BxCx1x1) de vuelta al tamaño original size usando F.interpolate.

    Concatenamos todo: torch.cat([feat1] + feat_atrous + [feat_global_upsampled], dim=1).

    Pasamos el resultado concatenado por la capa de fusión self.conv_out y lo retornamos.

# TAREA 2.3

Vamos a definir todos los bloques de construcción:

    Backbone (Encoder): Cargamos un resnet50 pre-entrenado. A diferencia del FCN, no solo extraemos las capas, sino que las guardamos como bloques. Necesitamos dos salidas de este backbone:

        Características de bajo nivel: La salida de layer1. Tiene alta resolución (stride 4) pero semántica simple. Es buena para definir bordes. layer1 de ResNet50 produce 256 canales.

        Características de alto nivel: La salida de layer4. Tiene baja resolución (stride 32) pero una semántica muy rica (sabe "qué" hay). layer4 produce 2048 canales.

    Módulo ASPP: Conectamos nuestro módulo ASPP (que ya hicimos) al final de layer4. Tomará los 2048 canales de entrada y los comprimirá en 256 canales de salida, ricos en contexto multi-escala.

    Proyección de Bajo Nivel: Las características de layer1 (256 canales) no se pueden concatenar directamente con las del ASPP. El paper de DeepLabV3+ las proyecta primero con una convolución 1x1 a 48 canales. Esto reduce la dimensionalidad y permite al decodificador funcionar.

    Decoder: Este módulo recibe la concatenación de:

        La salida del ASPP (256 canales), aumentada (upsampled) 8x (de stride 32 a stride 4).

        La salida de la proyección de bajo nivel (48 canales), que ya está en stride 4.

        Total de canales de entrada: 256 + 48 = 304.

        El decodificador usa un par de convoluciones de 3x3 para refinar estas características fusionadas y produce 256 canales de salida.

    Cabeza de Clasificación: Una convolución 1x1 final que toma los 256 canales refinados del decodificador y los mapea a las n_classes finales.

# TAREA 2.4

El forward pass debe conectar todas estas piezas:

    Guardar el tamaño: Guardamos el tamaño original de la imagen size = x.shape[2:].

    Encoder (Parte 1): Pasamos la imagen por las capas iniciales (conv1...maxpool).

    Encoder (Parte 2 - Low-Level): Pasamos por layer1. ¡Guardamos esta salida! (la llamaremos low_level_feat).

    Encoder (Parte 3 - High-Level): Continuamos pasando el feature map por layer2, layer3 y layer4 para obtener las características semánticas profundas (x). En este punto, x tiene stride 32 (es pequeño) y low_level_feat tiene stride 4 (es grande).

    ASPP: Pasamos x (stride 32) por self.aspp(x). La salida sigue teniendo stride 32 pero ahora tiene 256 canales.

    Decoder (Fusión):

        Aumentamos (upsample) x (stride 32) para que coincida con el tamaño de low_level_feat (stride 4). Esto es un upsampling de 8x.

        Proyectamos low_level_feat usando self.low_level_conv(low_level_feat).

        Concatenamos torch.cat() las dos salidas en la dimensión de canal (dim=1).

        Pasamos el resultado concatenado por self.decoder.

    Clasificación:

        Pasamos la salida del decodificador por self.classifier.

        Finalmente, aumentamos (upsample) el resultado (que está en stride 4) al tamaño original de la imagen size.

        Retornamos el resultado.