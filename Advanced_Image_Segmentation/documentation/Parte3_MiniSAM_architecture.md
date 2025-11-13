La arquitectura MiniSAM consta de tres partes clave:

1. Codificador de Imagen (Task 3.1): Un backbone ligero que extrae características de la imagen.
2. Codificadores de Prompts (Task 3.2): Redes diminutas que convierten los clics (puntos) o cajas (boxes) en vectores.
3. Decodificador y Cabezas (Task 3.3): Un decodificador que fusiona la imagen y los prompts para generar la máscara.

# 🧠 Tarea 3.1: Codificador de Imagen (Image Encoder)

Razonamiento: Necesitamos un backbone rápido y ligero. El SAM original usa un ViT (Vision Transformer) enorme, pero para un "Mini-SAM" usaremos MobileNetV3-Small.

    Cargar MobileNet: Cargamos models.mobilenet_v3_small(pretrained=True).

    Extraer Características: Estamos interesados en sus capas convolucionales, no en su cabeza de clasificación final. El TODO sugiere nn.Sequential(*list(backbone.features)).

    Corrección Importante (Gotcha): El backbone.features de MobileNetV3-Small termina con una capa de AdaptiveAvgPool2d que colapsa las dimensiones espaciales a 1x1. Esto no nos sirve. Debemos truncar el backbone antes de esa capa.

        backbone.features[:-1] nos da las capas hasta la última convolución, que produce 576 canales y un feature map con stride 16 (tamaño H/16, W/16). Esto coincide con el TODO que menciona 576 canales.

    Proyección: SAM funciona proyectando todo a una dimensión de embedding común (ej. 256). Usamos una convolución 1x1 (img_proj) para reducir los 576 canales del MobileNet a embed_dim.