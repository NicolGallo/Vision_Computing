# Parte 3: Vision Transformer

## Idea general
El Vision Transformer (ViT) adapta la arquitectura de transformadores al dominio de vision. En lugar de procesar pixeles uno a uno, la imagen se segmenta en parches disjuntos que actuan como tokens equivalentes a palabras. Cada token se proyecta a un espacio latente de dimension fija y se alimenta a una pila de bloques Transformer con autoatencion multi-cabeza. La salida asociada a un token especial de clasificacion se usa para predecir la etiqueta final.

## PatchEmbedding
1. **Discretizacion espacial**: la imagen de entrada `x` tiene forma `(B, C, H, W)` con `H = W = img_size`. Si `patch_size` divide exactamente a `img_size`, el numero de parches es
   \[
   N = (H / P)^2
   \]
   donde `P = patch_size`.
2. **Extraccion y proyeccion**: una convolucion `Conv2d` con `kernel_size=P` y `stride=P` extrae cada parche sin solapamiento y lo proyecta a `embed_dim`. El resultado inicial tiene forma `(B, embed_dim, H/P, W/P)`.
3. **Reformateo**: el tensor se aplana en la dimension espacial y se transpone para obtener `(B, N, embed_dim)`, lo cual emula una secuencia de tokens lista para el transformador. Si se desactiva `flatten`, se conserva la forma de rejilla para usos posteriores.
4. **Validacion**: la implementacion comprueba que `img_size` sea divisible por `patch_size` y lanza un error descriptivo si no se cumple.

## MultiHeadAttention
La autoatencion multi-cabeza calcula interacciones entre tokens mediante tres proyecciones lineales compartidas:
\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\]
con `X` de forma `(B, N, D)` y matrices `W` de dimension `(D, D)`. Despues se reordena el tensor para crear `num_heads` particiones de tamano `D_h = D / num_heads`. Para cada cabeza se evalua la atencion escalada:
\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{Q K^\top}{\sqrt{D_h}} \right) V
\]
La implementacion aplica `Dropout` tanto a la distribucion de atencion como a la proyeccion final `W_O`. En caso de que `embed_dim` no sea divisible por `num_heads`, se lanza una excepcion para alertar al usuario.

## TransformerBlock
Cada bloque corresponde al codificador clasico de un Transformer:
1. **LayerNorm previa**: normaliza cada token antes de la autoatencion.
2. **Residual + MHA**: `x = x + MHA(LayerNorm(x))` mantiene rutas de gradiente estables.
3. **MLP**: una red de dos capas con activacion `GELU` y `Dropout`. El tamano intermedio es `mlp_ratio * embed_dim`.
4. **Residual + MLP**: `x = x + MLP(LayerNorm(x))` cierra el bloque.
5. **Validacion**: se comprueba que `mlp_ratio` sea positivo para evitar dimensiones degeneradas.

## VisionTransformer
### Componentes
- **PatchEmbedding**: genera `N` tokens visuales.
- **Token CLS**: `cls_token` es un vector aprendible de forma `(1, 1, D)` que se concatena al inicio de la secuencia para agregacion global.
- **Positional embedding**: `pos_embed` aporta informacion de orden a cada token. El vector se recorta dinamicamente si se altera el numero de parches.
- **Stacks Transformer**: `depth` controla cuantas veces se repite el bloque descrito. Cada bloque opera sobre la secuencia completa.
- **Cabeza de clasificacion**: tras normalizar la salida, se extrae el token de clasificacion (`x[:, 0]`) y se envía a una capa lineal de tamaño `(D, num_classes)`.

### Flujo de datos
1. `x = PatchEmbedding(x)`
2. `x = concat(cls_token, x)`
3. `x = x + pos_embed` y dropout
4. `for block in blocks: x = block(x)`
5. `x_cls = LayerNorm(x)[:, 0]`
6. `logits = Linear(x_cls)`

### Inicializacion
- `trunc_normal_` con `std=0.02` inicializa `cls_token`, `pos_embed` y los pesos lineales.
- Las bias de las capas lineales arrancan en cero y las capas `LayerNorm` inician con peso 1 y bias 0.

### Helpers y reutilizacion
- Se definio `forward_features` para obtener embeddings antes de la cabeza de clasificacion. Esto permite reutilizar la red en tareas de transferencia (e.g., fine-tuning con cabeceras distintas).
- El metodo `forward` delega en `forward_features` y aplica `self.head` para obtener logits.

## Configuraciones predeterminadas
Se proveen factorias para tres tamaños comunes adaptados a CIFAR (imagenes 32x32, parches 4x4):
- `vit_tiny`: `embed_dim=192`, `num_heads=3`.
- `vit_small`: `embed_dim=384`, `num_heads=6`.
- `vit_base`: `embed_dim=768`, `num_heads=12`.
Todas usan `depth=12` y `mlp_ratio=4.0`. Estos parametros guardan proporciones similares a las variantes oficiales descritas en el paper original de ViT.

## Consideraciones practicas
- **Escalado de resolucion**: si se desea usar entradas de mayor tamaño, basta ajustar `img_size` y `patch_size`, teniendo cuidado de recalcular `pos_embed` o reinterpolarlo al cargar pesos preentrenados.
- **Dropout y regularizacion**: el parametro `dropout` se aplica tanto a la atencion como al MLP. Para tareas con poco dato se puede incrementar `dropout` o usar tecnicas adicionales (stochastic depth, data augmentation).
- **Costo computacional**: la autoatencion escala como `O(N^2)` donde `N` es el numero de parches. Con `patch_size` mas pequeno aumentan los parches y el coste. Un compromiso comun para 224x224 es `P=16` (N=196).
- **Transfer learning**: al cambiar `num_classes`, se puede crear el modelo y posteriormente sustituir `model.head` por otra capa adaptada sin tocar el resto del grafo.
- **Verificacion**: el script `test_vit_implementation` genera tensores de prueba y confirma que `PatchEmbedding` y las variantes `vit_*` producen las formas esperadas.

## Resumen matematico
Sea `X` la imagen y `\phi` la proyeccion de parches. La red implementa:
\[
\begin{aligned}
&Z_0 = [x_{cls}; \phi(X)] + E_{pos} \\
&Z_{l} = Z_{l-1} + \mathrm{MHA}(\mathrm{LN}(Z_{l-1})), \quad l=1..L \\
&Z'_{l} = Z_{l} + \mathrm{MLP}(\mathrm{LN}(Z_{l})) \\
&y = W_{head} \cdot \mathrm{LN}(Z'_{L}[0])
\end{aligned}
\]
con `Z_0` de dimension `(N+1, D)` y `y` de dimension `(num_classes,)` por muestra.

## Relacion con el paper original
- Arquitectura y notacion inspiradas en *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2020).
- El uso de `GELU`, inicializacion truncada y token CLS replica las decisiones de ese trabajo, lo cual facilita comparar resultados con publicaciones existentes.

## Pasos siguientes sugeridos
1. Entrenar `vit_tiny` en CIFAR-10/100 y observar la velocidad de convergencia frente a ResNet.
2. Implementar tecnicas adicionales como `DropPath` o `LayerScale` para mejorar estabilidad.
3. Probar interpolacion de `pos_embed` al cambiar `img_size` sin reiniciar pesos.
