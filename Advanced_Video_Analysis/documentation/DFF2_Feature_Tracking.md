🖱️ Aplicación 2: Seguimiento de Puntos de Interés (Feature Tracking)

    Objetivo: Permitir al usuario hacer clic en un píxel arbitrario (ej. la punta de una espada, el ojo de una persona) en el primer frame, y que el sistema siga ese punto a lo largo del vídeo.

    ¿Cómo Funciona?

        Selección (Frame 0): El usuario hace clic en un punto (x, y) del frame inicial.

        Extracción de Keyframe: Ejecutamos el costoso backbone (ResNet50) una sola vez sobre este frame inicial para obtener el mapa de características clave (key_feats).

        Vector Objetivo: Convertimos el clic (x, y) a coordenadas del mapa de características (ej. la celda (iy, ix) en la cuadrícula 7x7) y extraemos ese vector de características (ej. de 2048 dimensiones). Este es nuestro "vector objetivo" (V_target).

        Propagación (Frames 1...N): Para cada frame siguiente, no usamos el backbone. Usamos el DFF para propagar las key_feats usando el flujo óptico.

        Búsqueda (Tracking): En este nuevo mapa de características propagado, calculamos la "distancia" (ej. L2) entre nuestro V_target y todos los demás vectores de la cuadrícula 7x7.

        Mejor Coincidencia: La celda que tenga la distancia más baja (la más similar) a nuestro V_target es la nueva posición del objeto.

    Aspectos a Destacar:

        Eficiencia (El "Por Qué" de DFF): El cálculo costoso (ResNet) se hace 1 sola vez. El resto de los 300 frames son rápidos (Flujo Óptico + Búsqueda L2), logrando un seguimiento en (casi) tiempo real.

        Robustez (Seguimiento Semántico): No estamos siguiendo "píxeles grises". Estamos siguiendo una "firma semántica". V_target no significa "color gris claro"; significa "punta de espada" o "nariz de persona". Esto lo hace muy robusto a cambios de iluminación o rotaciones leves, a diferencia de los rastreadores clásicos de OpenCV.

        Zero-Shot / Sin Entrenamiento: Esta funcionalidad no requiere ningún entrenamiento. Usamos las características pre-entrenadas de ResNet50 (entrenado en ImageNet) "tal cual".