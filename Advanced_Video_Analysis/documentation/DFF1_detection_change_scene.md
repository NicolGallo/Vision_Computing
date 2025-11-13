1. Documentación para la Exposición

    📽️ Aplicación 1: Detector de Cambios de Escena (Shot Boundary Detection)

        Objetivo: Identificar automáticamente los fotogramas exactos donde ocurre un "corte brusco" (hard cut) en un vídeo, es decir, un cambio instantáneo de una escena a otra.

        ¿Cómo Funciona?

            Esta aplicación reutiliza la lógica de los "Keyframes Adaptativos" (AKR) que ya existía en el script de DFF.

            La lógica adaptativa ya estaba diseñada para detectar cambios grandes entre fotogramas y forzar un keyframe. Nosotros simplemente vamos a capturar esos "disparadores" y presentarlos como el resultado final.

        Métricas Clave (Las mismas que AKR):

            Error Cuadrático Medio (MSE): Compara la diferencia píxel a píxel en escala de grises entre el frame actual y el anterior. Un pico en el MSE indica un cambio visual grande.

            Magnitud del Flujo Óptico: Mide cuánto se están moviendo los píxeles en promedio. Un cambio de escena también causa un flujo óptico caótico y de gran magnitud.

        Aspectos a Destacar:

            Eficiencia: Es muy rápido. No requiere una red neuronal pesada (como el backbone de DFF), solo cálculos de OpenCV que pueden correr en CPU.

            Reutilización de Lógica: Demuestra cómo una sub-componente de un sistema complejo (la lógica AKR de DFF) puede aislarse para crear una aplicación completamente nueva y útil por sí misma.

            Aplicación Práctica: Esto se usa para indexar vídeos, crear capítulos automáticamente, detectar anuncios, etc.

    ¡Excelente! Este resultado es perfecto para tu presentación, porque ilustra un punto fundamental sobre este tipo de algoritmos.


Lo que has encontrado no son "cortes de escena", sino "falsos positivos" provocados por el movimiento dentro de la escena.

Este es un concepto clave:

El Umbral (Threshold) Define la "Sensibilidad"

Piensa en los umbrales (--diff_thresh y --flow_thresh) como un control de sensibilidad, o como el termostato de una calefacción:

    Umbral Alto (ej. 900): Es como poner el termostato a 30°C. Solo se activará si hay un "fuego" (un cambio de escena real y masivo). El movimiento normal (una estocada) no es suficiente. Por eso con 900, obtuviste 0 cortes.

    Umbral Bajo (ej. 500): Es como poner el termostato a 20°C. Ahora, un "fuego" (corte real) lo activará, pero también lo hará un día cálido (un movimiento rápido de cámara o una estocada del esgrimista).

¿Qué Muestra Tu Resultado?

Tu vídeo vfencing.mp4 es una escena continua, pero tiene movimientos rápidos de cámara y estocadas (lunges).

    En el frame 9, hubo un movimiento lo suficientemente brusco como para causar un MSE de 552.5.

    Esto es menor que el umbral original de 900 (por lo que fue ignorado), pero mayor que tu nuevo umbral de 500 (por lo que fue detectado).

Has demostrado perfectamente el equilibrio (trade-off) clásico en la detección:

    Si pones el umbral MUY ALTO: Corres el riesgo de no detectar cortes de escena reales que sean sutiles (Falsos Negativos).

    Si pones el umbral MUY BAJO: Empiezas a detectar erróneamente movimientos rápidos dentro de la misma escena, confundiéndolos con cortes (Falsos Positivos).

💡 Para tu Presentación (¡Esto es oro!)

Te recomiendo que muestres ambos resultados:

    Test 1 (El vídeo vfencing.mp4):

        Muestra el resultado con el umbral alto (900) -> "Detecta 0 cortes, lo cual es correcto, es una escena continua."

        Muestra este resultado con el umbral bajo (500) -> "Detecta 4 'cortes'. Si vemos los frames 9, 45, etc., vemos que no son cortes, sino estocadas. Esto demuestra cómo un ajuste de sensibilidad incorrecto crea falsos positivos."

    Test 2 (El vídeo test_cuts.mp4 que crearás):

        Usa el umbral alto (900). El script detectará el corte real que tú insertaste.

        Esto demuestra que el umbral alto sí funciona para lo que fue diseñado: encontrar cortes bruscos reales.

Hacer esto demuestra que entiendes perfectamente el algoritmo y sus parámetros.