import cv2

# Intenta abrir la cámara en el índice 0
indice_camara = 0
cap = cv2.VideoCapture(indice_camara)

# Comprueba si la cámara se abrió correctamente
if not cap.isOpened():
    print(f"ERROR: No se pudo abrir la cámara en el índice {indice_camara}.")
    print("Por favor, revisa los siguientes pasos.")
else:
    print(f"ÉXITO: La cámara en el índice {indice_camara} se abrió correctamente.")
    
    while True:
        # Lee un fotograma de la cámara
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: No se pudo leer un fotograma. ¿La cámara está desconectada?")
            break
        
        # Muestra el fotograma en una ventana
        cv2.imshow('Prueba de Camara - Pulsa Q para salir', frame)
        
        # Si se pulsa la tecla 'q', sal del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Libera la cámara y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()