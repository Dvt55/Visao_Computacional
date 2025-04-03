import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo
model = load_model("model_amb.h5")

# Inicializar a webcam (0 é geralmente a câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Pré-processamento do frame (igual ao que você fez com a imagem)
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = cv2.resize(processed_frame, (64, 64))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    
    # Fazer a previsão
    prediction = model.predict(processed_frame)
    
    # Determinar a classe
    if prediction[0][0] > 0.5:
        label = "Reciclavel (R)"
        color = (0, 255, 0)  # Verde
    else:
        label = "Organico (O)"
        color = (0, 0, 255)  # Vermelho
    
    # Mostrar o resultado no frame original
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Mostrar o frame
    cv2.imshow('Webcam - Classificacao', frame)
    
    # Sair do loop se pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e destruir as janelas
cap.release()
cv2.destroyAllWindows()