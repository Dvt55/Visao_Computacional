import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho_imagem = r"C:/Users/Usuario/Downloads/seam_carving/seam_carving_remove/Example/bolas.jpg"
imagem_original = cv2.imread(caminho_imagem)

imagem_redimensionada = cv2.resize(imagem_original, (400, 400))

imagem_hsv = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2HSV)

limite_inferior_vermelho1 = np.array([0, 120, 70])
limite_superior_vermelho1 = np.array([10, 255, 255])

mascara1 = cv2.inRange(imagem_hsv, limite_inferior_vermelho1, limite_superior_vermelho1)

limite_inferior_vermelho2 = np.array([170, 120, 70])
limite_superior_vermelho2 = np.array([180, 255, 255])

mascara2 = cv2.inRange(imagem_hsv, limite_inferior_vermelho2, limite_superior_vermelho2)

mascara_final = mascara1 + mascara2

elemento_estruturante = np.ones((5, 5), np.uint8)
mascara_limpa = cv2.morphologyEx(mascara_final, cv2.MORPH_CLOSE, elemento_estruturante)
mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_OPEN, elemento_estruturante)

mascara_suavizada = cv2.GaussianBlur(mascara_limpa, (5, 5), 0)

mascara_invertida = cv2.bitwise_not(mascara_suavizada)

plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Máscara Invertida (Cubo Vermelho Preto)')
plt.imshow(mascara_invertida, cmap='gray')

plt.show()

cv2.imwrite("C:/Users/Usuario/Downloads/mascara_invertida.png", mascara_invertida)
