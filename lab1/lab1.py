import cv2
import numpy as np

# Carregando a imagem
imagem = cv2.imread('C:/Users/Usuario/Downloads/lab01/gamora_nebula.jpg')

# Convertendo a imagem de BGR para HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Definindo os valores de cor azul
azul_min = np.array([67, 0, 0], np.uint8)
azul_max = np.array([114, 255, 255], np.uint8)

green_min = np.array([0, 0, 0], np.uint8)
green_max = np.array([71, 141, 255], np.uint8)

# Criando uma máscara da cor azul na imagem HSV
mascara_azul = cv2.inRange(imagem_hsv, azul_min, azul_max)

mascara_verde = cv2.inRange(imagem_hsv, green_min, green_max)

azul = cv2.bitwise_and(imagem, imagem, mask=mascara_azul)

for y in range(0, imagem_hsv.shape[0]): #percorre linhas
 for x in range(0, imagem_hsv.shape[1]): #percorre colunas
    if mascara_azul[y,x] > 0:
      imagem_hsv[y, x, 0] = 71

    elif mascara_verde[y,x] > 0:
      imagem_hsv[y,x,0] = 105

    else:
      imagem_hsv[y, x, 0] = imagem_hsv[y, x, 0]
      


# Converte de HSV para BGR
im_bgr = cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR)

# Mostrar a imagem
cv2.imshow('Imagem_Modificada', im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()