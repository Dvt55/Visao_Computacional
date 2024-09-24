import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para converter uma imagem colorida para tons de cinza
def rgb_to_gray(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    
    gray = 0.3 * R + 0.59 * G + 0.11 * B
    gray = np.uint8(gray)  
    
    return gray


image_path = r'C:/Users/Usuario/Downloads/lab3/lab3_fix/brasil.jpg'
color_img = cv2.imread(image_path)

gray_img = rgb_to_gray(color_img)

blur_median = cv2.blur(gray_img, (5, 5))

gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 1)

median_blur = cv2.medianBlur(gray_img, 5)


# Ajustar o layout para 2x3 (6 espaços)
plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title('Img Colorida')
plt.subplot(232), plt.imshow(gray_img, cmap='gray'), plt.title('Img em Tons de Cinza')
plt.subplot(233), plt.imshow(blur_median, cmap='gray'), plt.title('Média')
plt.subplot(234), plt.imshow(gaussian_blur, cmap='gray'), plt.title('Gaussiano')
plt.subplot(235), plt.imshow(median_blur, cmap='gray'), plt.title('Median Blur')

plt.tight_layout()
plt.show()
