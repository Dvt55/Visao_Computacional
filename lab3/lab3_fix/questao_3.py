import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_sepia(img):
    sepia_img = np.copy(img)

    for y in range(sepia_img.shape[0]):
        for x in range(sepia_img.shape[1]):
            r, g, b = sepia_img[y, x]
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b

            sepia_img[y, x] = [min(tb, 255), min(tg, 255), min(tr, 255)]

    return sepia_img

image_path = r'C:/Users/Usuario/Downloads/lab3/lab3_fix/brasil.jpg'
color_img = cv2.imread(image_path)

sepia_img = apply_sepia(color_img)

blur_median = cv2.blur(sepia_img, (5, 5))

gaussian_blur = cv2.GaussianBlur(sepia_img, (5, 5), 1)

median_blur = cv2.medianBlur(sepia_img, 5)

# Ajustar o layout para 2x3 (6 espaços)
plt.figure(figsize=(12, 8))

plt.subplot(231), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title('Img Colorida')
plt.subplot(232), plt.imshow(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB)), plt.title('Efeito Sépia')
plt.subplot(233), plt.imshow(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB)), plt.title('Média')
plt.subplot(234), plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)), plt.title('Gaussiano')
plt.subplot(235), plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)), plt.title('Median Blur')

plt.tight_layout()
plt.show()
