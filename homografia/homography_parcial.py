import cv2
import numpy as np

image1_path = r'C:/Users/Usuario/Downloads/homografia/campus_quixada1.png'
image2_path = r'C:/Users/Usuario/Downloads/homografia/campus_quixada2.png'

# Ler as imagens
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Reduzir o tamanho das imagens para melhor visualização
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]
image1 = cv2.resize(image1, (int(w1 * 0.5), int(h1 * 0.5)))
image2 = cv2.resize(image2, (int(w2 * 0.5), int(h2 * 0.5)))

# Recalcular as dimensões após o redimensionamento
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Converter as imagens para escala de cinza
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detectar keypoints e descritores com SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Realizar a correspondência dos descritores com BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar o teste de razão de Lowe
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Verificar se há correspondências suficientes
if len(good) >= 4:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Encontrar a homografia usando RANSAC
    transformation_matrix, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)
else:
    raise AssertionError("Não há keypoints suficientes.")

corners_image1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
corners_transformed = cv2.perspectiveTransform(corners_image1, transformation_matrix)

[x_min, y_min] = np.int32(corners_transformed.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(corners_transformed.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]

translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
transformation_matrix = translation_matrix @ transformation_matrix

img1_transformed = cv2.warpPerspective(image1, transformation_matrix, 
                                       (x_max - x_min + width2, y_max - y_min))

output_width = max(img1_transformed.shape[1], width2 + translation_dist[0])
output_height = max(img1_transformed.shape[0], height2 + translation_dist[1])
output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)


output_image[0:img1_transformed.shape[0], 0:img1_transformed.shape[1]] = img1_transformed

y_offset = translation_dist[1]
x_offset = translation_dist[0]
output_image[y_offset:y_offset + height2, x_offset:x_offset + width2] = image2

# Remover bordas escuras
gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
cropped_image = output_image[y:y + h, x:x + w]

# Mostrar a imagem combinada
cv2.imshow("Imagem Combinada", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()