import cv2

# Caminhos da imagem e da máscara
caminho_imagem = r"C:/Users/Usuario/Downloads/seam_carving/seam_carving_remove/Example/bolas.jpg"
caminho_mascara = r"C:/Users/Usuario/Downloads/mask.png"

# Carregar a imagem e a máscara
imagem = cv2.imread(caminho_imagem)
mascara = cv2.imread(caminho_mascara, 0)  # Carrega a máscara em escala de cinza

# Verificar as dimensões da imagem e da máscara
altura_imagem, largura_imagem = imagem.shape[:2]
altura_mascara, largura_mascara = mascara.shape[:2]

# Redimensionar a máscara se as dimensões não coincidirem
if (altura_imagem, largura_imagem) != (altura_mascara, largura_mascara):
    mascara_redimensionada = cv2.resize(mascara, (largura_imagem, altura_imagem))  # Redimensiona a máscara para o tamanho da imagem
else:
    mascara_redimensionada = mascara

# Converter a máscara para booleano (não 255 == True)
mascara_booleana = mascara_redimensionada != 255

# Aplicar o RedutorImagem
from seam import RedutorImagem
redutor_imagem = RedutorImagem(imagem)
redutor_imagem.remover_mascara(mascara_booleana)

# Exibir a imagem original e a imagem processada
cv2.imshow('Original', imagem)
cv2.imshow('Removida', redutor_imagem.obter_imagem())
cv2.waitKey(0)
cv2.destroyAllWindows()
