import numpy as np

class RedutorImagem():
    __energia_maxima = 1000000.0

    def __init__(self, imagem):
        self.__matriz = imagem.astype(int)
        self.__altura, self.__largura = imagem.shape[:2]
        self.__matriz_energia = np.empty((self.__altura, self.__largura))
        self.__calcular_matriz_energia()

    def __eh_borda(self, x, y):
        return (x == 0 or x == self.__altura - 1) or (y == 0 or y == self.__largura - 1)

    def __calcular_energia(self, x, y):
        if self.__eh_borda(x, y):
            return self.__energia_maxima

        canal_azul = abs(self.__matriz[x - 1, y, 0] - self.__matriz[x + 1, y, 0])
        canal_verde = abs(self.__matriz[x - 1, y, 1] - self.__matriz[x + 1, y, 1])
        canal_vermelho = abs(self.__matriz[x - 1, y, 2] - self.__matriz[x + 1, y, 2])

        canal_azul += abs(self.__matriz[x, y - 1, 0] - self.__matriz[x, y + 1, 0])
        canal_verde += abs(self.__matriz[x, y - 1, 1] - self.__matriz[x, y + 1, 1])
        canal_vermelho += abs(self.__matriz[x, y - 1, 2] - self.__matriz[x, y + 1, 2])

        energia_total = canal_azul + canal_verde + canal_vermelho

        return energia_total

    def __trocar_eixos(self):
        self.__matriz_energia = np.swapaxes(self.__matriz_energia, 0, 1)
        self.__matriz = np.swapaxes(self.__matriz, 0, 1)
        self.__altura, self.__largura = self.__largura, self.__altura

    def __calcular_matriz_energia(self):
        self.__matriz_energia[[0, -1], :] = self.__energia_maxima
        self.__matriz_energia[:, [0, -1]] = self.__energia_maxima

        self.__matriz_energia[1:-1, 1:-1] = np.add.reduce(
            np.abs(self.__matriz[:-2, 1:-1] - self.__matriz[2:, 1:-1]), -1)
        self.__matriz_energia[1:-1, 1:-1] += np.add.reduce(
            np.abs(self.__matriz[1:-1, :-2] - self.__matriz[1:-1, 2:]), -1)

    def __calcular_caminho(self, horizontal=False):
        if horizontal:
            self.__trocar_eixos()

        matriz_acumulada = np.empty_like(self.__matriz_energia)

        matriz_acumulada[0] = self.__matriz_energia[0]
        for i in range(1, self.__altura):
            matriz_acumulada[i, :-1] = np.minimum(
                matriz_acumulada[i - 1, :-1], matriz_acumulada[i - 1, 1:])
            matriz_acumulada[i, 1:] = np.minimum(
                matriz_acumulada[i, :-1], matriz_acumulada[i - 1, 1:])
            matriz_acumulada[i] += self.__matriz_energia[i]

        caminho = np.empty(self.__altura, dtype=int)
        caminho[-1] = np.argmin(matriz_acumulada[-1, :])
        energia_caminho = matriz_acumulada[-1, caminho[-1]]

        for i in range(self.__altura - 2, -1, -1):
            esq, dir = max(0, caminho[i + 1] - 1), min(caminho[i + 1] + 2, self.__largura)
            caminho[i] = esq + np.argmin(matriz_acumulada[i, esq: dir])

        if horizontal:
            self.__trocar_eixos()

        return (energia_caminho, caminho)

    def __reduzir(self, horizontal=False, caminho=None, remover=True):
        if horizontal:
            self.__trocar_eixos()

        if caminho is None:
            caminho = self.__calcular_caminho()[1]

        if remover:
            self.__largura -= 1
        else:
            self.__largura += 1

        nova_matriz = np.empty((self.__altura, self.__largura, 3))
        nova_matriz_energia = np.empty((self.__altura, self.__largura))
        pixels_removidos = 0

        for i, j in enumerate(caminho):
            if remover:
                if self.__matriz_energia[i, j] < 0:
                    pixels_removidos += 1
                nova_matriz_energia[i] = np.delete(self.__matriz_energia[i], j)
                nova_matriz[i] = np.delete(self.__matriz[i], j, 0)
            else:
                nova_matriz_energia[i] = np.insert(self.__matriz_energia[i], j, 0, 0)

                novo_pixel = self.__matriz[i, j]
                if not self.__eh_borda(i, j):
                    novo_pixel = (self.__matriz[i, j - 1] + self.__matriz[i, j + 1]) // 2

                nova_matriz[i] = np.insert(self.__matriz[i], j, novo_pixel, 0)

        self.__matriz = nova_matriz
        self.__matriz_energia = nova_matriz_energia

        for i, j in enumerate(caminho):
            for k in range(j - 1, j + 1):
                if 0 <= k < self.__largura and self.__matriz_energia[i, k] >= 0:
                    self.__matriz_energia[i, k] = self.__calcular_energia(i, k)

        if horizontal:
            self.__trocar_eixos()

        return pixels_removidos

    def redimensionar(self, nova_altura=None, nova_largura=None):
        if nova_altura is None:
            nova_altura = self.__altura
        if nova_largura is None:
            nova_largura = self.__largura

        while self.__largura != nova_largura:
            self.__reduzir(horizontal=False, remover=self.__largura > nova_largura)
        while self.__altura != nova_altura:
            self.__reduzir(horizontal=True, remover=self.__altura > nova_altura)

    def remover_mascara(self, mascara):
        pixels_contados = np.count_nonzero(mascara)

        self.__matriz_energia[mascara] *= -(self.__energia_maxima ** 2)
        self.__matriz_energia[mascara] -= (self.__energia_maxima ** 2)

        while pixels_contados:
            energia_vertical, caminho_vertical = self.__calcular_caminho(False)
            energia_horizontal, caminho_horizontal = self.__calcular_caminho(True)

            horizontal, caminho = False, caminho_vertical

            if energia_vertical > energia_horizontal:
                horizontal, caminho = True, caminho_horizontal

            pixels_contados -= self.__reduzir(horizontal, caminho)

    def obter_imagem(self):
        return self.__matriz.astype(np.uint8)
