import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para mostrar a matriz da imagem usando Matplotlib
def show_image_matrix(matrix, title="Image Matrix"):
    plt.imshow(matrix, cmap='gray')
    plt.title(title)
    plt.axis('on')
    plt.show()

# Carregar a imagem
input_image_path = 'bettle.jpg'  # Substitua pelo caminho da sua imagem
output_image_path = 'bettleConv.jpg'  # Caminho para salvar a imagem de saída

# Ler a imagem
image = cv2.imread(input_image_path)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print(f"Erro: não foi possível carregar a imagem a partir do caminho {input_image_path}")
else:
    # Converter a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image_matrix(gray_image, "Gray Image Matrix")

    # Aplicar o filtro de detecção de bordas verticais (filtro Sobel)
    # Kernel para detectar bordas verticais
    sobel_vertical = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    # Aplicar a convolução
    edges = cv2.filter2D(gray_image, -1, sobel_vertical)
    show_image_matrix(edges, "Edges Image Matrix")

    # Salvar a imagem resultante
    cv2.imwrite(output_image_path, edges)

    print(f"Imagem de bordas verticais salva em: {output_image_path}")
