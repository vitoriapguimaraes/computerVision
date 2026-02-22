# Classificação de Imagens com Redes Neurais Convolucionais e TensorFlow

> Este projeto desenvolve um modelo de Inteligência Artificial baseado em Redes Neurais Convolucionais (CNNs) para classificar imagens em 10 categorias distintas, utilizando o dataset CIFAR-10. O objetivo é permitir que o modelo reconheça corretamente novas imagens pertencentes a categorias como avião, carro, gato, navio, entre outras, promovendo uma aplicação prática de aprendizado profundo.

## Funcionalidades Principais

- Carregamento e pré-processamento do dataset CIFAR-10
- Visualização de imagens do conjunto de dados para análise inicial
- Construção de um modelo CNN com camadas de convolução, pooling e densas
- Treinamento do modelo com ajuste de hiperparâmetros
- Avaliação do desempenho do modelo no conjunto de teste
- Classificação de novas imagens externas

## Tecnologias Utilizadas

- Python
- TensorFlow/Keras
- Matplotlib
- Pillow (PIL)
- NumPy

## Como Executar

1. Instale as dependências que estão na raiz do repositório (recomenda-se ambiente virtual):

   ```bash
   pip install -r ../requirements.txt
   ```

2. Execute o projeto:
   - Para rodar o notebook:
     Abra o arquivo `scripts/app.ipynb` em um ambiente Jupyter Notebook ou Google Colab.
   - Para rodar o script Python:

     ```bash
     python scripts/app.py
     ```

## Como Usar

- Execute o notebook ou script conforme instruções acima.
- O modelo irá carregar, treinar e avaliar automaticamente.
- Para classificar uma nova imagem, coloque o arquivo desejado na pasta `images/` e ajuste o caminho no código.
- O resultado da classificação será exibido no terminal ou notebook.
