# Visão Computacional: Hand Tracking com OpenCV e MediaPipe

> Projeto Python para detecção e rastreamento de mãos em tempo real, reconhecimento de gestos, teclado virtual e quadro de desenho usando OpenCV e MediaPipe. Permite interação intuitiva com o computador por gestos, incluindo controle de aplicativos e desenho virtual.

![Demonstração do sistema](https://github.com/vitoriapguimaraes/Python-HandTrackingOpenCV/blob/main/results/display.gif)

## Funcionalidades Principais

- Detecção e rastreamento de mãos em tempo real com MediaPipe
- Teclado virtual: digite tocando nas teclas virtuais com o dedo indicador
- Abrir e fechar aplicativos (Word, Excel, Firefox) com gestos da mão esquerda
- Quadro de desenho virtual: desenhe, apague e altere cor/espessura do pincel com gestos
- Feedback visual para gestos e ações detectadas

## Tecnologias Utilizadas

- Python 3
- OpenCV
- MediaPipe
- pynput
- NumPy

## Como Executar

1. Clone o repositório
2. Instale as dependências:
   ```
   pip install opencv-python mediapipe pynput numpy
   ```
3. Execute o projeto:
   ```
   python scripts/app.py
   ```

## Como Usar

- Use a mão direita para digitar no teclado virtual. Toque em uma tecla com o indicador para digitar. Levante apenas o mindinho direito para apagar.
- Use a mão esquerda para abrir/fechar aplicativos:
  - Indicador levantado: abre o Word
  - Indicador e médio levantados: abre o Excel
  - Indicador, médio e anelar levantados: abre o Firefox
  - Todos os dedos abaixados: fecha o Firefox
- Use as duas mãos para desenhar:
  - Mão esquerda define a cor do pincel (1 dedo: azul, 2: verde, 3: vermelho, 4: borracha, todos abaixados: limpa quadro)
  - Mão direita desenha com o indicador. A distância até a câmera altera a espessura do pincel.
- Pressione 'ESC' para sair.

## Estrutura de Diretórios

```
/Python-VisaoComputacionalHandTrackingOpenCV
├── class_files/
├── results/
├── scripts/
│   ├── app.py
│   └── teste_dedos.py
├── README.md
└── ...
```

## Status

- ✅ Concluído

> Veja as [issues abertas](https://github.com/vitoriapguimaraes/Python-HandTrackingOpenCV/issues) para sugestões de melhorias e próximos passos.

## Mais Sobre Mim

Acesse os arquivos disponíveis na [Pasta Documentos](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/DOCUMENTOS) para mais informações sobre minhas qualificações e certificações.
