# Detecção de sonolência com OpenCV e MediaPipe

> Sistema de detecção de sonolência em motoristas usando visão computacional, OpenCV e MediaPipe. O projeto visa aumentar a segurança no trânsito ao identificar sinais de fadiga facial em tempo real, alertando o condutor para possíveis riscos.

![Demonstração do sistema](https://github.com/vitoriapguimaraes/Python-AnaliseFacialOpenCV/blob/main/results/display.gif)

## Funcionalidades Principais

- Detecção em tempo real de pontos faciais usando MediaPipe Face Mesh.
- Cálculo do índice de abertura dos olhos (EAR) e da boca (MAR).
- Identificação de piscadas e bocejos.
- Alerta visual para possíveis sinais de sonolência (ex: baixa frequência de piscadas ou olhos fechados por tempo prolongado).
- Exibição de métricas na tela: EAR, MAR, contagem de piscadas, tempo de olhos fechados.
- Código modular e fácil de adaptar para integração com sistemas automotivos ou nuvem.

## Tecnologias Utilizadas

- Python 3
- OpenCV
- MediaPipe
- NumPy

## Como Executar

1. Clone o repositório.
2. Instale as dependências:
   ```
   pip install opencv-python mediapipe numpy
   ```
3. Execute o projeto:
   ```
   python scripts/main.ipynb
   ```
   > Ou execute o notebook no Jupyter/VSCode.

## Como Usar

- Certifique-se de que sua webcam está conectada.
- Execute o notebook `scripts/main.ipynb`.
- Observe as métricas e alertas na janela de vídeo.
- Para encerrar, pressione a tecla `c` na janela da câmera.

## Estrutura de Diretórios

```
/Python-VisaoComputacionalAnaliseFacialOpenCV
├── class_files/         # Notebooks de aula, PDFs e anotações
├── scripts/             # Notebook principal do projeto
│   └── main.ipynb
├── README.md
└── LICENSE
```

## Status

✅ Concluído

> Veja as [issues abertas](https://github.com/vitoriapguimaraes/Python-AnaliseFacialOpenCV/issues) para sugestões de melhorias e próximos passos.

## Mais Sobre Mim

Acesse os arquivos disponíveis na [Pasta Documentos](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/DOCUMENTOS) para mais informações sobre minhas qualificações e certificações.
