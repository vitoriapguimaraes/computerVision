# Detecção de sonolência com OpenCV e MediaPipe

> Sistema de detecção de sonolência em motoristas usando visão computacional, OpenCV e MediaPipe. O projeto visa aumentar a segurança no trânsito ao identificar sinais de fadiga facial em tempo real, alertando o condutor para possíveis riscos.

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

## Como Usar

- Certifique-se de que sua webcam está conectada.
- Execute o notebook `scripts/main.ipynb`.
- Observe as métricas e alertas na janela de vídeo.
- Para encerrar, pressione a tecla `c` na janela da câmera.
