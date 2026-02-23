# 👁️‍🗨️ Computer Vision Hub

> **A fronteira da percepção computacional.**  
> Um Centro de Operações unificado que apresenta projetos focados em processamento de imagens, detecção de objetos e reconhecimento interativo usando OpenCV, MediaPipe e Deep Learning.

![Demonstração do sistema](https://github.com/vitoriapguimaraes/computerVision/blob/main/results/display.gif)

## Funcionalidades Principais

- **Classificação de Imagens**: Classificação de imagens baseada em CNN treinada no dataset CIFAR-10.
- **Análise de Tráfego**: Contagem automática de veículos usando algoritmos de subtração de fundo.
- **Interação Humano-Máquina**: Interfaces sem toque utilizando detecção de pontos de referência da mão em tempo real.
- **Segurança Viária**: Detecção de fadiga de motoristas monitorando o Eye Aspect Ratio (EAR).
- **Interface Centralizada**: Todos os algoritmos rodam a partir de um único painel interativo Streamlit em estilo "CCTV".

## Tecnologias Utilizadas

- **Linguagem:** Python (Recomendado fortemente o uso da versão **3.10** para evitar conflitos de dependência com MediaPipe/TensorFlow)
- **Framework Web:** Streamlit
- **Visão Computacional:** OpenCV, MediaPipe
- **Deep Learning:** TensorFlow, Keras
- **Visualização de Dados:** Plotly

## Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/vitoriapguimaraes/dataScience.git
   cd dataScience/computerVision
   ```

2. Instale as dependências:

   ```bash
   # É altamente recomendado criar um ambiente virtual (venv ou conda) com Python 3.10
   pip install -r requirements.txt
   ```

3. Execute o projeto:

   ```bash
   streamlit run Painel.py
   ```

## Como Usar

- Após rodar o comando do Streamlit, o hub abrirá automaticamente no seu navegador em `http://localhost:8501`.
- Navegue pelas abas na barra lateral para acessar as diferentes ferramentas de visão computacional.
- Cada ferramenta possui abas internas de "Instruções" para entender a teoria, e "Execução" para ligar a câmera/fazer uploads reais.

## Estrutura de Diretórios

As pastas individuais dos projetos antigos de CV continuam disponíveis (`projects/`), mas sua lógica interativa agora está integrada nesta aplicação central.

```dash
computerVision/
├── projects/                        # Lógica original e scripts isolados dos projetos
├── assets/                          # Imagens e GIFs de demonstração
├── models/                          # Pesos dos modelos treinados (ex: H5)
├── pages/                           # Páginas do Hub Central
│   ├── 1_Image_Classification.py
│   ├── 2_Traffic_Analysis.py
│   ├── 3_Human_Machine_Interaction.py
│   └── 4_Road_Safety.py
├── utils/                           # Componentes e utilitários compartilhados
│   ├── ui.py
│   ├── config.py
│   └── hand_tracking.py
├── Painel.py                        # Dashboard de Operações (Home)
├── requirements.txt                 # Dependências do Hub (Requer Python 3.10)
└── README.md
```

## Status

Em manutenção

## Mais Sobre Mim

Acesse os arquivos disponíveis na [Pasta Documentos](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/DOCUMENTOS) para mais informações sobre minhas qualificações e certificações.
