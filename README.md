# Classificador de Imagens 

Este é um projeto baseado no [smart-image-sorter](https://github.com/bellingcat/smart-image-sorter), desenvolvido como protótipo para testar a viabilidade de um aplicativo classificador de imagens.

## Sobre o Projeto

- **Baseado em**: [smart-image-sorter](https://github.com/bellingcat/smart-image-sorter) da Bellingcat
- **Tecnologia**: Streamlit + PyTorch
- **Objetivo**: Classificação automática de imagens

## Experimentos e Desenvolvimento

Para a escolha do modelo utilizado, foram realizados experimentos detalhados que podem ser visualizados de duas formas:

1. [Google Colab - Experimentos](https://colab.research.google.com/drive/1DDE0j19sKFudkCxvzIeZzTJE-qwqrxqc?usp=sharing)
2. Arquivo local: `Notebook de experimentos.ipynb`

## Como Executar

### Pré-requisitos

Antes de executar o projeto, você precisará ter instalado:

- Python 3.8 ou superior
- Pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
2. Instale as bibliotecas e dependências:
```bash
pip install streamlit torch torchvision pillow
pip install -r requirements.txt
```
3. Execute o aplicativo:
```bash
streamlit run App.py
```

