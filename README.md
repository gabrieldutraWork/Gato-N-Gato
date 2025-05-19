# Gato-vs-Nao-Gato
## Pre-Requesito
[python>=3.10.12](https://www.python.org/downloads/)

## Sobre

Este projeto tem como objetivo construir um classificador de imagens capaz de distinguir entre imagens que contêm gatos e imagens que não contêm gatos.

Utilizando técnicas de aprendizado de máquina e visão computacional, o sistema foi treinado com um conjunto de dados rotulado, aplicando pré-processamento de imagens, extração de características e algoritmos de classificação supervisionada.

- O projeto demonstra os principais conceitos:
  - Manipulação e preparação de dados visuais
  - Treinamento e avaliação de modelos de classificação
  - Uso de métricas de desempenho para validação dos resultados.

## Dependências
Instale as dependências do projeto:
```bash
  pip install numpy h5py scikit-learn seaborn matplotlib tensorflow
```


## Execução

### Regressão Logística
- Teste1
  - Utilizando 10 imagens do conjunto de testes.
  ```bash
    python3 RegrecaoLogistica/Teste1.py
  ```
- Teste2
  - Utilizando 80 imagens do conjunto de testes.
  ```bash
    python3 RegrecaoLogistica/Teste2.py
  ```
- Teste3
  - Utilizando 209 imagens do conjunto de testes.
  ```bash
    python3 RegrecaoLogistica/Teste3.py
  ```

### Rede de camada rasa

- Teste1
  - Camada de entrada: 64x64x3 vetores.
  - Camada intermediária: 100 neurônios com função de ativação sigmoid.
  - Camada de saída: 1 neurônio com função de ativação sigmoid.
    ```bash
      python3 RedeDeCamadaRasa/Teste1.py
    ```
- Teste2
  - Camada de entrada: 64x64x3 vetores.
  - Camada intermediária: 200 neurônios com função de ativação sigmoid.
  - Camada de saída: 1 neurônio com função de ativação sigmoid.
    ```bash
      python3 RedeDeCamadaRasa/Teste2.py
    ```
- Teste3
  - Camada de entrada: 64x64x3 vetores.
  - Camada intermediária: 100 neurônios com função de ativação ReLU.
  - Camada de saída: 1 neurônio com função de ativação sigmoid.
    ```bash
      python3 RedeDeCamadaRasa/Teste3.py
    ```
- Teste4
  - Camada de entrada: 64x64x3 vetores.
  - Camada intermediária: 200 neurônios com função de ativação ReLU.
  - Camada de saída: 1 neurônio com função de ativação sigmoid.
    ```bash
      python3 RedeDeCamadaRasa/Teste4.py
    ```
