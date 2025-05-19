# INE5430-Inteligencia-ArtificiaTrabalho-Pratico-Gato-vs-Nao-Gato
## Pre-Requesito
[python>=3.10.12](https://www.python.org/downloads/)

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
