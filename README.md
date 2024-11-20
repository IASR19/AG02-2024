# Classificador de Canais de Venda (HoReCa ou Retail)

Este programa utiliza **Machine Learning** para classificar canais de venda em dois grupos: **HoReCa** (Hotéis, Restaurantes e Cafés) ou **Retail** (Varejo). Ele permite treinar diferentes modelos, visualizar métricas de desempenho e realizar predições com base em dados inseridos pelo usuário.


# Funcionalidades

1. **Escolha de Modelos de Machine Learning:**
   - Árvore de Decisão
   - k-Nearest Neighbors (k-NN)
   - Multilayer Perceptron (MLP)
   - Naive Bayes

2. **Treinamento e Avaliação:**
   - O programa treina o modelo escolhido e avalia seu desempenho, exibindo um **relatório de classificação** (precision, recall, f1-score).

3. **Predição com Dados Inseridos:**
   - O usuário pode inserir valores manualmente e receber a classificação correspondente (HoReCa ou Retail).

4. **Visualização de Resultados:**
   - Geração de uma **matriz de confusão** para análise de desempenho.
   - Geração de um **gráfico personalizado** com os valores inseridos pelo usuário e suas respectivas classificações.


# Como Usar

### Execução

1. Certifique-se de que o arquivo `wholesale.csv` está no mesmo diretório do script.
2. Execute o programa:
```bash
   python AG2.py
```

3. Escolha o modelo de treinamento no menu:
    - `1`: Árvore de Decisão
    - `2`: k-Nearest Neighbors
    - `3`: Multilayer Perceptron
    - `4`: Naive Bayes

4. Após o treinamento:
    - Digite valores para classificação no formato:
      `region,fresh,milk,grocery,frozen,detergents_paper,delicatessen`.
    - Utilize as seguintes opções especiais:
        - `g`: Exibir matriz de confusão.
        - `e`: Exibir gráfico com os valores digitados e suas classificações.
        - `sair`: Encerrar o programa.
    - O programa espera os seguintes valores (em ordem):
        1. `region`: Região (0 = Lisbon, 1 = Oporto, 2 = Other).
        2. `fresh`: Gasto anual em produtos frescos.
        3. `milk`: Gasto anual em leite.
        4. `grocery`: Gasto anual em mercearia.
        5. `frozen`: Gasto anual em produtos congelados.
        6. `detergents_paper`: Gasto anual em detergentes e papel.
        7. `delicatessen`: Gasto anual em guloseimas.


# Requisitos

- Python 3.x
- Bibliotecas:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn



## Desenvolvido por:
### Itamar Augusto Silva Ribeiro - GES - Matrícula: 91
### Lucas Ribeiro de Martha - GES - 198
