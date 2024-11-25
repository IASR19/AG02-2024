import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

def carregar_dados():
    # Carrega o dataset e ajusta os rótulos para começar em 0
    dados = pd.read_csv('wholesale.csv')
    X = dados.drop(columns=['Channel'])
    y = dados['Channel'] - 1
    return X, y

def escolher_modelo(opcao):
    # Retorna o modelo e o nome com base na escolha do usuário
    if opcao == 1:
        return DecisionTreeClassifier(), "Árvore de Decisão"
    elif opcao == 2:
        return KNeighborsClassifier(), "k-Nearest Neighbors"
    elif opcao == 3:
        return MLPClassifier(max_iter=500), "Multilayer Perceptron"
    elif opcao == 4:
        return GaussianNB(), "Naive Bayes"
    else:
        return None, None

def exibir_matriz_confusao(y_teste, y_pred):
    # Gera e exibe a matriz de confusão
    matriz = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["HoReCa", "Retail"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show(block=True)

def exibir_grafico_personalizado(valores_digitados, classificacoes):
    # Gera um gráfico com os valores digitados e suas classificações
    if not valores_digitados:
        print("Nenhum valor digitado para exibir no gráfico.")
        return
    
    valores_array = np.array(valores_digitados)
    x = np.arange(len(valores_digitados))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, valores_array[:, 0], c=classificacoes, cmap='coolwarm', label="Classificação")
    plt.title("Valores Digitados e suas Classificações")
    plt.xlabel("Índice")
    plt.ylabel("Primeiro Atributo dos Valores")
    plt.legend()
    plt.show()

def main():
    # Fluxo principal do programa
    print("\nBem-vindo! Vamos classificar canais de venda (HoReCa ou Retail).")

    # Carregar os dados e dividir em treino e teste
    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menu de escolha do modelo
    print("\n--- MENU PRINCIPAL ---")
    print("1. Árvore de Decisão")
    print("2. k-Nearest Neighbors")
    print("3. Multilayer Perceptron")
    print("4. Naive Bayes")
    modelo_opcao = int(input("Escolha o modelo desejado: ").strip())
    modelo, nome_modelo = escolher_modelo(modelo_opcao)

    if modelo is None:
        print("Opção inválida. Encerrando o programa.")
        return

    print(f"\nModelo escolhido: {nome_modelo}")

    # Treinar o modelo
    modelo.fit(X_treino, y_treino)

    # Testar o modelo e exibir o relatório
    y_pred = modelo.predict(X_teste)
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred))

    # Interação com o usuário para entrada de valores
    valores_digitados = []
    classificacoes = []

    while True:
        entrada = input("\nDigite os valores separados por vírgula (ou 'sair' para encerrar, 'g' para exibir matriz de confusão, 'e' para exibir gráfico personalizado): ").strip()
        if entrada.lower() == 'sair':
            print("Encerrando o programa. Até mais!")
            break
        elif entrada.lower() == 'g':
            # Exibir a matriz de confusão
            exibir_matriz_confusao(y_teste, y_pred)
        elif entrada.lower() == 'e':
            # Exibir o gráfico personalizado
            exibir_grafico_personalizado(valores_digitados, classificacoes)
        else:
            try:
                # Processar os valores digitados e classificar
                valores = np.array([float(x) for x in entrada.split(',')]).reshape(1, -1)
                resultado = modelo.predict(valores)[0]
                print(f"Resultado: {'HoReCa' if resultado == 0 else 'Retail'}")
                valores_digitados.append(valores[0])
                classificacoes.append(resultado)
            except ValueError:
                print("Erro nos valores. Insira os dados corretamente.")

if __name__ == "__main__":
    # Configuração para backend do matplotlib
    os.environ['MPLBACKEND'] = 'TkAgg'
    main()
