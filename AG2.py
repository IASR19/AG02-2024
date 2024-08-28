import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Leitura dos dados do CSV "Wholesale customers"
df = pd.read_csv('/home/itamar/Área de Trabalho/AG02-2024/wholesale_customers.csv')

# 2. Mapeamento das colunas categóricas para inteiros
channel_map = {'HoReCa': 0, 'Retail': 1}
region_map = {'Lisbon': 0, 'Oporto': 1, 'Other': 2}
df['Channel'] = df['Channel'].map(channel_map)
df['Region'] = df['Region'].map(region_map)

# 3. Reorganização das colunas conforme especificado
column_order = ['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen', 'Channel']
df = df[column_order]

# 4. Visualização dos dados com gráficos de dispersão
colors = {0: 'blue', 1: 'orange'}  # Usando os valores numéricos de Channel para colorir

# Gráfico 1: Gasto com Fresh vs. Gasto com Milk
plt.figure(figsize=(10, 6))
for channel, group in df.groupby('Channel'):
    plt.scatter(group['Fresh'], group['Milk'], c=colors[channel], label=f'Channel {channel}')
plt.xlabel('Gasto com Fresh (u.m.)')
plt.ylabel('Gasto com Milk (u.m.)')
plt.title('Gasto com Fresh vs. Gasto com Milk por Canal de Vendas')
plt.legend()
plt.show()

# Gráfico 2: Gasto com Grocery vs. Gasto com Detergents_Paper
plt.figure(figsize=(10, 6))
for channel, group in df.groupby('Channel'):
    plt.scatter(group['Grocery'], group['Detergents_Paper'], c=colors[channel], label=f'Channel {channel}')
plt.xlabel('Gasto com Grocery (u.m.)')
plt.ylabel('Gasto com Detergents Paper (u.m.)')
plt.title('Gasto com Grocery vs. Gasto com Detergents Paper por Canal de Vendas')
plt.legend()
plt.show()

# Gráfico 3: Gasto com Frozen vs. Gasto com Delicatessen
plt.figure(figsize=(10, 6))
for channel, group in df.groupby('Channel'):
    plt.scatter(group['Frozen'], group['Delicatessen'], c=colors[channel], label=f'Channel {channel}')
plt.xlabel('Gasto com Frozen (u.m.)')
plt.ylabel('Gasto com Delicatessen (u.m.)')
plt.title('Gasto com Frozen vs. Gasto com Delicatessen por Canal de Vendas')
plt.legend()
plt.show()

# 5. Separar os atributos (X) da classe (y)
X = df.drop('Channel', axis=1)
y = df['Channel']

# 6. Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Treinamento dos modelos de ML
clf_dt = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_mlp = MLPClassifier(random_state=42)
clf_nb = GaussianNB()

# Treinamento dos modelos
clf_dt.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)
clf_mlp.fit(X_train, y_train)
clf_nb.fit(X_train, y_train)

# 8. Avaliação dos modelos
y_pred_dt = clf_dt.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)
y_pred_mlp = clf_mlp.predict(X_test)
y_pred_nb = clf_nb.predict(X_test)

print("Árvore de Decisão:")
print(classification_report(y_test, y_pred_dt))
print("\nKNN:")
print(classification_report(y_test, y_pred_knn))
print("\nMLP:")
print(classification_report(y_test, y_pred_mlp))
print("\nNaive Bayes:")
print(classification_report(y_test, y_pred_nb))

# 9. Função para previsão com base na entrada do usuário
def predict_channel_with_input(region, fresh, milk, grocery, frozen, detergents_paper, delicatessen):
    # Fazendo a previsão
    prediction = clf_dt.predict([[region, fresh, milk, grocery, frozen, detergents_paper, delicatessen]])[0]
    
    # Traduzindo a previsão
    inverse_channel_map = {v: k for k, v in channel_map.items()}
    predicted_channel = inverse_channel_map[prediction]
    
    # Mostrando o resultado
    print(f"O canal de vendas previsto é: {predicted_channel}")

# Exemplo de uso
region_input = int(input("Digite o número da região (0 para Lisbon, 1 para Oporto, 2 para Other): "))
fresh_input = float(input("Digite o gasto anual com produtos frescos (u.m.): "))
milk_input = float(input("Digite o gasto anual com laticínios (u.m.): "))
grocery_input = float(input("Digite o gasto anual com produtos de mercearia (u.m.): "))
frozen_input = float(input("Digite o gasto anual com produtos congelados (u.m.): "))
detergents_paper_input = float(input("Digite o gasto anual com detergentes e produtos de papel (u.m.): "))
delicatessen_input = float(input("Digite o gasto anual com guloseimas (u.m.): "))

predict_channel_with_input(region_input, fresh_input, milk_input, grocery_input, frozen_input, detergents_paper_input, delicatessen_input)
