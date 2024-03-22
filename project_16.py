"""
=======================
# CASE DE TREINAMENTO
=======================
"""

import warnings
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=UserWarning)

# ANÁLISE EXPLORATÓRIA
# Carregamento dos Dados Inicial
dataset = pandas.read_csv('dados_pocos.csv')
dataset.columns = ["item", "nome", "fase", "tipo", "lda", "dn", "metragem", "nfases"]

# Análise das Variáveis com Dados Distintos
values_count = dataset['tipo'].value_counts()
distinct_data = dataset['tipo'].unique()

# Análise da Tipologia, Formato e Descrição dos Dados
t1 = (dataset.dtypes)
f1 = (dataset.shape)
d1 = (dataset.describe())
d2 = (dataset.describe(include=['object']))

# Análise dos Dados Nulos e Iguais a Zero
zero_data = (dataset['nfases'] == 0).sum()

# Análise dos Dados Distintos de uma Variável
dataset['tipo'].value_counts()

# Análise de dados Duplicados
duplicated_data = dataset.duplicated()
if duplicated_data.any():
    double = "YES."
else:
    double = "NO"

# Análise da correlação entre as variáveis numéricas
colunas_correlacao = ['fase', 'lda', 'metragem', 'nfases']
dataset_correlacao = dataset[colunas_correlacao]

# Carregamento dos Dados
def load_data(file_path):
    dataset = pandas.read_csv(file_path)
    dataset.columns = ["item", "nome", "fase", "tipo", "lda", "dn", "metragem", "nfases"]
    dataset = dataset[dataset['tipo'].isin(['VERTICAL', 'HORIZONTAL'])].copy()
    return dataset

# Pré-processamento dos Dados
def preprocess_data(dataset):
    label_encoding_tipo = {'VERTICAL': 1, 'HORIZONTAL': 0}
    dataset['tipo'] = dataset['tipo'].map(label_encoding_tipo)

    def converter_dn(valor):
        if len(valor) <= 2:
            return int(valor)
        else:
            if '/' in valor:
                numerador = int(valor.split()[0])
                denominador_numerico = int(valor.split()[1].split('/')[0])
                denominador_divisao = int(valor.split()[1].split('/')[1])
                return numerador + (denominador_numerico / denominador_divisao)
            else:
                return float(valor)

    dataset['dn'] = dataset['dn'].apply(converter_dn)

    label_encoder = LabelEncoder()
    dataset['codinome'] = label_encoder.fit_transform(dataset['nome'])

    return dataset

# Divisão em Dados de Treino e Teste
def split_data(dataset):
    X = dataset[['fase', 'tipo', 'lda', 'dn', 'metragem', 'nfases']]
    y = dataset['codinome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

# Treinamento do Modelo
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Função para calcular a métrica de erro
def calculate_error(input_data, closest_data):
    input_data = numpy.array(input_data)
    closest_data = numpy.array(closest_data)
    with numpy.errstate(divide='ignore', invalid='ignore'):  # Tratando divisão por zero e valores inválidos
        error = ((closest_data - input_data) / input_data) * 100
    error[numpy.isinf(error)] = 0  # Convertendo infinitos para zero
    error[numpy.isnan(error)] = 0  # Convertendo NaNs para zero
    error_sum = numpy.sum(error, axis=1)
    return error_sum

# Função para encontrar as quatro previsões mais próximas com "codinomes" distintos
def find_closest_predictions(X_test, y_test, input_data):
    input_data = [float(x) if isinstance(x, str) else x for x in input_data]  # Convertendo os dados de entrada para float
    distances = numpy.linalg.norm(X_test - input_data, axis=1)
    closest_indices = numpy.argsort(distances)
    unique_codinomes = set()
    result_indices = []
    for index in closest_indices:
        codinome = y_test.iloc[index]
        if codinome not in unique_codinomes:
            unique_codinomes.add(codinome)
            result_indices.append(index)
            if len(result_indices) == 4:
                break

    closest_data = X_test[result_indices]
    error = calculate_error(input_data, closest_data)

    # Ordenar as previsões com base no erro
    sorted_indices = numpy.argsort(error)
    closest_indices_sorted = [result_indices[i] for i in sorted_indices]
    error_sorted = error[sorted_indices]

    return closest_indices_sorted, error_sorted

# Função Principal
def main():
    file_path = 'dados_pocos.csv'
    dataset = load_data(file_path)
    dataset = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = split_data(dataset)
    model = train_model(X_train, y_train)

    print("Insira os dados de entrada:")
    fase = int(input("Fase: "))
    tipo = int(input("Tipo: "))
    lda = float(input("Lda: "))
    dn = float(input("Dn: "))
    metragem = float(input("Metragem: "))
    nfases = int(input("Nfases: "))

    input_data = [fase, tipo, lda, dn, metragem, nfases]

    closest_indices, error = find_closest_predictions(X_test.to_numpy(), y_test, input_data)

    print("Resultados das 4 Previsões mais Próximas (Codinome) e Erro:")
    # Ordenar resultados por erro decrescente
    sorted_results = sorted(zip(closest_indices, error), key=lambda x: x[1], reverse=True)
    for index, err in sorted_results:
        row_data = list(X_test.iloc[index])  # Dados da linha correspondente ao codinome sugerido
        prediction = y_test.iloc[index]
        print(f"Codinome: {prediction}, Erro: {err:.2f}%, Dados: {row_data}")

if __name__ == "__main__":
    main()
