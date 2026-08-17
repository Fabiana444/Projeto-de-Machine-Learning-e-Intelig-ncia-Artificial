# Projeto-de-Machine-Learning-e-Inteligencia-Artificial
Projeto de Machine Learning para previsão de cancelamentos de reservas hoteleiras, utilizando análise exploratória de dados, tratamento e Engenharia de Features, classificação e métricas de avaliação.

## Etapas desenvolvidas no projeto


### 🧹 **Tratamento e preparação dos dados**

- [x] Importação da biblioteca Pandas
- [x] Carregamento do dataset Hotel Booking Demand
- [x] Visualização inicial dos dados com `df.head()`
- [x] Verificação das dimensões do dataset
- [x] Identificação das colunas disponíveis
- [x] Verificação da estrutura e dos tipos de dados com `df.info()`
- [x] Identificação e quantificação de valores ausentes
- [x] Cálculo do percentual de valores ausentes por coluna
- [x] Verificação de registros duplicados
- [x] Cálculo do percentual de registros duplicados
- [x] Investigação dos registros duplicados
- [x] Remoção dos registros duplicados
- [x] Verificação do dataset após a remoção dos duplicados
- [x] Criação de uma cópia do DataFrame original
- [x] Criação do DataFrame `df_tratado`
- [x] Preservação do DataFrame original para evitar alterações no dataset carregado
- [x] Tratamento dos valores ausentes da coluna `children` utilizando a mediana
- [x] Análise da distribuição da coluna `country`
- [x] Tratamento dos valores ausentes de `country` com `Unknown`
- [x] Análise da coluna `agent`
- [x] Tratamento dos valores ausentes de `agent` com `Unknown`
- [x] Análise da coluna `company`
- [x] Remoção da coluna `company` devido à elevada quantidade de valores ausentes
- [x] Verificação dos tipos de dados após os tratamentos
- [x] Conversão de `reservation_status_date` de texto para `datetime`
- [x] Verificação da conversão da variável temporal
- [x] Verificação de valores ausentes após a conversão da data
- [x] Identificação de possíveis outliers utilizando o método IQR
- [x] Análise estatística das principais variáveis numéricas
- [x] Investigação de valores negativos na variável `adr`
- [x] Remoção do registro inconsistente com `adr` negativo
- [x] Investigação de valores extremos na variável `adults`
- [x] Identificação de reservas com mais de 10 adultos
- [x] Remoção dos registros considerados inconsistentes na variável `adults`
- [x] Investigação do valor extremo `adr = 5400`
- [x] Remoção do registro com `adr = 5400`
- [x] Verificação final dos valores ausentes
- [x] Verificação final dos tipos de dados
- [x] Verificação da dimensão final do DataFrame tratado
      

### 🔎 **Análise Exploratória de Dados (EDA)**

- [x] Definição da variável-alvo `is_canceled`
- [x] Verificação dos valores existentes no TARGET
- [x] Análise da distribuição do TARGET
- [x] Cálculo da distribuição percentual do TARGET
- [x] Análise da distribuição das reservas por tipo de hotel
- [x] Comparação de cancelamentos e não cancelamentos por tipo de hotel
- [x] Cálculo da taxa de cancelamento por tipo de hotel
- [x] Análise da relação entre `lead_time` e `is_canceled`
- [x] Cálculo de estatísticas de `lead_time` por situação de cancelamento
- [x] Criação de faixas de antecedência da reserva
- [x] Análise da taxa de cancelamento por faixa de antecedência
- [x] Criação de gráfico da taxa de cancelamento por faixa de antecedência
- [x] Análise da sazonalidade dos cancelamentos
- [x] Organização dos meses em ordem cronológica
- [x] Cálculo da taxa de cancelamento por mês
- [x] Criação de gráfico da taxa de cancelamento por mês
- [x] Análise da evolução das reservas/cancelamentos por ano
- [x] Cálculo da taxa de cancelamento por ano
- [x] Criação de gráfico da taxa de cancelamento por ano
- [x] Análise da relação entre `adr` e cancelamento
- [x] Comparação das estatísticas de `adr` entre reservas canceladas e não canceladas
- [x] Investigação do valor extremo `adr = 5400`
- [x] Criação de boxplot de `adr` por status de cancelamento
- [x] Análise do país de origem dos hóspedes
- [x] Identificação dos 15 países com maior número de reservas
- [x] Cálculo da taxa de cancelamento para os 15 países com mais reservas
- [x] Criação de gráfico da taxa de cancelamento por país de origem
- [x] Identificação de padrões e tendências
- [x] Registro dos principais insights da análise exploratória
      

### 🛠️ **Engenharia de Features**

- [x] Criação da variável derivada `lead_time_faixa`
- [x] Agrupamento de `lead_time` em faixas de antecedência
- [x] Utilização da variável derivada na análise exploratória
- [x] Definição de `is_canceled` como variável-alvo (TARGET)
- [x] Definição das variáveis preditoras (X)
- [x] Exclusão de `is_canceled` das variáveis de entrada
- [x] Exclusão de `reservation_status` para evitar vazamento de dados (data leakage)
- [x] Exclusão de `reservation_status_date` para evitar utilização de informação posterior ao momento da reserva
- [x] Exclusão de `lead_time_faixa` das entradas do modelo, pois `lead_time` foi utilizado diretamente como variável preditora
- [x] Identificação automática das variáveis categóricas
- [x] Identificação automática das variáveis numéricas
- [x] Conversão de `agent` para texto antes da codificação
- [x] Aplicação de One-Hot Encoding às variáveis categóricas
- [x] Manutenção das variáveis numéricas em seu formato original
- [x] Uso de `handle_unknown='ignore'`
- [x] Ajuste do codificador somente com os dados de treinamento
- [x] Aplicação da mesma transformação aos dados de teste
      

### 🤖 **Modelagem de Machine Learning**

- [x] Definição do problema como classificação binária
- [x] Definição de `is_canceled` como variável-alvo
- [x] Separação entre variáveis preditoras (X) e alvo (y)
- [x] Separação dos dados em treinamento e teste
- [x] Utilização de 80% dos dados para treinamento
- [x] Utilização de 20% dos dados para teste
- [x] Utilização de `stratify=y`
- [x] Utilização de `random_state=42`
- [x] Identificação das variáveis categóricas
- [x] Identificação das variáveis numéricas
- [x] One-Hot Encoding das variáveis categóricas
- [x] Treinamento do modelo Random Forest Classifier
- [x] Configuração de `n_estimators=100`
- [x] Utilização de `random_state=42`
- [x] Utilização de processamento paralelo com `n_jobs=-1`
- [x] Treinamento exclusivamente com o conjunto de treinamento
- [x] Geração de previsões no conjunto de teste
- [x] Comparação das previsões com os valores reais


### 📊 **Avaliação do modelo**

- [x] Cálculo da Acurácia (Accuracy)
- [x] Cálculo da Matriz de Confusão
- [x] Visualização da Matriz de Confusão
- [x] Identificação de Verdadeiros Negativos (TN)
- [x] Identificação de Falsos Positivos (FP)
- [x] Identificação de Falsos Negativos (FN)
- [x] Identificação de Verdadeiros Positivos (TP)
- [x] Geração do Classification Report
- [x] Avaliação de Precision
- [x] Avaliação de Recall
- [x] Avaliação de F1-score
- [x] Avaliação separada das classes “Não cancelado” e “Cancelado”
- [x] Interpretação das métricas
- [x] Análise dos erros e acertos do modelo
- [x] Análise da capacidade do modelo de identificar cancelamentos
- [x] Identificação da maior dificuldade do modelo na classe de cancelamentos
      

### 📈 **Resultados obtidos**

- [x] Acurácia de 84,91%
- [x] Precision de 78% para a classe “Cancelado”
- [x] Recall de 63% para a classe “Cancelado”
- [x] F1-score de 70% para a classe “Cancelado”
- [x] Identificação de 3.026 cancelamentos corretamente
- [x] Identificação de 1.777 falsos negativos
- [x] Análise da diferença de desempenho entre as classes
- [x] Identificação de maior dificuldade na previsão de cancelamentos
- [x] Documentação das conclusões do modelo
 
