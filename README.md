# Projeto-de-Machine-Learning-e-Intelig-ncia-Artificial
Projeto de Machine Learning para previsão de cancelamentos de reservas hoteleiras, utilizando análise exploratória de dados, tratamento e Engenharia de Features, classificação e métricas de avaliação.

**Etapas desenvolvidas no projeto**

Tratamento e preparação dos dados
 Importação da biblioteca Pandas
 Carregamento do dataset Hotel Booking Demand
 Visualização inicial dos dados com df.head()
 Verificação das dimensões do dataset
 Identificação das colunas disponíveis
 Verificação da estrutura e dos tipos de dados com df.info()
 Identificação e quantificação de valores ausentes
 Cálculo do percentual de valores ausentes por coluna
 Verificação de registros duplicados
 Cálculo do percentual de registros duplicados
 Investigação dos registros duplicados
 Remoção dos registros duplicados
 Verificação do dataset após a remoção dos duplicados
 Criação de uma cópia do DataFrame original
 Criação do DataFrame df_tratado
 Preservação do DataFrame original para evitar alterações no dataset carregado
 Tratamento dos valores ausentes da coluna children utilizando a mediana
 Análise da distribuição da coluna country
 Tratamento dos valores ausentes de country com Unknown
 Análise da coluna agent
 Tratamento dos valores ausentes de agent com Unknown
 Análise da coluna company
 Remoção da coluna company devido à elevada quantidade de valores ausentes
 Verificação dos tipos de dados após os tratamentos
 Conversão de reservation_status_date de texto para datetime
 Verificação da conversão da variável temporal
 Verificação de valores ausentes após a conversão da data
 Identificação de possíveis outliers utilizando o método IQR
 Análise estatística das principais variáveis numéricas
 Investigação de valores negativos na variável adr
 Remoção do registro inconsistente com adr negativo
 Investigação de valores extremos na variável adults
 Identificação de reservas com mais de 10 adultos
 Remoção dos registros considerados inconsistentes na variável adults
 Investigação do valor extremo adr = 5400
 Remoção do registro com adr = 5400
 Verificação final dos valores ausentes
 Verificação final dos tipos de dados
 Verificação da dimensão final do DataFrame tratado
 

🔎 **Análise Exploratória de Dados (EDA)**

 Definição da variável-alvo is_canceled
 Verificação dos valores existentes no TARGET
 Análise da distribuição do TARGET
 Cálculo da distribuição percentual do TARGET
 Análise da distribuição das reservas por tipo de hotel
 Comparação de cancelamentos e não cancelamentos por tipo de hotel
 Cálculo da taxa de cancelamento por tipo de hotel
 Análise da relação entre lead_time e is_canceled
 Cálculo de estatísticas de lead_time por situação de cancelamento
 Criação de faixas de antecedência da reserva
 Análise da taxa de cancelamento por faixa de antecedência
 Criação de gráfico da taxa de cancelamento por faixa de antecedência
 Análise da sazonalidade dos cancelamentos
 Organização dos meses em ordem cronológica
 Cálculo da taxa de cancelamento por mês
 Criação de gráfico da taxa de cancelamento por mês
 Análise da evolução das reservas/cancelamentos por ano
 Cálculo da taxa de cancelamento por ano
 Criação de gráfico da taxa de cancelamento por ano
 Análise da relação entre adr e cancelamento
 Comparação das estatísticas de adr entre reservas canceladas e não canceladas
 Investigação do valor extremo adr = 5400
 Criação de boxplot de adr por status de cancelamento
 Análise do país de origem dos hóspedes
 Identificação dos 15 países com maior número de reservas
 Cálculo da taxa de cancelamento para os 15 países com mais reservas
 Criação de gráfico da taxa de cancelamento por país de origem
 Identificação de padrões e tendências
 Registro dos principais insights da análise exploratória


 🛠️ **Engenharia de Features**

 Criação da variável derivada lead_time_faixa
 Agrupamento de lead_time em faixas de antecedência
 Utilização da variável derivada na análise exploratória
 Definição de is_canceled como variável-alvo (TARGET)
 Definição das variáveis preditoras (X)
 Exclusão de is_canceled das variáveis de entrada
 Exclusão de reservation_status para evitar vazamento de dados (data leakage)
 Exclusão de reservation_status_date para evitar utilização de informação posterior ao momento da reserva
 Exclusão de lead_time_faixa das entradas do modelo, pois lead_time foi utilizado diretamente como variável preditora
 Identificação automática das variáveis categóricas
 Identificação automática das variáveis numéricas
 Conversão de agent para texto antes da codificação
 Aplicação de One-Hot Encoding às variáveis categóricas
 Manutenção das variáveis numéricas em seu formato original
 Uso de handle_unknown='ignore'
 Ajuste do codificador somente com os dados de treinamento
 Aplicação da mesma transformação aos dados de teste


 🤖 **Modelagem de Machine Learning**
 
 Definição do problema como classificação binária
 Definição de is_canceled como variável-alvo
 Separação entre variáveis preditoras (X) e alvo (y)
 Separação dos dados em treinamento e teste
 Utilização de 80% dos dados para treinamento
 Utilização de 20% dos dados para teste
 Utilização de stratify=y
 Utilização de random_state=42
 Identificação das variáveis categóricas
 Identificação das variáveis numéricas
 One-Hot Encoding das variáveis categóricas
 Treinamento do modelo Random Forest Classifier
 Configuração de n_estimators=100
 Utilização de random_state=42
 Utilização de processamento paralelo com n_jobs=-1
 Treinamento exclusivamente com o conjunto de treinamento
 Geração de previsões no conjunto de teste
 Comparação das previsões com os valores reais


 📊 **Avaliação do modelo**
 
 Cálculo da Acurácia (Accuracy)
 Cálculo da Matriz de Confusão
 Visualização da Matriz de Confusão
 Identificação de Verdadeiros Negativos (TN)
 Identificação de Falsos Positivos (FP)
 Identificação de Falsos Negativos (FN)
 Identificação de Verdadeiros Positivos (TP)
 Geração do Classification Report
 Avaliação de Precision
 Avaliação de Recall
 Avaliação de F1-score
 Avaliação separada das classes “Não cancelado” e “Cancelado”
 Interpretação das métricas
 Análise dos erros e acertos do modelo
 Análise da capacidade do modelo de identificar cancelamentos
 Identificação da maior dificuldade do modelo na classe de cancelamentos


📈 **Resultados obtidos**

 Acurácia de 84,91%
 Precision de 78% para a classe “Cancelado”
 Recall de 63% para a classe “Cancelado”
 F1-score de 70% para a classe “Cancelado”
 Identificação de 3.026 cancelamentos corretamente
 Identificação de 1.777 falsos negativos
 Análise da diferença de desempenho entre as classes
 Identificação de maior dificuldade na previsão de cancelamentos
 Documentação das conclusões do modelo 
 
