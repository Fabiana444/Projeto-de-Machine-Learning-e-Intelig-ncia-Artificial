Desafio Extra — Análise Exploratória de Dados e Modelo Preditivo de Cancelamento de Reservas de Hotel

Projeto: Desafio Extra — IA e Análise de Dados
Autor: Fabiana Nonjah
Linguagem: Python
Ambiente: Google Colab
Dataset: Hotel Booking Demand
Variável-alvo: is_canceled
Modelo preditivo: Random Forest Classifier

Fonte dos dados: Hotel Booking Demand — Kaggle
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand



1. Identificação e objetivo do projeto

Este projeto foi desenvolvido como parte do Desafio Extra da Carreira Tech — Ciclo 2, com o objetivo de aplicar conhecimentos introdutórios de Análise Exploratória de Dados (AED) e Inteligência Artificial, utilizando uma base de dados real de reservas hoteleiras.

O conjunto de dados utilizado é o Hotel Booking Demand, que contém informações sobre reservas de dois tipos de estabelecimento: City Hotel e Resort Hotel.

O objetivo principal foi importar, organizar, tratar e analisar os dados para identificar padrões relacionados às reservas e aos cancelamentos e, posteriormente, desenvolver um modelo preditivo introdutório capaz de classificar se uma reserva foi cancelada.

A variável is_canceled foi utilizada como variável-alvo do modelo:

0 = reserva não cancelada;
1 = reserva cancelada.

O projeto foi desenvolvido em Python, utilizando bibliotecas voltadas à manipulação, visualização e modelagem de dados.



2. Importação e compreensão inicial dos dados

O dataset foi carregado no ambiente Google Colab utilizando Pandas.

Na exploração inicial, o conjunto original apresentou:

119.390 registros;
32 colunas.

A primeira etapa consistiu em compreender a estrutura da base, os nomes das variáveis, seus tipos de dados, quantidade de registros e presença de valores ausentes e duplicados.

Também foi realizada uma análise inicial com df.info(), isnull() e outras verificações para identificar as características do conjunto de dados antes do tratamento.

Essa etapa foi importante para orientar as decisões posteriores de limpeza e preparação.



3. Preservação do dataset original

Antes de realizar os tratamentos, foi criada uma cópia independente do DataFrame original:

df_tratado = df.copy()

A decisão foi tomada para preservar o DataFrame original, evitando que as alterações realizadas durante a limpeza e preparação modificassem diretamente os dados carregados originalmente.

Dessa forma:

df representa o conjunto de dados original carregado;
df_tratado representa a cópia utilizada para limpeza, preparação, análise e modelagem.

Essa estratégia permite manter uma referência dos dados originais e realizar os tratamentos de maneira controlada.



4. Tratamento de registros duplicados

Foi realizada a verificação de registros duplicados utilizando:

df.duplicated().sum()

Foram identificados 31.994 registros duplicados, correspondentes a aproximadamente 26,80% do dataset original.

A investigação mostrou que esses registros eram duplicações exatas, ou seja, apresentavam os mesmos valores em todas as colunas.

Por esse motivo, os registros duplicados foram removidos antes das etapas de análise e modelagem:

df = df.drop_duplicates()

Após essa etapa, o dataset passou de 119.390 para 87.396 registros.

A remoção foi realizada porque duplicações exatas poderiam fazer determinados registros terem peso artificialmente maior nas análises e no treinamento do modelo.



5. Tratamento de valores ausentes

Após a remoção dos duplicados, foi realizada uma nova análise dos valores ausentes.

As principais colunas que exigiram tratamento foram:

children;
country;
agent;
company.

Cada variável recebeu um tratamento de acordo com sua natureza e quantidade de dados ausentes.



5.1. Tratamento da coluna children

A coluna children apresentava 4 valores ausentes, correspondentes a aproximadamente 0,0046% dos registros.

Como se trata de uma variável numérica e a quantidade de ausências era muito pequena, foi utilizada a mediana da própria coluna para preencher os valores ausentes.

A mediana foi escolhida por ser menos sensível a valores extremos do que a média.

Foi utilizado:

df_tratado['children'] = df_tratado['children'].fillna(
    df_tratado['children'].median()
)

Após o tratamento, foi realizada uma nova verificação para confirmar que não restavam valores ausentes nessa variável.



6. Tratamento da coluna country

A coluna country representa o país de origem do hóspede e apresentava 452 valores ausentes, aproximadamente 0,52% dos registros.

Como não seria correto atribuir arbitrariamente um país específico aos registros sem essa informação, os valores ausentes foram substituídos pela categoria:

Unknown

A decisão permite preservar os registros sem inventar uma informação que não estava disponível no dataset.

Foi utilizado:

df_tratado['country'] = df_tratado['country'].fillna('Unknown')

Após o tratamento, foi verificado que não restavam valores nulos nessa coluna.

Análise dos países

Como a variável country apresenta muitos países diferentes, a análise não foi realizada sobre todos eles de forma indiscriminada.

Primeiramente foram identificados os 15 países com maior número de reservas.

Essa escolha permite concentrar a comparação nos países com maior representatividade na base, evitando que países com poucas observações produzam taxas pouco representativas.

Os 15 países mais frequentes foram:

Portugal (PRT) — 27.440 reservas;
Reino Unido (GBR) — 10.432;
França (FRA) — 8.837;
Espanha (ESP) — 7.252;
Alemanha (DEU) — 5.387;
Itália (ITA) — 3.066;
Irlanda (IRL) — 3.016;
Bélgica (BEL) — 2.081;
Brasil (BRA) — 1.995;
Países Baixos (NLD) — 1.911;
Estados Unidos (USA) — 1.875;
Suíça (CHE) — 1.570;
China (CN) — 1.093;
Áustria (AUT) — 947;
Suécia (SWE) — 837.

A categoria Unknown foi mantida no dataset tratado, mas a comparação específica por país foi limitada aos 15 países com maior volume de reservas.



7. Tratamento da coluna agent

A coluna agent apresentou 12.193 valores ausentes, aproximadamente 13,95% dos registros.

A variável representa identificadores de agentes e, portanto, não seria adequado utilizar média ou mediana para preencher os valores ausentes.

Foi adotada a mesma estratégia utilizada para country: os valores ausentes foram substituídos por:

Unknown

Foi utilizado:

df_tratado['agent'] = df_tratado['agent'].fillna('Unknown')

Posteriormente, na preparação para o modelo, a coluna agent foi convertida para o tipo texto (string) para garantir que suas categorias fossem tratadas de forma consistente durante o One-Hot Encoding.

A verificação realizada confirmou que os registros da coluna estavam padronizados como texto.



8. Tratamento da coluna company

A coluna company apresentou 82.137 valores ausentes, aproximadamente 93,98% dos registros após a remoção dos duplicados.

Por apresentar uma quantidade extremamente elevada de valores ausentes, foi considerado inadequado criar uma categoria artificial para representar essa ausência.

Além disso, a variável não era essencial para o objetivo principal do projeto, que consiste na previsão de is_canceled.

Por isso, a coluna foi removida:

df_tratado = df_tratado.drop(columns=['company'])

Essa decisão reduziu uma variável com baixa disponibilidade de informação e evitou que sua grande quantidade de dados ausentes prejudicasse a preparação dos dados.



9. Adequação dos tipos de dados

Após os tratamentos de valores ausentes e duplicados, foram verificados os tipos das variáveis.

A coluna reservation_status_date, inicialmente armazenada como texto, foi convertida para o formato de data/hora do Pandas:

df_tratado['reservation_status_date'] = pd.to_datetime(
    df_tratado['reservation_status_date']
)

Após a conversão, o tipo passou a ser datetime64[ns].

Também foi verificado que a conversão não introduziu novos valores ausentes.

As demais variáveis foram mantidas de acordo com sua natureza numérica ou categórica.



10. Verificação e padronização dos nomes das colunas

Os nomes das colunas foram verificados para garantir consistência e adequação às etapas seguintes.

Após a análise, constatou-se que os nomes já estavam em um padrão consistente, utilizando nomenclatura adequada para o desenvolvimento da análise e da modelagem.

Por esse motivo, não foi necessária a renomeação das colunas.

Essa decisão preservou os nomes originais das variáveis do dataset, evitando alterações desnecessárias.



11. Identificação e tratamento de possíveis outliers

A identificação de possíveis valores extremos foi realizada utilizando estatísticas descritivas e o método do Intervalo Interquartil (IQR).

A análise identificou possíveis valores extremos em diversas variáveis numéricas, incluindo:

lead_time;
adr;
stays_in_weekend_nights;
stays_in_week_nights;
adults;
children;
babies;
previous_cancellations;
previous_bookings_not_canceled;
booking_changes;
days_in_waiting_list;
required_car_parking_spaces;
total_of_special_requests.

A identificação estatística de um possível outlier não significou sua remoção automática.

Cada situação considerada relevante foi investigada individualmente para verificar se o valor poderia representar um comportamento real ou uma inconsistência.



11.1. Valor negativo de adr

Foi identificado um registro com:

adr = -6,38

Como adr representa a tarifa diária média de uma reserva, uma tarifa negativa não possui significado nesse contexto.

O registro foi investigado e considerado inconsistente.

Por não haver informação suficiente para determinar qual seria a tarifa correta, optou-se pela remoção somente desse registro.

Após a remoção, o menor valor de adr passou a ser:

0,0


11.2. Valores extremos em adults

A análise da variável adults identificou registros com quantidades muito elevadas de adultos por reserva.

Foram encontrados 12 registros com mais de 10 adultos.

Esses registros apresentavam valores como 20, 26, 27, 40, 50 e 55 adultos.

Após a investigação, esses valores foram considerados inconsistentes para o contexto analisado e foram removidos.

O tratamento foi realizado somente para registros com:

adults > 10

Após o tratamento, a verificação confirmou que não restavam registros acima desse limite.



11.3. Valor extremo de adr = 5400

Durante a análise da tarifa diária foi identificado um registro com:

adr = 5400

Esse valor era muito superior aos demais valores observados.

A investigação mostrou que o registro correspondia a uma reserva de uma noite, para dois adultos, no City Hotel, com status de cancelamento.

Considerando a discrepância extrema e seu potencial de distorcer médias, visualizações e o modelo preditivo, esse único registro foi removido.

Após a remoção, o maior valor de adr passou a ser:

450,0

A decisão foi específica para esse registro extremo e não significou a remoção automática de todos os valores estatisticamente identificados como possíveis outliers.



12. Verificação final do tratamento

Após todas as etapas de limpeza e preparação, o DataFrame tratado apresentou:

87.382 registros;
32 colunas;
0 valores nulos.

Também foram novamente conferidos:

tipos das variáveis;
estrutura do DataFrame;
ausência de valores nulos;
adequação das variáveis para a modelagem.

O conjunto tratado foi então utilizado nas etapas de Análise Exploratória de Dados e modelagem.



13. Definição da variável-alvo

A variável:

is_canceled

foi definida como variável-alvo (TARGET), conforme solicitado pelo regulamento.

Sua distribuição no conjunto tratado foi:

Situação	Quantidade	Percentual
Não cancelada (0)	63.370	72,52%
Cancelada (1)	24.013	27,48%
Total	87.383*	

Observação: as verificações do notebook apresentam a distribuição do TARGET antes da exclusão posterior do registro extremo de adr = 5400; para a modelagem, o DataFrame final utilizado possui 87.382 registros.

A distribuição mostra que existe maior quantidade de reservas não canceladas, caracterizando uma diferença entre as classes que deve ser considerada na interpretação das métricas do modelo.



14. Análise Exploratória de Dados — AED

A Análise Exploratória buscou compreender o comportamento das reservas e investigar os principais fatores associados aos cancelamentos.

Foram utilizados filtros, agrupamentos (GroupBy), estatísticas descritivas e visualizações gráficas.

As principais análises foram:

distribuição por tipo de hotel;
cancelamento por tipo de hotel;
antecedência da reserva (lead_time);
faixas de antecedência;
sazonalidade mensal;
tendência anual;
tarifa diária (adr);
país de origem;
matriz de confusão na avaliação do modelo.


15. Distribuição das reservas por tipo de hotel

No DataFrame tratado foram identificadas:

53.428 reservas do City Hotel;
33.955 reservas do Resort Hotel.

Portanto, o City Hotel apresentou maior quantidade de reservas no conjunto analisado.



16. Cancelamentos por tipo de hotel

Foram observados:

City Hotel
37.379 reservas não canceladas;
16.049 reservas canceladas.

Taxa de cancelamento:

30,04%

Resort Hotel
25.991 reservas não canceladas;
7.964 reservas canceladas.

Taxa de cancelamento:

23,45%

Embora o City Hotel possua maior quantidade absoluta de cancelamentos, a comparação percentual é mais adequada porque os dois hotéis possuem volumes diferentes de reservas.

O resultado indica que, no conjunto analisado, o City Hotel apresentou maior taxa proporcional de cancelamento.



17. Antecedência da reserva e cancelamento

A variável lead_time representa a quantidade de dias entre a realização da reserva e a data de chegada.

A análise mostrou diferenças entre reservas canceladas e não canceladas.

Reservas não canceladas
média: 70,10 dias;
mediana: 38 dias.
Reservas canceladas
média: 105,60 dias;
mediana: 80 dias.

Assim, as reservas canceladas apresentaram maior antecedência média e mediana.

A diferença entre as médias foi de aproximadamente 35,50 dias.



18. Feature Engineering — faixas de lead_time

Para facilitar a interpretação da relação entre antecedência e cancelamento, foi criada a variável derivada:

lead_time_faixa

As reservas foram agrupadas nas seguintes categorias:

0-30 dias;
31-90 dias;
91-180 dias;
Mais de 180 dias.

A utilização de -1 como limite inferior do primeiro intervalo permitiu incluir corretamente o valor 0 de lead_time.



19. Insight — antecedência e cancelamento

A análise apresentou uma tendência crescente da taxa de cancelamento conforme aumentava a antecedência da reserva.

Faixa de antecedência	Taxa de cancelamento
0–30 dias	16,42%
31–90 dias	32,01%
91–180 dias	34,98%
Mais de 180 dias	39,68%

A taxa praticamente aumentou de forma contínua entre as faixas analisadas.

As reservas realizadas com mais de 180 dias de antecedência apresentaram taxa de cancelamento de 39,68%, enquanto aquelas realizadas com até 30 dias apresentaram 16,42%.

Esse resultado indica uma associação entre maior antecedência e maior ocorrência de cancelamentos no conjunto analisado. A relação observada não implica causalidade.



20. Sazonalidade por mês de chegada

Também foi analisada a taxa de cancelamento conforme o mês de chegada.

Os resultados foram:

Mês	Taxa de cancelamento
Janeiro	22,12%
Fevereiro	23,20%
Março	24,36%
Abril	30,46%
Maio	29,23%
Junho	30,32%
Julho	31,80%
Agosto	32,18%
Setembro	24,45%
Outubro	23,64%
Novembro	21,10%
Dezembro	26,86%

A maior taxa foi observada em agosto, com 32,18%, enquanto a menor ocorreu em novembro, com 21,10%.

As taxas foram mais elevadas principalmente entre abril e agosto, com redução a partir de setembro.

Esse comportamento indica uma possível variação sazonal nos cancelamentos no conjunto de dados analisado.



21. Tendência de cancelamento por ano

Para complementar a análise mensal, foi analisada a taxa de cancelamento por ano de chegada.

Ano	Taxa de cancelamento
2015	20,24%
2016	26,44%
2017	31,91%

O resultado mostra crescimento da taxa de cancelamento ao longo dos anos disponíveis.

A taxa passou de aproximadamente 20,24% em 2015 para 26,44% em 2016, chegando a 31,91% em 2017.

Portanto, existe uma tendência temporal de aumento da proporção de cancelamentos no período analisado.



22. Tarifa diária (adr) e cancelamento

A variável adr representa a tarifa diária média da reserva.

Após o tratamento do valor negativo e do registro extremo de 5400, foram comparadas as distribuições de adr entre reservas canceladas e não canceladas.

Reservas não canceladas
média: 102,00;
mediana: 94,50.
Reservas canceladas
média: 117,61;
mediana: 109,80.

Os resultados indicam que, neste conjunto de dados, as reservas canceladas apresentaram tarifas diárias médias e medianas superiores às reservas não canceladas.

A análise representa uma associação observada nos dados e não permite afirmar que tarifas mais elevadas sejam a causa dos cancelamentos.

Foi utilizada visualização em boxplot para comparar a distribuição das tarifas entre os dois grupos.



23. País de origem e cancelamento

A variável country foi analisada considerando os 15 países com maior volume de reservas.

A utilização dos 15 países mais representativos permite concentrar a análise em grupos com maior quantidade de observações e evitar interpretações baseadas em países com pouquíssimos registros.

Entre os países analisados, as maiores taxas de cancelamento foram:

Brasil: 36,44%;
Portugal: 35,63%;
Itália: 35,06%.

A menor taxa entre os 15 países analisados foi observada na:

Áustria: 17,95%.

Os resultados mostram diferenças nas taxas de cancelamento entre os principais países de origem representados no dataset.

Entretanto, essa análise é descritiva e não permite estabelecer causalidade entre país de origem e cancelamento.



24. Visualizações produzidas

Foram geradas visualizações para apoiar a interpretação dos resultados, incluindo:

distribuição de reservas por tipo de hotel;
cancelamentos por tipo de hotel;
taxa de cancelamento por faixa de lead_time;
taxa de cancelamento por mês de chegada;
taxa de cancelamento por ano de chegada;
distribuição da tarifa diária (adr) por status de cancelamento, utilizando boxplot;
taxa de cancelamento nos 15 países com maior número de reservas;
matriz de confusão do modelo preditivo.

As visualizações complementam as estatísticas e permitem identificar visualmente padrões e diferenças entre os grupos analisados.



25. Preparação para a modelagem preditiva

Após a conclusão da AED, foi preparada a base para o modelo de classificação.

A variável-alvo foi:

is_canceled

Foram excluídas da entrada do modelo:

is_canceled
reservation_status
reservation_status_date
lead_time_faixa

A exclusão de reservation_status e reservation_status_date foi realizada para evitar data leakage, pois essas variáveis contêm informações relacionadas ao resultado final da reserva.

A variável lead_time_faixa foi criada para a análise exploratória, mas não foi utilizada como entrada do modelo porque representa uma transformação de lead_time, que já é utilizado diretamente como variável preditora.



26. Separação entre treino e teste

Os dados foram divididos em:

80% para treinamento;
20% para teste.

O conjunto de treinamento possui:

69.905 registros

e o conjunto de teste possui:

17.477 registros.

Foi utilizada estratificação pela variável-alvo is_canceled, preservando a proporção entre reservas canceladas e não canceladas nos dois conjuntos.

Também foi utilizado:

random_state=42

para permitir a reprodução da divisão dos dados.



27. Codificação das variáveis categóricas

As variáveis categóricas foram transformadas utilizando One-Hot Encoding.

O codificador foi ajustado exclusivamente sobre os dados de treinamento:

fit_transform()

e posteriormente aplicado aos dados de teste:

transform()

Essa abordagem evita que informações do conjunto de teste influenciem a preparação dos dados.

Também foi utilizado:

handle_unknown='ignore'

para permitir que categorias existentes no teste, mas ausentes no treinamento, fossem processadas sem gerar erro.

As variáveis numéricas foram mantidas em seu formato original.

Após a codificação, os conjuntos passaram a possuir:

69.905 linhas e 577 variáveis no treinamento;
17.477 linhas e 577 variáveis no teste.


28. Modelo preditivo — Random Forest

Foi utilizado o algoritmo:

Random Forest Classifier

O modelo foi escolhido por se tratar de um problema de classificação binária, no qual o objetivo é prever:

0 = não cancelado
1 = cancelado

O modelo foi configurado com:

n_estimators=100
random_state=42
n_jobs=-1

O treinamento foi realizado exclusivamente sobre:

X_treino_codificado
y_treino

O conjunto de teste permaneceu separado para a avaliação posterior.



29. Avaliação do modelo

Após o treinamento, o Random Forest foi utilizado para gerar previsões para os 17.477 registros do conjunto de teste.

A avaliação utilizou as métricas solicitadas pela atividade:

acurácia;
matriz de confusão;
precisão (Precision);
Recall;
F1-score.


30. Acurácia

O modelo apresentou:

Acurácia: 84,91%

Isso significa que aproximadamente 84,91% das previsões realizadas para o conjunto de teste estavam corretas.

Entretanto, a acurácia não deve ser analisada isoladamente, pois a variável-alvo apresenta distribuição desigual entre as classes:

72,52% não canceladas;
27,48% canceladas.

Por isso, foram analisadas também as demais métricas.



31. Matriz de confusão

A matriz de confusão apresentou:

Resultado	Quantidade
Verdadeiros Negativos (TN)	11.813
Falsos Positivos (FP)	861
Falsos Negativos (FN)	1.777
Verdadeiros Positivos (TP)	3.026

Dos 4.803 cancelamentos reais presentes no conjunto de teste:

3.026 foram identificados corretamente;
1.777 não foram identificados pelo modelo.

Esse resultado evidencia que o modelo possui maior dificuldade para identificar todos os cancelamentos reais.



32. Precision, Recall e F1-score
Classe 0 — Não cancelado
Precision: 0,87;
Recall: 0,93;
F1-score: 0,90.

O modelo apresentou bom desempenho para identificar reservas que não foram canceladas.

Classe 1 — Cancelado
Precision: 0,78;
Recall: 0,63;
F1-score: 0,70.

A Precision de 0,78 indica que, entre as reservas classificadas pelo modelo como canceladas, 78% realmente pertenciam à classe de cancelamento.

O Recall de 0,63 mostra que o modelo identificou corretamente 63% dos cancelamentos reais.

Esse é o principal ponto de atenção no desempenho do modelo.



33. Análise final do modelo preditivo

O Random Forest apresentou 84,91% de acurácia, demonstrando um desempenho geral satisfatório no conjunto de teste.

Para a classe de reservas não canceladas, o modelo apresentou desempenho superior, com:

Precision de 87%;
Recall de 93%;
F1-score de 90%.

Para a classe de reservas canceladas, apresentou:

Precision de 78%;
Recall de 63%;
F1-score de 70%.

Portanto, embora o modelo apresente bom desempenho geral, existe uma dificuldade maior na identificação dos cancelamentos reais.

Dos 4.803 cancelamentos existentes no conjunto de teste, 3.026 foram identificados corretamente e 1.777 não foram identificados.

Esse resultado é especialmente relevante porque demonstra por que a análise de Precision, Recall e F1-score é necessária além da acurácia.

O modelo atende ao objetivo de desenvolver uma modelagem preditiva introdutória, conforme proposto pelo desafio, sem que seja necessário afirmar que ele representa uma solução otimizada ou definitiva para previsão de cancelamentos.



34. Principais insights obtidos

A Análise Exploratória permitiu identificar os seguintes padrões principais:


1. Tipo de hotel

O City Hotel apresentou maior volume de reservas e também maior taxa proporcional de cancelamento:

City Hotel: 30,04%;
Resort Hotel: 23,45%.

2. Antecedência da reserva

Foi identificada uma relação crescente entre lead_time e cancelamento.

A taxa passou de:

16,42% para reservas de até 30 dias;

para:

39,68% para reservas com mais de 180 dias.


3. Sazonalidade

A taxa de cancelamento variou ao longo dos meses.

Maior: agosto — 32,18%;
Menor: novembro — 21,10%.

4. Tendência ao longo dos anos

A taxa de cancelamento aumentou:

2015: 20,24%;
2016: 26,44%;
2017: 31,91%.

5. Tarifa diária

As reservas canceladas apresentaram adr médio e mediano superiores às não canceladas:

Não canceladas: média 102,00;
Canceladas: média 117,61.

6. País de origem

Entre os 15 países com maior volume de reservas:

Brasil apresentou a maior taxa: 36,44%;
Portugal: 35,63%;
Itália: 35,06%;
Áustria apresentou a menor: 17,95%.

Esses resultados são associações observadas no conjunto de dados e não representam relações de causalidade.



35. Conclusão

O projeto contemplou as etapas previstas no desafio: importação e compreensão dos dados, tratamento e preparação, Análise Exploratória de Dados, Feature Engineering, separação entre treino e teste, codificação das variáveis categóricas, treinamento de modelo de classificação e avaliação do desempenho.

Durante o tratamento, foram removidos registros duplicados exatos, tratados valores ausentes, convertidos tipos de dados, removidas variáveis com quantidade excessiva de ausências e investigados valores extremos.

A criação do df_tratado permitiu preservar o DataFrame original e realizar as transformações de maneira controlada.

A AED permitiu identificar padrões relacionados ao tipo de hotel, antecedência da reserva, sazonalidade, evolução anual, tarifa diária e país de origem.

Na etapa de Inteligência Artificial, o Random Forest alcançou 84,91% de acurácia, com desempenho superior para a classe de reservas não canceladas e maior dificuldade para identificar cancelamentos.

Dessa forma, o projeto demonstra a aplicação integrada de técnicas introdutórias de análise de dados e aprendizado de máquina sobre uma base de dados real, atendendo aos objetivos propostos para o desafio.



36. Arquivos que compõem a entrega

Conforme o regulamento, o arquivo compactado final deverá conter:

Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.ipynb
hotel_bookings.csv
documentação do projeto (.md, .txt, .doc, .docx ou .pdf)
visualizações geradas

O conjunto deverá ser entregue em um único arquivo .zip ou .rar, com tamanho máximo de 20 MB.

As visualizações produzidas no notebook podem ser exportadas para arquivos .png e incluídas no pacote final para tornar explícita a presença dos gráficos solicitados pela banca.


Conferência contra a banca

Esta documentação cobre diretamente os itens obrigatórios:

Exigência da banca	Onde está na documentação
Identificação do projeto e autor	1. Identificação e objetivo
Descrição das etapas de desenvolvimento	2 a 33
Decisões no tratamento dos dados	3 a 12
Valores nulos	5 a 8
Duplicados	4
Tipos de dados	9
Padronização das colunas	10
Outliers	11
AED / GroupBy	14 a 23
Insights	34
Visualizações	24
Feature Engineering	18
Treino/teste	26
Modelo de classificação	28
is_canceled como TARGET	13 e 25
Acurácia	30
Matriz de confusão	31
Precision, Recall e F1	32
Breve análise do modelo	33
Entrega dos arquivos	36