# Projeto Big Data
## Análise do Impacto de Eventos Mundiais no Mercado Financeiro Global
### Utilizando Apache Spark e Yahoo Finance API

**Pós-Graduação em Big Data**

**Felipe Martins**
**Ricardo Areas**
**Pedro Silva**
**Ana Luiza Pazze**
**Anny Caroline Sousa**
**Fabio Silva**

**Data: Janeiro 2025**

---

## Índice

1. [Resumo Executivo](#resumo-executivo)
2. [Objetivos](#objetivos)
3. [Escopo do Projeto](#escopo-do-projeto)
4. [Metodologia](#metodologia)
5. [Divisão de Responsabilidades da Equipe](#divisão-de-responsabilidades-da-equipe)
6. [Tecnologias e Ferramentas](#tecnologias-e-ferramentas)
7. [Métricas de Sucesso](#métricas-de-sucesso)
8. [Entregáveis Finais](#entregáveis-finais)
9. [Conclusão](#conclusão)
10. [Referências](#referências)

---

## Resumo Executivo

Este projeto tem como objetivo analisar o impacto de grandes eventos mundiais nos mercados financeiros globais, utilizando tecnologias de Big Data, especificamente Apache Spark, para processar e analisar dados históricos de ações obtidos através da Yahoo Finance API.

O estudo focará em eventos significativos como eleições presidenciais americanas, conflitos geopolíticos (Guerra Rússia-Ucrânia), e grandes catástrofes naturais, investigando como esses eventos afetam os preços de ações, índices de bolsas de valores e a volatilidade dos mercados em escala global.

---

## Objetivos

### Objetivo Geral
Desenvolver uma solução de Big Data capaz de identificar e quantificar o impacto de eventos mundiais significativos nos mercados financeiros globais, utilizando Apache Spark para processamento distribuído de grandes volumes de dados financeiros.

### Objetivos Específicos
- Implementar um pipeline de dados para coleta automatizada de informações financeiras via Yahoo Finance API
- Criar um sistema de processamento distribuído usando Apache Spark para análise de grandes volumes de dados históricos
- Desenvolver algoritmos de correlação temporal entre eventos mundiais e movimentações do mercado financeiro
- Construir visualizações interativas para apresentação dos resultados da análise
- Implementar modelos preditivos para estimar o impacto de futuros eventos similares
- Documentar metodologias e criar relatórios técnicos dos achados

---

## Escopo do Projeto

### Eventos Mundiais a Serem Analisados

#### 1. Eventos Políticos
- Eleições presidenciais americanas (2016, 2020, 2024)
- Brexit e suas fases
- Mudanças de governo em economias emergentes

#### 2. Conflitos Geopolíticos
- Guerra Rússia-Ucrânia (início em 2022)
- Tensões comerciais EUA-China
- Conflitos no Oriente Médio

#### 3. Catástrofes e Pandemias
- COVID-19 (2020-2023)
- Grandes terremotos (Japão 2011, Turquia 2023)
- Furacões e desastres climáticos extremos

#### 4. Eventos Econômicos
- Decisões de política monetária do Federal Reserve
- Crises bancárias (SVB, Credit Suisse 2023)
- Anúncios de grandes fusões e aquisições

### Mercados e Ativos Financeiros
- **Índices Principais**: S&P 500, NASDAQ, Dow Jones, FTSE 100, Nikkei 225, DAX, Bovespa
- **Setores Específicos**: Tecnologia, Energia, Saúde, Financeiro, Commodities
- **Moedas**: USD, EUR, JPY, GBP, BRL, RUB
- **Commodities**: Ouro, Petróleo, Gás Natural, Trigo

---

## Metodologia

### Arquitetura Técnica

#### 1. Coleta de Dados
- Yahoo Finance API para dados financeiros históricos
- APIs de notícias para identificação temporal de eventos
- Web scraping para dados complementares

#### 2. Processamento de Dados
- Apache Spark para processamento distribuído
- PySpark para desenvolvimento em Python
- Spark SQL para consultas complexas
- Spark MLlib para machine learning

#### 3. Armazenamento
- HDFS para armazenamento distribuído
- Parquet para formato otimizado de dados
- Delta Lake para versionamento de dados

#### 4. Análise e Visualização
- Jupyter Notebooks para análise exploratória
- Matplotlib/Plotly para visualizações
- Tableau/Power BI para dashboards interativos

### Metodologia de Análise

#### 1. Análise Temporal
- Identificação de janelas temporais de impacto (pré, durante, pós-evento)
- Análise de volatilidade antes e depois dos eventos
- Correlação temporal entre eventos e movimentações de preços

#### 2. Machine Learning
- Algoritmos de classificação para identificar padrões
- Modelos de previsão de volatilidade
- Análise de sentimento de notícias

---

## Divisão de Responsabilidades da Equipe

### Ana Luiza Pazze - Arquitetura e Infraestrutura

**Responsabilidades:**
- Configuração do ambiente Apache Spark
- Implementação da arquitetura de dados distribuída
- Configuração do HDFS e sistemas de armazenamento
- Otimização de performance do cluster Spark
- Implementação de pipelines ETL

**Entregáveis:**
- Documentação da arquitetura técnica
- Scripts de configuração do ambiente
- Pipeline de ETL funcional
- Relatório de performance e otimizações

### Felipe Martins - API e Coleta de Dados

**Responsabilidades:**
- Implementação da integração com Yahoo Finance API
- Desenvolvimento de coletores de dados de notícias
- Implementação de web scraping para dados complementares
- Tratamento de rate limits e otimização de requisições
- Validação e limpeza de dados coletados

**Entregáveis:**
- Módulos de coleta de dados funcionais
- Documentação das APIs utilizadas
- Scripts de validação de dados
- Relatório de qualidade dos dados coletados

### Pedro Silva - Análise de Dados

**Responsabilidades:**
- Análise exploratória dos dados financeiros
- Implementação de análises estatísticas
- Desenvolvimento de métricas de impacto
- Análise de correlações e causalidade
- Validação estatística dos resultados

**Entregáveis:**
- Relatórios de análise exploratória
- Implementação de testes estatísticos
- Métricas de impacto definidas e calculadas
- Relatório de correlações identificadas

### Anny Caroline Sousa - Machine Learning

**Responsabilidades:**
- Desenvolvimento de modelos preditivos
- Implementação de algoritmos de classificação
- Análise de sentimento de notícias
- Otimização de hiperparâmetros
- Validação e avaliação de modelos

**Entregáveis:**
- Modelos de machine learning treinados
- Relatório de performance dos modelos
- Sistema de análise de sentimento
- Documentação dos algoritmos implementados

### Ricardo Areas - Visualização e Dashboard

**Responsabilidades:**
- Desenvolvimento de visualizações interativas
- Criação de dashboards executivos
- Implementação de relatórios automatizados
- Design de interface de usuário
- Otimização de performance das visualizações

**Entregáveis:**
- Dashboard interativo funcional
- Biblioteca de visualizações reutilizáveis
- Relatórios automatizados
- Documentação de uso das visualizações

### Fabio Silva - Gestão de Projeto

**Responsabilidades:**
- Coordenação geral do projeto
- Gestão de cronograma e entregas
- Documentação técnica e acadêmica
- Preparação de apresentações
- Controle de qualidade e integração

**Entregáveis:**
- Documentação completa do projeto
- Relatório final acadêmico
- Apresentações executivas
- Plano de projeto e cronograma

---

## Tecnologias e Ferramentas

### Tecnologias Principais
- **Apache Spark 3.4+**: Processamento distribuído de dados
- **PySpark**: Interface Python para Spark
- **Python 3.9+**: Linguagem principal de desenvolvimento
- **Yahoo Finance API**: Fonte de dados financeiros
- **HDFS**: Sistema de arquivos distribuído

### Bibliotecas Python
- **yfinance**: Interface para Yahoo Finance API
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Machine learning
- **matplotlib/plotly**: Visualizações
- **requests**: Requisições HTTP
- **beautifulsoup4**: Web scraping

### Infraestrutura
- **Docker**: Containerização
- **Jupyter Notebooks**: Desenvolvimento interativo
- **Git**: Controle de versão
- **Apache Airflow**: Orquestração de pipelines

---

## Métricas de Sucesso

### Métricas Técnicas
- Processamento de pelo menos 1TB de dados financeiros históricos
- Latência de processamento inferior a 30 minutos para análises completas
- Precisão dos modelos preditivos superior a 75%
- Disponibilidade do sistema superior a 99%

### Métricas Acadêmicas
- Identificação de pelo menos 10 correlações significativas
- Documentação completa de metodologias utilizadas
- Apresentação de resultados com visualizações claras
- Relatório final com padrões acadêmicos

---

## Entregáveis Finais

### Entregáveis Técnicos
1. Sistema completo de análise funcionando
2. Código fonte documentado no GitHub
3. Dashboard interativo para visualização
4. Modelos de machine learning treinados
5. Documentação técnica completa

### Entregáveis Acadêmicos
1. Relatório final de 50+ páginas
2. Apresentação executiva (30 minutos)
3. Artigo científico para submissão
4. Dataset processado para pesquisas futuras
5. Metodologia replicável documentada

---

## Conclusão

Este projeto representa uma oportunidade única de aplicar tecnologias de Big Data em um problema real e relevante do mercado financeiro. A combinação de Apache Spark para processamento distribuído e Yahoo Finance API para dados de qualidade permitirá análises profundas sobre o impacto de eventos mundiais nos mercados.

A divisão clara de responsabilidades entre os 6 membros da equipe garante que cada aspecto do projeto seja desenvolvido por especialistas, enquanto a metodologia estruturada assegura entregas pontuais e qualidade técnica.

Os resultados esperados não apenas contribuirão para o conhecimento acadêmico na área de Big Data e finanças, mas também fornecerão insights valiosos sobre como eventos geopolíticos afetam os mercados globais, informação crucial para investidores, analistas e formuladores de políticas.

---

## Referências

1. Apache Spark Documentation. *Apache Software Foundation*. Disponível em: https://spark.apache.org/docs/
2. Yahoo Finance API Documentation. *Yahoo Inc*. Disponível em: https://finance.yahoo.com/
3. Zaharia, M. et al. (2016). *Apache Spark: A Unified Engine for Big Data Processing*. Communications of the ACM.
4. Chen, C. & Zhang, J. (2019). *Big Data Analytics in Financial Markets*. Journal of Financial Data Science.
5. Kumar, S. et al. (2020). *Event-Driven Market Analysis Using Machine Learning*. IEEE Transactions on Big Data.
