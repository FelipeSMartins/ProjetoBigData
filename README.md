# Projeto Big Data - Análise de Mercados Financeiros

## 📊 Visão Geral

Este projeto implementa uma plataforma completa de análise de Big Data para mercados financeiros, integrando coleta de dados em tempo real, processamento distribuído, análise de sentimentos e machine learning para identificar correlações entre eventos mundiais e movimentos de mercado.

### 🎯 Objetivos

- **Coleta Automatizada**: Dados financeiros (Yahoo Finance) e eventos mundiais (APIs de notícias)
- **Processamento Distribuído**: Apache Spark e HDFS para análise de grandes volumes
- **Análise Avançada**: Correlações entre eventos e movimentos de mercado
- **Machine Learning**: Modelos preditivos e análise de sentimentos
- **Visualização**: Dashboards interativos e relatórios automatizados

## 👥 Equipe de Desenvolvimento

| Responsável | Área de Atuação | Módulos |
|-------------|-----------------|---------|
| **Fabio** | Gestão do Projeto | Coordenação geral, planejamento |
| **Felipe Martins** | APIs e Coleta de Dados | Yahoo Finance, News API, endpoints |
| **Ana Luiza Pazze** | Arquitetura e Infraestrutura | Spark, HDFS, Docker, configurações |
| **Pedro Silva** | Análise Estatística | Análise exploratória, estatísticas |
| **Anny Caroline Sousa** | Machine Learning | Modelos preditivos, sentiment analysis |
| **Ricardo Areas** | Visualização | Dashboards, gráficos interativos |

## 🏗️ Arquitetura do Sistema

```
ProjetoBigData/
├── src/                          # Código fonte principal
│   ├── data_collection/          # Coleta de dados
│   │   ├── yahoo_finance_collector.py
│   │   └── news_collector.py
│   ├── infrastructure/           # Infraestrutura distribuída
│   │   ├── spark_manager.py
│   │   └── hdfs_manager.py
│   ├── data_analysis/           # Análise estatística
│   │   ├── statistical_analyzer.py
│   │   └── exploratory_analyzer.py
│   ├── machine_learning/        # ML e Sentiment Analysis
│   │   ├── sentiment_analyzer.py
│   │   └── predictive_models.py
│   └── visualization/           # Dashboards e gráficos
│       ├── dashboard.py
│       └── charts.py
├── data/                        # Dados do projeto
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   └── external/                # Dados externos
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Testes unitários
├── docker/                      # Configurações Docker
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── spark/
├── config/                      # Configurações
└── docs/                        # Documentação
```

## 🚀 Tecnologias Utilizadas

### Core Technologies
- **Python 3.9+**: Linguagem principal
- **Apache Spark 3.4.0**: Processamento distribuído
- **Hadoop HDFS**: Armazenamento distribuído
- **Docker & Docker Compose**: Containerização

### Data & Analytics
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Machine learning
- **scipy**: Análise estatística
- **Delta Lake**: Versionamento de dados

### APIs & Data Sources
- **yfinance**: Yahoo Finance API
- **newsapi-python**: News API
- **requests**: HTTP requests
- **beautifulsoup4**: Web scraping

### Visualization
- **streamlit**: Dashboards web
- **plotly**: Gráficos interativos
- **matplotlib**: Visualizações estáticas
- **seaborn**: Visualizações estatísticas

### NLP & Sentiment Analysis
- **textblob**: Análise de sentimentos
- **vaderSentiment**: Sentiment analysis
- **nltk**: Processamento de linguagem natural

### Infrastructure & Monitoring
- **jupyter**: Notebooks interativos
- **pytest**: Testes unitários
- **python-dotenv**: Gerenciamento de configurações

## 📋 Pré-requisitos

### Sistema
- **Docker**: 28.4.0+
- **Docker Compose**: 2.0+
- **Python**: 3.9+
- **Git**: Para controle de versão

### APIs (Opcionais)
- **News API Key**: Para coleta de notícias
- **Alpha Vantage API**: Dados financeiros alternativos

## 🛠️ Instalação e Configuração

### 1. Clone o Repositório
```bash
git clone <repository-url>
cd ProjetoBigData
```

### 2. Configuração de Ambiente

#### Opção A: Docker (Recomendado)
```bash
# Construir e iniciar todos os serviços
docker-compose up -d

# Verificar status dos containers
docker-compose ps

# Acessar logs
docker-compose logs -f
```

#### Opção B: Instalação Local
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Configuração de APIs
```bash
# Copiar arquivo de configuração
cp config/.env.example config/.env

# Editar com suas chaves de API
nano config/.env
```

Exemplo de `.env`:
```env
# APIs
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Spark Configuration
SPARK_MASTER_URL=spark://spark-master:7077
HDFS_NAMENODE_URL=hdfs://namenode:9000

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bigdata_finance
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
```

## 🎮 Como Usar

### 1. Acesso aos Serviços

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **Jupyter Notebook** | http://localhost:8888 | Notebooks interativos |
| **Spark Master UI** | http://localhost:8080 | Interface do Spark |
| **HDFS NameNode** | http://localhost:9870 | Interface do HDFS |
| **Streamlit Dashboard** | http://localhost:8501 | Dashboard principal |

### 2. Coleta de Dados

#### Yahoo Finance
```python
from src.data_collection.yahoo_finance_collector import YahooFinanceCollector

# Inicializar coletor
collector = YahooFinanceCollector()

# Coletar dados de ações
stock_data = collector.collect_stock_data(['AAPL', 'GOOGL', 'MSFT'])

# Coletar dados de índices
index_data = collector.collect_index_data(['^GSPC', '^DJI', '^IXIC'])
```

#### Notícias e Eventos
```python
from src.data_collection.news_collector import NewsCollector

# Inicializar coletor de notícias
news_collector = NewsCollector()

# Coletar eventos históricos
events = news_collector.collect_historical_events(
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### 3. Processamento com Spark

```python
from src.infrastructure.spark_manager import SparkManager

# Inicializar Spark
spark_manager = SparkManager()

# Processar dados financeiros
processed_data = spark_manager.process_financial_data(
    input_path="hdfs://namenode:9000/raw/financial_data",
    output_path="hdfs://namenode:9000/processed/financial_data"
)

# Executar pipeline ETL completo
spark_manager.run_etl_pipeline()
```

### 4. Análise Estatística

```python
from src.data_analysis.statistical_analyzer import StatisticalAnalyzer

# Inicializar analisador
analyzer = StatisticalAnalyzer()

# Análise descritiva
stats = analyzer.descriptive_statistics(data)

# Teste de normalidade
normality_results = analyzer.test_normality(data['returns'])

# Análise de correlação
correlation_matrix = analyzer.correlation_analysis(data)
```

### 5. Machine Learning

```python
from src.machine_learning.predictive_models import PredictiveModels
from src.machine_learning.sentiment_analyzer import SentimentAnalyzer

# Modelos preditivos
ml_models = PredictiveModels()
model_results = ml_models.train_regression_models(data)

# Análise de sentimentos
sentiment_analyzer = SentimentAnalyzer()
sentiment_scores = sentiment_analyzer.analyze_sentiment_batch(news_texts)
```

### 6. Visualização

```python
from src.visualization.dashboard import FinancialDashboard
from src.visualization.charts import FinancialCharts

# Dashboard interativo
dashboard = FinancialDashboard()
dashboard.create_streamlit_dashboard()

# Gráficos especializados
charts = FinancialCharts()
price_chart = charts.plot_price_action(data)
technical_chart = charts.plot_technical_indicators(data)
```

## 📊 Exemplos de Uso

### Análise de Impacto de Eventos
```python
# 1. Coletar dados financeiros
financial_data = collector.collect_stock_data(['SPY'], 
                                            start_date='2020-01-01',
                                            end_date='2023-12-31')

# 2. Coletar eventos
events = news_collector.collect_historical_events(
    keywords=['COVID', 'Federal Reserve', 'inflation'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 3. Análise de correlação
correlation_results = analyzer.analyze_event_impact(
    financial_data, events, window_days=30
)

# 4. Visualização
impact_chart = charts.plot_event_impact(
    financial_data, events, title="Impacto de Eventos no S&P 500"
)
```

### Pipeline Completo de ML
```python
# 1. Preparação de features
feature_engineer = FeatureEngineer()
features = feature_engineer.create_technical_indicators(data)
features = feature_engineer.create_lag_features(features)

# 2. Treinamento de modelos
models = PredictiveModels()
results = models.train_all_models(features, target='price_direction')

# 3. Avaliação
evaluation = models.evaluate_models(results)

# 4. Predições
predictions = models.predict(new_data)
```

## 🧪 Testes

### Executar Testes
```bash
# Todos os testes
pytest tests/

# Testes específicos
pytest tests/test_data_collection.py
pytest tests/test_spark_manager.py

# Com cobertura
pytest --cov=src tests/
```

### Estrutura de Testes
```
tests/
├── test_data_collection/
│   ├── test_yahoo_finance_collector.py
│   └── test_news_collector.py
├── test_infrastructure/
│   ├── test_spark_manager.py
│   └── test_hdfs_manager.py
├── test_analysis/
│   ├── test_statistical_analyzer.py
│   └── test_exploratory_analyzer.py
└── test_ml/
    ├── test_sentiment_analyzer.py
    └── test_predictive_models.py
```

## 📈 Monitoramento e Performance

### Métricas do Sistema
- **Spark Jobs**: Monitoramento via Spark UI
- **HDFS Usage**: Utilização de armazenamento
- **Memory Usage**: Consumo de memória
- **Processing Time**: Tempo de processamento

### Logs
```bash
# Logs do Docker Compose
docker-compose logs -f

# Logs específicos
docker-compose logs spark-master
docker-compose logs namenode
docker-compose logs jupyter
```

## 🔧 Configurações Avançadas

### Spark Configuration
```python
# spark-defaults.conf
spark.sql.adaptive.enabled=true
spark.sql.adaptive.coalescePartitions.enabled=true
spark.serializer=org.apache.spark.serializer.KryoSerializer
spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension
spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
```

### HDFS Configuration
```xml
<!-- core-site.xml -->
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000</value>
    </property>
</configuration>
```

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. Container não inicia
```bash
# Verificar logs
docker-compose logs container_name

# Recriar containers
docker-compose down
docker-compose up -d --force-recreate
```

#### 2. Erro de memória no Spark
```bash
# Ajustar configurações no docker-compose.yml
environment:
  - SPARK_WORKER_MEMORY=4g
  - SPARK_DRIVER_MEMORY=2g
```

#### 3. HDFS não acessível
```bash
# Verificar status do namenode
docker-compose exec namenode hdfs dfsadmin -report

# Formatar namenode (CUIDADO: apaga dados)
docker-compose exec namenode hdfs namenode -format
```

#### 4. Jupyter não carrega notebooks
```bash
# Verificar permissões
docker-compose exec jupyter ls -la /home/jovyan/work/

# Recriar container
docker-compose restart jupyter
```

## 📚 Documentação Adicional

### Notebooks de Exemplo
- `notebooks/01_data_collection_example.ipynb`: Coleta de dados
- `notebooks/02_spark_processing_example.ipynb`: Processamento Spark
- `notebooks/03_statistical_analysis_example.ipynb`: Análise estatística
- `notebooks/04_ml_models_example.ipynb`: Machine Learning
- `notebooks/05_visualization_example.ipynb`: Visualizações

### APIs Documentation
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [News API](https://newsapi.org/docs)
- [Apache Spark](https://spark.apache.org/docs/latest/)
- [Streamlit](https://docs.streamlit.io/)

## 🤝 Contribuição

### Guidelines
1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Code Style
- **PEP 8**: Seguir padrões Python
- **Type Hints**: Usar anotações de tipo
- **Docstrings**: Documentar funções e classes
- **Tests**: Incluir testes para novas funcionalidades

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

### Contatos da Equipe
- **Felipe Martins**: felipe.martins@email.com (Coordenação)
- **Pedro Silva**: pedro.silva@email.com (Análise)
- **Anny Caroline**: anny.sousa@email.com (ML)
- **Ricardo Areas**: ricardo.areas@email.com (Visualização)
- **Fabio**: fabio@email.com (Documentação)

### Issues e Bugs
- Reporte bugs via [GitHub Issues](https://github.com/your-repo/issues)
- Para dúvidas, use [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 🎯 Roadmap

### Versão 1.0 (Atual)
- ✅ Coleta de dados Yahoo Finance
- ✅ Processamento Spark/HDFS
- ✅ Análise estatística básica
- ✅ Modelos de ML
- ✅ Dashboard Streamlit

### Versão 1.1 (Próxima)
- 🔄 API REST para dados
- 🔄 Alertas em tempo real
- 🔄 Mais fontes de dados
- 🔄 Modelos deep learning

### Versão 2.0 (Futuro)
- 📋 Streaming de dados
- 📋 Kubernetes deployment
- 📋 Interface web completa
- 📋 Mobile app

---

**Desenvolvido com ❤️ pela Equipe Big Data Finance**