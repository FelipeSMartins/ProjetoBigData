# Dockerfile para aplicação Big Data - Análise de Impacto de Eventos Mundiais
# Base: Python 3.11 (mais estável que slim para apt) com Spark e dependências

FROM python:3.11-bullseye

# Metadados
LABEL maintainer="Equipe Big Data"
LABEL description="Aplicação para análise de impacto de eventos mundiais nos mercados financeiros"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Atualizar e instalar dependências do sistema + ferramentas de build necessárias para pip
RUN apt-get update --fix-missing && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       build-essential \
       python3-dev \
       openjdk-11-jdk-headless \
       wget \
       curl \
       procps \
       net-tools \
       vim \
       git \
       unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar Apache Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz \
    && tar -xzf spark-3.4.0-bin-hadoop3.tgz \
    && mv spark-3.4.0-bin-hadoop3 /opt/spark \
    && rm spark-3.4.0-bin-hadoop3.tgz

# Instalar Hadoop Client (para HDFS)
RUN wget -q https://archive.apache.org/dist/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz \
    && tar -xzf hadoop-3.2.1.tar.gz \
    && mv hadoop-3.2.1 /opt/hadoop \
    && rm hadoop-3.2.1.tar.gz

# Configurar Hadoop
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Criar diretório da aplicação
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY src/ ./src/
COPY config/ ./config/
COPY notebooks/ ./notebooks/

# Criar diretórios necessários
RUN mkdir -p /app/data/raw /app/data/processed /app/logs

# Configurar permissões
RUN chmod +x /opt/spark/bin/* /opt/spark/sbin/*
RUN chmod +x /opt/hadoop/bin/* /opt/hadoop/sbin/*

# Copiar configurações do Spark (se houver)
COPY docker/spark/conf/* /opt/spark/conf/

# Configurar PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expor portas
EXPOSE 4040 4041 4042 4043

# Comando padrão
CMD ["python", "-c", "print('Big Data Application Ready'); import time; time.sleep(3600)"]
