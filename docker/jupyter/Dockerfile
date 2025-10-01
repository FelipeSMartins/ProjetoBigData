# Dockerfile para Jupyter Notebook com Spark e Big Data
FROM jupyter/pyspark-notebook:spark-3.4.0

# Mudar para usuário root para instalações
USER root

# Instalar dependências adicionais do sistema
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Voltar para usuário jovyan
USER $NB_UID

# Instalar dependências Python específicas do projeto
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Instalar extensões úteis do Jupyter
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server \
    jupyterlab_code_formatter \
    black \
    isort

# Configurar Spark para Jupyter
ENV PYSPARK_DRIVER_PYTHON=jupyter
ENV PYSPARK_DRIVER_PYTHON_OPTS='lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""'

# Criar diretórios de trabalho
RUN mkdir -p /home/jovyan/work/notebooks \
             /home/jovyan/work/src \
             /home/jovyan/work/data \
             /home/jovyan/work/config

# Configurar PYTHONPATH
ENV PYTHONPATH=/home/jovyan/work/src:$PYTHONPATH

# Expor porta do Jupyter
EXPOSE 8888

# Comando de inicialização
CMD ["start-notebook.sh", "--NotebookApp.token=bigdata2024"]