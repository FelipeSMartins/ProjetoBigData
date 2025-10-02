# Imagem base com Java 11
FROM openjdk:11-jdk-slim

# Evita warnings do Python
ENV PYTHONUNBUFFERED=1

# Instala Python 3 e pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Define python e pip padrão (forçando se já existir)
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Cria pasta do app
WORKDIR /app

# Copia dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY app.py .

# Comando default
CMD ["python", "app.py"]
