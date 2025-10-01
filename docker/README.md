# Configurações Docker

Esta pasta contém os arquivos de configuração para containerização do ambiente.

## Arquivos

- `Dockerfile` - Imagem principal da aplicação
- `docker-compose.yml` - Orquestração dos serviços
- `spark/` - Configurações específicas do Spark
- `hdfs/` - Configurações do HDFS
- `jupyter/` - Configurações do Jupyter Notebook

## Uso

```bash
# Construir e iniciar todos os serviços
docker-compose up -d

# Parar todos os serviços
docker-compose down
```