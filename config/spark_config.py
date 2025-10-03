# Configurações do Apache Spark
# Responsável: Ana Luiza Pazze

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os

class SparkConfig:
    """Configurações centralizadas para Apache Spark"""
    
    @staticmethod
    def get_spark_session(app_name="BigDataFinanceAnalysis"):
        """
        Cria e retorna uma sessão Spark configurada
        
        Args:
            app_name (str): Nome da aplicação Spark
            
        Returns:
            SparkSession: Sessão Spark configurada
        """
        conf = SparkConf()
        
        # Configurações básicas
        conf.set("spark.app.name", app_name)
        # Permitir trocar entre local e cluster via variável de ambiente
        spark_master = os.getenv("SPARK_MASTER_URL", "local[*]")
        conf.set("spark.master", spark_master)
        
        # Configurações de memória
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.driver.memory", "2g")
        conf.set("spark.executor.cores", "2")
        
        # Configurações para otimização
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Configurações para HDFS (quando disponível)
        hdfs_fs = os.getenv("HDFS_NAMENODE_URL", "hdfs://localhost:9000")
        conf.set("spark.hadoop.fs.defaultFS", hdfs_fs)
        
        # Configurações para Delta Lake
        conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        
        # Configurar nível de log
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
    
    @staticmethod
    def get_hdfs_config():
        """
        Retorna configurações para HDFS
        
        Returns:
            dict: Configurações HDFS
        """
        return {
            "hdfs_host": "localhost",
            "hdfs_port": 9000,
            "hdfs_user": "hadoop",
            "data_path": "/bigdata/finance",
            "checkpoint_path": "/bigdata/checkpoints"
        }