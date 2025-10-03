# Spark Manager
# Responsável: Ana Luiza Pazze
# Gerenciamento do Apache Spark para processamento distribuído

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.conf import SparkConf
import logging
from pathlib import Path
from typing import Optional, Dict, List
import os

logger = logging.getLogger(__name__)

class SparkManager:
    """
    Gerenciador do Apache Spark para processamento distribuído de dados financeiros
    """
    
    def __init__(self, app_name: str = "BigDataFinanceAnalysis"):
        """
        Inicializa o gerenciador Spark
        
        Args:
            app_name (str): Nome da aplicação Spark
        """
        self.app_name = app_name
        self.spark = None
        # URL do HDFS pode ser configurada via variável de ambiente (ex.: hdfs://namenode:9000)
        self.hdfs_url = os.getenv("HDFS_NAMENODE_URL", "hdfs://localhost:9000")
    
    def create_spark_session(self, 
                           master: str = "local[*]",
                           executor_memory: str = "4g",
                           driver_memory: str = "2g") -> SparkSession:
        """
        Cria sessão Spark otimizada para análise financeira
        
        Args:
            master: Configuração do master Spark
            executor_memory: Memória do executor
            driver_memory: Memória do driver
            
        Returns:
            SparkSession configurada
        """
        try:
            conf = SparkConf()
            
            # Configurações básicas
            conf.set("spark.app.name", self.app_name)
            conf.set("spark.master", master)
            
            # Configurações de memória
            conf.set("spark.executor.memory", executor_memory)
            conf.set("spark.driver.memory", driver_memory)
            conf.set("spark.executor.cores", "2")
            conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            
            # Configurações de otimização
            conf.set("spark.sql.adaptive.enabled", "true")
            conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
            
            # Configurações para Delta Lake
            conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            # Configurações HDFS
            conf.set("spark.hadoop.fs.defaultFS", self.hdfs_url)
            
            # Configurações para processamento de dados financeiros
            conf.set("spark.sql.shuffle.partitions", "200")
            conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            
            self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            
            logger.info(f"Sessão Spark criada: {self.app_name}")
            return self.spark
            
        except Exception as e:
            logger.error(f"Erro ao criar sessão Spark: {str(e)}")
            raise
    
    def load_financial_data(self, 
                          file_path: str, 
                          format: str = "parquet") -> DataFrame:
        """
        Carrega dados financeiros no Spark
        
        Args:
            file_path: Caminho para o arquivo
            format: Formato do arquivo (parquet, csv, json)
            
        Returns:
            DataFrame Spark com os dados
        """
        try:
            if format == "parquet":
                df = self.spark.read.parquet(file_path)
            elif format == "csv":
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
            elif format == "json":
                df = self.spark.read.json(file_path)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"Dados carregados: {df.count()} registros de {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de {file_path}: {str(e)}")
            raise
    
    def process_financial_data(self, df: DataFrame) -> DataFrame:
        """
        Processa dados financeiros com transformações básicas
        
        Args:
            df: DataFrame com dados financeiros
            
        Returns:
            DataFrame processado
        """
        try:
            # Adicionar colunas calculadas
            processed_df = df.withColumn(
                "daily_return", 
                (col("Close") - col("Open")) / col("Open") * 100
            ).withColumn(
                "volatility",
                abs(col("High") - col("Low")) / col("Open") * 100
            ).withColumn(
                "volume_ma_7",
                avg("Volume").over(
                    Window.partitionBy("Symbol").orderBy("Date").rowsBetween(-6, 0)
                )
            ).withColumn(
                "price_ma_7",
                avg("Close").over(
                    Window.partitionBy("Symbol").orderBy("Date").rowsBetween(-6, 0)
                )
            ).withColumn(
                "price_ma_30",
                avg("Close").over(
                    Window.partitionBy("Symbol").orderBy("Date").rowsBetween(-29, 0)
                )
            )
            
            logger.info("Dados financeiros processados com sucesso")
            return processed_df
            
        except Exception as e:
            logger.error(f"Erro ao processar dados financeiros: {str(e)}")
            raise
    
    def calculate_event_impact(self, 
                             df: DataFrame, 
                             event_date: str,
                             pre_days: int = 30,
                             post_days: int = 30) -> DataFrame:
        """
        Calcula impacto de eventos nos dados financeiros
        
        Args:
            df: DataFrame com dados financeiros
            event_date: Data do evento (YYYY-MM-DD)
            pre_days: Dias antes do evento
            post_days: Dias após o evento
            
        Returns:
            DataFrame com análise de impacto
        """
        try:
            # Definir períodos
            event_date_col = to_date(lit(event_date))
            
            impact_df = df.withColumn(
                "days_from_event",
                datediff(col("Date"), event_date_col)
            ).withColumn(
                "period",
                when(col("days_from_event") < -pre_days, "before_window")
                .when(col("days_from_event").between(-pre_days, -1), "pre_event")
                .when(col("days_from_event") == 0, "event_day")
                .when(col("days_from_event").between(1, post_days), "post_event")
                .when(col("days_from_event") > post_days, "after_window")
                .otherwise("unknown")
            )
            
            # Calcular estatísticas por período
            period_stats = impact_df.groupBy("Symbol", "period").agg(
                avg("daily_return").alias("avg_return"),
                stddev("daily_return").alias("return_volatility"),
                avg("Volume").alias("avg_volume"),
                count("*").alias("trading_days")
            )
            
            logger.info(f"Análise de impacto calculada para evento em {event_date}")
            return period_stats
            
        except Exception as e:
            logger.error(f"Erro ao calcular impacto do evento: {str(e)}")
            raise
    
    def save_to_hdfs(self, 
                     df: DataFrame, 
                     path: str, 
                     format: str = "parquet",
                     mode: str = "overwrite") -> None:
        """
        Salva DataFrame no HDFS
        
        Args:
            df: DataFrame para salvar
            path: Caminho no HDFS
            format: Formato do arquivo
            mode: Modo de escrita (overwrite, append)
        """
        try:
            hdfs_path = f"{self.hdfs_url}{path}"
            
            if format == "parquet":
                df.write.mode(mode).parquet(hdfs_path)
            elif format == "delta":
                df.write.format("delta").mode(mode).save(hdfs_path)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"Dados salvos no HDFS: {hdfs_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar no HDFS: {str(e)}")
            raise
    
    def create_etl_pipeline(self, 
                          input_path: str, 
                          output_path: str,
                          event_date: Optional[str] = None) -> None:
        """
        Pipeline ETL completo para dados financeiros
        
        Args:
            input_path: Caminho dos dados de entrada
            output_path: Caminho dos dados de saída
            event_date: Data do evento para análise (opcional)
        """
        try:
            logger.info("Iniciando pipeline ETL")
            
            # Extract
            raw_df = self.load_financial_data(input_path)
            
            # Transform
            processed_df = self.process_financial_data(raw_df)
            
            # Análise de evento se especificado
            if event_date:
                event_impact_df = self.calculate_event_impact(processed_df, event_date)
                event_output_path = f"{output_path}/event_analysis"
                self.save_to_hdfs(event_impact_df, event_output_path)
            
            # Load
            self.save_to_hdfs(processed_df, f"{output_path}/processed_data")
            
            logger.info("Pipeline ETL concluído com sucesso")
            
        except Exception as e:
            logger.error(f"Erro no pipeline ETL: {str(e)}")
            raise
    
    def get_spark_ui_url(self) -> str:
        """
        Retorna URL da interface web do Spark
        
        Returns:
            URL da interface Spark
        """
        if self.spark:
            return self.spark.sparkContext.uiWebUrl
        return "Spark não inicializado"
    
    def stop_spark(self) -> None:
        """
        Para a sessão Spark
        """
        if self.spark:
            self.spark.stop()
            logger.info("Sessão Spark finalizada")

# Exemplo de uso
if __name__ == "__main__":
    spark_manager = SparkManager()
    
    # Criar sessão
    spark = spark_manager.create_spark_session()
    
    print(f"Spark UI disponível em: {spark_manager.get_spark_ui_url()}")
    
    # Finalizar
    spark_manager.stop_spark()