# Yahoo Spark Job
# Processa dados brutos coletados do Yahoo Finance no HDFS e grava Parquet

from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_date, when, lag
from pyspark.sql.window import Window
import logging

from src.infrastructure.spark_manager import SparkManager

logger = logging.getLogger(__name__)


def process_raw_to_curated(raw_dir: str = "/bigdata/finance/raw",
                           output_dir: str = "/bigdata/finance/processed/yahoo_daily",
                           app_name: str = "YahooFinanceProcessing") -> Optional[DataFrame]:
    """
    Lê arquivos brutos (CSV/Parquet) do HDFS, padroniza esquemas e grava Parquet curado.

    Args:
        raw_dir: Diretório no HDFS com os arquivos brutos
        output_dir: Diretório de saída no HDFS para dados processados
        app_name: Nome da aplicação Spark

    Returns:
        DataFrame resultante após processamento, ou None em caso de erro
    """

    try:
        sm = SparkManager(app_name=app_name)
        spark = sm.get_spark_session()

        # Tenta ler Parquet primeiro, depois CSV
        df = None
        try:
            df = spark.read.parquet(f"{raw_dir}/*")
            logger.info("Arquivos Parquet lidos do HDFS")
        except Exception:
            df = spark.read.option("header", True).csv(f"{raw_dir}/*")
            logger.info("Arquivos CSV lidos do HDFS")

        # Campos esperados padrão do yfinance
        expected_cols = [
            "Open", "High", "Low", "Close", "Volume", "Date", "Symbol"
        ]
        for c in expected_cols:
            if c not in df.columns:
                logger.warning(f"Coluna ausente: {c}")

        # Tipagens e limpeza
        df_clean = (
            df
            .withColumn("Date", to_date(col("Date")))
            .withColumn("Open", col("Open").cast("double"))
            .withColumn("High", col("High").cast("double"))
            .withColumn("Low", col("Low").cast("double"))
            .withColumn("Close", col("Close").cast("double"))
            .withColumn("Volume", col("Volume").cast("long"))
        )

        # Retorno diário por símbolo
        w = Window.partitionBy("Symbol").orderBy("Date")
        df_clean = df_clean.withColumn(
            "PrevClose",
            lag("Close").over(w)
        ).withColumn(
            "DailyReturn",
            when(col("PrevClose").isNull(), None)
            .otherwise((col("Close") - col("PrevClose")) / col("PrevClose"))
        )

        # Gravar Parquet particionado
        sm.save_to_hdfs(df_clean, output_dir, format="parquet", mode="overwrite")
        logger.info(f"Dados processados gravados em {output_dir}")

        return df_clean

    except Exception as e:
        logger.error(f"Erro no job Spark: {str(e)}")
        return None


if __name__ == "__main__":
    process_raw_to_curated()