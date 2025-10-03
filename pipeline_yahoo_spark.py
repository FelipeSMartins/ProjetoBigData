"""
Pipeline Yahoo Finance -> HDFS -> Spark

Uso:
  - Requer serviços do Docker em execução (namenode, datanode, spark-master)
  - Variáveis: HDFS_NAMENODE_URL, SPARK_MASTER_URL opcionais

Etapas:
  1) Coletar dados do Yahoo Finance e salvar localmente
  2) Enviar arquivos para HDFS (/bigdata/finance/raw)
  3) Executar job Spark para processar e gerar Parquet curado
"""

import os
from datetime import datetime, timedelta
import logging

from src.data_collection.yahoo_finance_collector import YahooFinanceCollector
from src.infrastructure.hdfs_manager import HDFSManager
from src.data_processing.yahoo_spark_job import process_raw_to_curated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")


def run_pipeline(event_date: str = None,
                 days_before: int = 30,
                 days_after: int = 0,
                 output_format: str = "parquet") -> None:
    # 1) Preparar HDFS
    hdfs = HDFSManager()
    ok = hdfs.setup_hdfs_environment()
    if not ok:
        logger.error("HDFS não está configurado/acessível. Verifique containers do Hadoop.")
        return

    # 2) Coletar dados
    collector = YahooFinanceCollector(output_path="data/raw/financial_data")

    if event_date is None:
        # Coleta do último mês
        end = datetime.today()
        start = end - timedelta(days=30)
        data = collector.collect_historical_data(
            symbols=list(collector.major_indices.values()) +
                    list(collector.major_stocks.values()) +
                    list(collector.commodities.values()) +
                    list(collector.currencies.values()),
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            interval="1d",
        )

        if data is not None and not data.empty:
            collector.save_data(data, f"yahoo_last30days", format=output_format, hdfs_manager=hdfs)
        else:
            logger.warning("Nenhum dado coletado para o período." )
    else:
        # Coleta ao redor de um evento
        results = collector.collect_event_period_data(event_date, days_before, days_after)
        for asset_type, df in results.items():
            collector.save_data(df, f"event_{event_date}_{asset_type}", format=output_format, hdfs_manager=hdfs)

    # 3) Processar com Spark
    process_raw_to_curated()
    logger.info("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    run_pipeline()