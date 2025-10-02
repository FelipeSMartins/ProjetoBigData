# Yahoo Finance Data Collector - Docker-ready
# Responsável: Felipe Martins / Anny Sousa
# Coleta de dados financeiros confiáveis da Yahoo Finance API

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import List, Dict, Union

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """
    Coletor de dados financeiros da Yahoo Finance API
    """
    
    def __init__(self, output_path: str = "data/raw"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Apenas tickers confiáveis para teste inicial
        self.major_indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI'
        }
        
        self.major_stocks = {
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Google': 'GOOGL',
            'Amazon': 'AMZN'
        }

    def collect_historical_data(self, 
                                symbols: Union[str, List[str]], 
                                start_date: str, 
                                end_date: str,
                                interval: str = '1d') -> pd.DataFrame:
        """
        Coleta dados históricos para os símbolos especificados
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_data = []
        for symbol in symbols:
            try:
                logger.info(f"Coletando dados para {symbol}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not data.empty:
                    data['Symbol'] = symbol
                    data['Date'] = data.index
                    data.reset_index(drop=True, inplace=True)
                    all_data.append(data)
                    time.sleep(0.1)  # Rate limiting
                else:
                    logger.warning(f"Nenhum dado encontrado para {symbol}")
            except Exception as e:
                logger.error(f"Erro ao coletar dados para {symbol}: {str(e)}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Salva os dados coletados
        """
        if data.empty:
            logger.warning(f"DataFrame vazio para {filename}, não salvando")
            return
        
        filepath = self.output_path / f"{filename}.{format}"
        try:
            if format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'parquet':
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            logger.info(f"Dados salvos em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {str(e)}")
    
    def collect_event_period_data(self, 
                                  event_date: str, 
                                  days_before: int = 30, 
                                  days_after: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados para um período específico ao redor de um evento
        """
        event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        start_date = (event_dt - timedelta(days=days_before)).strftime('%Y-%m-%d')
        end_date = (event_dt + timedelta(days=days_after)).strftime('%Y-%m-%d')
        
        logger.info(f"Coletando dados do evento de {start_date} a {end_date}")
        results = {}
        
        indices_data = self.collect_historical_data(list(self.major_indices.values()), start_date, end_date)
        if not indices_data.empty:
            results['indices'] = indices_data
        
        stocks_data = self.collect_historical_data(list(self.major_stocks.values()), start_date, end_date)
        if not stocks_data.empty:
            results['stocks'] = stocks_data
        
        return results

if __name__ == "__main__":
    collector = YahooFinanceCollector()
    
    # Exemplo de coleta: evento "covid_start"
    covid_data = collector.collect_event_period_data('2020-03-11')
    
    for asset_type, data in covid_data.items():
        collector.save_data(data, f"covid_start_{asset_type}", format='csv')
    
    print("Coleta de dados concluída!")
