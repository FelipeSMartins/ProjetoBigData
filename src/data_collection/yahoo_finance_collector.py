# Yahoo Finance Data Collector
# Responsável: Felipe Martins
# Integração com Yahoo Finance API para coleta de dados financeiros

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import time
import logging
from pathlib import Path
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """
    Coletor de dados financeiros da Yahoo Finance API
    
    Baseado na documentação: https://python-yahoofinance.readthedocs.io/en/latest/api.html
    """
    
    def __init__(self, output_path: str = "data/raw/financial_data"):
        """
        Inicializa o coletor
        
        Args:
            output_path (str): Caminho para salvar os dados coletados
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Símbolos dos principais índices e ações conforme documentação
        self.major_indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            'FTSE 100': '^FTSE',
            'Nikkei 225': '^N225',
            'DAX': '^GDAXI',
            'Bovespa': '^BVSP'
        }
        
        self.major_stocks = {
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Google': 'GOOGL',
            'Amazon': 'AMZN',
            'Tesla': 'TSLA',
            'Meta': 'META',
            'Netflix': 'NFLX',
            'Nvidia': 'NVDA'
        }
        
        self.commodities = {
            'Gold': 'GC=F',
            'Oil': 'CL=F',
            'Natural Gas': 'NG=F',
            'Silver': 'SI=F'
        }
        
        self.currencies = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'USD/BRL': 'USDBRL=X',
            'USD/RUB': 'USDRUB=X'
        }
    
    def collect_historical_data(self, 
                              symbols: Union[str, List[str]], 
                              start_date: str, 
                              end_date: str,
                              interval: str = '1d') -> pd.DataFrame:
        """
        Coleta dados históricos para os símbolos especificados
        
        Args:
            symbols: Símbolo ou lista de símbolos para coletar
            start_date: Data de início (formato YYYY-MM-DD)
            end_date: Data de fim (formato YYYY-MM-DD)
            interval: Intervalo dos dados (1d, 1wk, 1mo, etc.)
            
        Returns:
            DataFrame com os dados históricos
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Coletando dados para {symbol}")
                
                # Usar yfinance para coleta
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not data.empty:
                    # Adicionar informações do símbolo
                    data['Symbol'] = symbol
                    data['Date'] = data.index
                    data.reset_index(drop=True, inplace=True)
                    
                    all_data.append(data)
                    
                    # Rate limiting para evitar bloqueios
                    time.sleep(0.1)
                else:
                    logger.warning(f"Nenhum dado encontrado para {symbol}")
                    
            except Exception as e:
                logger.error(f"Erro ao coletar dados para {symbol}: {str(e)}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def collect_company_info(self, symbols: Union[str, List[str]]) -> Dict:
        """
        Coleta informações das empresas
        
        Args:
            symbols: Símbolo ou lista de símbolos
            
        Returns:
            Dicionário com informações das empresas
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        company_info = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                company_info[symbol] = {
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'country': info.get('country', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'currency': info.get('currency', 'USD')
                }
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Erro ao coletar info para {symbol}: {str(e)}")
                company_info[symbol] = {'error': str(e)}
        
        return company_info
    
    def collect_event_period_data(self, 
                                event_date: str, 
                                days_before: int = 30, 
                                days_after: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados para um período específico ao redor de um evento
        
        Args:
            event_date: Data do evento (YYYY-MM-DD)
            days_before: Dias antes do evento para coletar
            days_after: Dias após o evento para coletar
            
        Returns:
            Dicionário com DataFrames para cada categoria de ativo
        """
        event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        start_date = (event_dt - timedelta(days=days_before)).strftime('%Y-%m-%d')
        end_date = (event_dt + timedelta(days=days_after)).strftime('%Y-%m-%d')
        
        logger.info(f"Coletando dados do evento de {start_date} a {end_date}")
        
        results = {}
        
        # Coletar dados dos índices
        indices_data = self.collect_historical_data(
            list(self.major_indices.values()), 
            start_date, 
            end_date
        )
        if not indices_data.empty:
            results['indices'] = indices_data
        
        # Coletar dados das ações
        stocks_data = self.collect_historical_data(
            list(self.major_stocks.values()), 
            start_date, 
            end_date
        )
        if not stocks_data.empty:
            results['stocks'] = stocks_data
        
        # Coletar dados das commodities
        commodities_data = self.collect_historical_data(
            list(self.commodities.values()), 
            start_date, 
            end_date
        )
        if not commodities_data.empty:
            results['commodities'] = commodities_data
        
        # Coletar dados das moedas
        currencies_data = self.collect_historical_data(
            list(self.currencies.values()), 
            start_date, 
            end_date
        )
        if not currencies_data.empty:
            results['currencies'] = currencies_data
        
        return results
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'parquet', hdfs_manager=None, hdfs_subdir: str = "/bigdata/finance/raw"):
        """
        Salva os dados coletados
        
        Args:
            data: DataFrame com os dados
            filename: Nome do arquivo
            format: Formato do arquivo (parquet, csv)
        """
        if data.empty:
            logger.warning("DataFrame vazio, não salvando")
            return
        
        filepath = self.output_path / f"{filename}.{format}"
        
        try:
            if format == 'parquet':
                data.to_parquet(filepath, index=False)
            elif format == 'csv':
                data.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"Dados salvos em {filepath}")

            # Se um HDFSManager for fornecido, enviar arquivo para HDFS
            if hdfs_manager is not None:
                hdfs_target = os.path.join(hdfs_subdir, f"{filename}.{format}")
                ok = hdfs_manager.upload_file(str(filepath), hdfs_target)
                if ok:
                    logger.info(f"Arquivo enviado ao HDFS: {hdfs_target}")
                else:
                    logger.error(f"Falha ao enviar arquivo ao HDFS: {hdfs_target}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {str(e)}")
    
    def collect_major_events_data(self) -> Dict[str, Dict]:
        """
        Coleta dados para os principais eventos mencionados na documentação
        
        Returns:
            Dicionário com dados de cada evento
        """
        major_events = {
            'covid_start': '2020-03-11',  # OMS declara pandemia
            'russia_ukraine_war': '2022-02-24',  # Início da guerra
            'us_election_2020': '2020-11-03',  # Eleição presidencial
            'brexit_referendum': '2016-06-23',  # Referendo Brexit
            'svb_collapse': '2023-03-10',  # Colapso do SVB
        }
        
        all_events_data = {}
        
        for event_name, event_date in major_events.items():
            logger.info(f"Coletando dados para evento: {event_name}")
            
            event_data = self.collect_event_period_data(event_date)
            all_events_data[event_name] = event_data
            
            # Salvar dados de cada evento
            for asset_type, data in event_data.items():
                filename = f"{event_name}_{asset_type}"
                self.save_data(data, filename)
        
        return all_events_data

# Exemplo de uso
if __name__ == "__main__":
    collector = YahooFinanceCollector()
    
    # Coletar dados de um evento específico
    covid_data = collector.collect_event_period_data('2020-03-11')
    
    # Salvar dados
    for asset_type, data in covid_data.items():
        collector.save_data(data, f"covid_start_{asset_type}")
    
    print("Coleta de dados concluída!")