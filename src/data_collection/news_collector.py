# News Data Collector
# Responsável: Felipe Martins
# Coleta de dados de notícias para identificação de eventos mundiais

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class NewsCollector:
    """
    Coletor de dados de notícias para identificação de eventos mundiais
    """
    
    def __init__(self, output_path: str = "data/raw/news_data"):
        """
        Inicializa o coletor de notícias
        
        Args:
            output_path (str): Caminho para salvar os dados coletados
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Palavras-chave para eventos relevantes
        self.event_keywords = {
            'political': [
                'election', 'president', 'government', 'policy', 'brexit', 
                'trade war', 'sanctions', 'diplomatic'
            ],
            'economic': [
                'federal reserve', 'interest rate', 'inflation', 'gdp', 
                'unemployment', 'recession', 'economic crisis', 'bank collapse'
            ],
            'geopolitical': [
                'war', 'conflict', 'military', 'invasion', 'terrorism', 
                'nuclear', 'missile', 'peace talks'
            ],
            'natural_disasters': [
                'earthquake', 'tsunami', 'hurricane', 'flood', 'wildfire', 
                'pandemic', 'covid', 'climate change'
            ],
            'market_events': [
                'stock market', 'crash', 'bull market', 'bear market', 
                'volatility', 'merger', 'acquisition', 'ipo'
            ]
        }
    
    def collect_historical_events(self) -> List[Dict]:
        """
        Coleta informações sobre eventos históricos relevantes
        
        Returns:
            Lista de eventos com metadados
        """
        # Eventos históricos importantes baseados na documentação
        historical_events = [
            {
                'date': '2020-03-11',
                'event': 'WHO declares COVID-19 pandemic',
                'category': 'natural_disasters',
                'impact_level': 'high',
                'description': 'World Health Organization declares COVID-19 a global pandemic',
                'affected_markets': ['global', 'healthcare', 'travel', 'technology']
            },
            {
                'date': '2022-02-24',
                'event': 'Russia invades Ukraine',
                'category': 'geopolitical',
                'impact_level': 'high',
                'description': 'Russia launches military invasion of Ukraine',
                'affected_markets': ['energy', 'commodities', 'european', 'defense']
            },
            {
                'date': '2020-11-03',
                'event': 'US Presidential Election 2020',
                'category': 'political',
                'impact_level': 'medium',
                'description': 'US Presidential Election between Biden and Trump',
                'affected_markets': ['us_stocks', 'healthcare', 'energy', 'technology']
            },
            {
                'date': '2016-06-23',
                'event': 'Brexit Referendum',
                'category': 'political',
                'impact_level': 'high',
                'description': 'UK votes to leave European Union',
                'affected_markets': ['uk_stocks', 'european', 'currency', 'banking']
            },
            {
                'date': '2023-03-10',
                'event': 'Silicon Valley Bank Collapse',
                'category': 'economic',
                'impact_level': 'medium',
                'description': 'Silicon Valley Bank fails, triggering banking sector concerns',
                'affected_markets': ['banking', 'technology', 'regional_banks']
            },
            {
                'date': '2008-09-15',
                'event': 'Lehman Brothers Collapse',
                'category': 'economic',
                'impact_level': 'high',
                'description': 'Lehman Brothers files for bankruptcy',
                'affected_markets': ['global', 'banking', 'real_estate', 'insurance']
            },
            {
                'date': '2016-11-08',
                'event': 'Trump Election Victory',
                'category': 'political',
                'impact_level': 'medium',
                'description': 'Donald Trump wins US Presidential Election',
                'affected_markets': ['us_stocks', 'healthcare', 'infrastructure', 'energy']
            },
            {
                'date': '2020-03-23',
                'event': 'COVID-19 Market Crash',
                'category': 'natural_disasters',
                'impact_level': 'high',
                'description': 'Global markets crash due to COVID-19 pandemic fears',
                'affected_markets': ['global', 'travel', 'hospitality', 'oil']
            }
        ]
        
        return historical_events
    
    def get_event_timeline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Cria timeline de eventos para um período específico
        
        Args:
            start_date: Data de início (YYYY-MM-DD)
            end_date: Data de fim (YYYY-MM-DD)
            
        Returns:
            DataFrame com eventos no período
        """
        events = self.collect_historical_events()
        
        # Filtrar eventos no período
        filtered_events = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for event in events:
            event_dt = datetime.strptime(event['date'], '%Y-%m-%d')
            if start_dt <= event_dt <= end_dt:
                filtered_events.append(event)
        
        return pd.DataFrame(filtered_events)
    
    def analyze_event_sentiment(self, event_description: str) -> Dict:
        """
        Análise básica de sentimento de eventos
        
        Args:
            event_description: Descrição do evento
            
        Returns:
            Dicionário com análise de sentimento
        """
        # Palavras negativas que indicam impacto negativo no mercado
        negative_words = [
            'crash', 'collapse', 'war', 'invasion', 'pandemic', 'crisis',
            'recession', 'unemployment', 'inflation', 'conflict', 'disaster'
        ]
        
        # Palavras positivas
        positive_words = [
            'growth', 'recovery', 'peace', 'agreement', 'stimulus', 
            'vaccine', 'breakthrough', 'success', 'victory'
        ]
        
        description_lower = event_description.lower()
        
        negative_count = sum(1 for word in negative_words if word in description_lower)
        positive_count = sum(1 for word in positive_words if word in description_lower)
        
        if negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / (negative_count + positive_count + 1)
        elif positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / (negative_count + positive_count + 1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'negative_indicators': negative_count,
            'positive_indicators': positive_count
        }
    
    def enrich_events_data(self, events: List[Dict]) -> List[Dict]:
        """
        Enriquece dados de eventos com análise de sentimento
        
        Args:
            events: Lista de eventos
            
        Returns:
            Lista de eventos enriquecidos
        """
        enriched_events = []
        
        for event in events:
            enriched_event = event.copy()
            
            # Adicionar análise de sentimento
            sentiment_analysis = self.analyze_event_sentiment(event['description'])
            enriched_event.update(sentiment_analysis)
            
            # Adicionar janelas temporais de análise
            event_date = datetime.strptime(event['date'], '%Y-%m-%d')
            enriched_event['analysis_windows'] = {
                'pre_event_start': (event_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'pre_event_end': (event_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                'event_date': event['date'],
                'post_event_start': (event_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                'post_event_end': (event_date + timedelta(days=30)).strftime('%Y-%m-%d')
            }
            
            enriched_events.append(enriched_event)
        
        return enriched_events
    
    def save_events_data(self, events: List[Dict], filename: str = "historical_events"):
        """
        Salva dados de eventos
        
        Args:
            events: Lista de eventos
            filename: Nome do arquivo
        """
        # Salvar como JSON
        json_path = self.output_path / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        
        # Salvar como CSV
        df = pd.DataFrame(events)
        csv_path = self.output_path / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Eventos salvos em {json_path} e {csv_path}")
    
    def get_events_for_analysis(self) -> List[Dict]:
        """
        Retorna eventos preparados para análise
        
        Returns:
            Lista de eventos enriquecidos
        """
        events = self.collect_historical_events()
        enriched_events = self.enrich_events_data(events)
        
        # Salvar dados
        self.save_events_data(enriched_events)
        
        return enriched_events

# Exemplo de uso
if __name__ == "__main__":
    collector = NewsCollector()
    
    # Coletar e processar eventos
    events = collector.get_events_for_analysis()
    
    print(f"Coletados {len(events)} eventos históricos")
    for event in events[:3]:  # Mostrar primeiros 3 eventos
        print(f"- {event['date']}: {event['event']} (Sentimento: {event['sentiment']})")