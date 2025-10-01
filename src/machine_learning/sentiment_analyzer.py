# Sentiment Analyzer
# Responsável: Anny Caroline Sousa
# Análise de sentimento de notícias e eventos mundiais

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de processamento de texto
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analisador de sentimento para notícias e eventos financeiros
    """
    
    def __init__(self):
        """
        Inicializa o analisador de sentimento
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download de recursos NLTK necessários
        self._download_nltk_resources()
        
        # Palavras-chave financeiras para contexto
        self.financial_keywords = {
            'positive': [
                'growth', 'profit', 'gain', 'rise', 'increase', 'bull', 'rally',
                'surge', 'boom', 'recovery', 'expansion', 'optimism', 'confidence',
                'strong', 'robust', 'healthy', 'improvement', 'upgrade', 'beat'
            ],
            'negative': [
                'loss', 'decline', 'fall', 'crash', 'bear', 'recession', 'crisis',
                'drop', 'plunge', 'collapse', 'downturn', 'weakness', 'concern',
                'fear', 'uncertainty', 'risk', 'volatility', 'downgrade', 'miss'
            ],
            'neutral': [
                'stable', 'unchanged', 'flat', 'sideways', 'consolidation',
                'range', 'mixed', 'moderate', 'steady', 'maintain'
            ]
        }
    
    def _download_nltk_resources(self):
        """
        Baixa recursos necessários do NLTK
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto para análise de sentimento
        
        Args:
            text: Texto para processar
            
        Returns:
            Texto processado
        """
        try:
            if not isinstance(text, str):
                return ""
            
            # Converter para minúsculas
            text = text.lower()
            
            # Remover URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remover menções e hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remover caracteres especiais, manter apenas letras, números e espaços
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # Remover espaços extras
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {str(e)}")
            return text
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Análise de sentimento usando TextBlob
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com métricas de sentimento
        """
        try:
            processed_text = self.preprocess_text(text)
            blob = TextBlob(processed_text)
            
            # Polaridade: -1 (negativo) a 1 (positivo)
            # Subjetividade: 0 (objetivo) a 1 (subjetivo)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classificar sentimento
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment_label,
                'confidence': abs(polarity)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise TextBlob: {str(e)}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Análise de sentimento usando VADER
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com métricas de sentimento
        """
        try:
            processed_text = self.preprocess_text(text)
            scores = self.vader_analyzer.polarity_scores(processed_text)
            
            # Determinar sentimento dominante
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment_label = 'positive'
            elif compound <= -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': compound,
                'sentiment': sentiment_label,
                'confidence': abs(compound)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise VADER: {str(e)}")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
    
    def analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Análise de sentimento específica para contexto financeiro
        
        Args:
            text: Texto para análise
            
        Returns:
            Análise de sentimento financeiro
        """
        try:
            processed_text = self.preprocess_text(text)
            words = word_tokenize(processed_text)
            
            # Contar palavras-chave financeiras
            positive_count = sum(1 for word in words if word in self.financial_keywords['positive'])
            negative_count = sum(1 for word in words if word in self.financial_keywords['negative'])
            neutral_count = sum(1 for word in words if word in self.financial_keywords['neutral'])
            
            total_financial_words = positive_count + negative_count + neutral_count
            
            # Calcular scores
            if total_financial_words > 0:
                positive_ratio = positive_count / total_financial_words
                negative_ratio = negative_count / total_financial_words
                neutral_ratio = neutral_count / total_financial_words
            else:
                positive_ratio = negative_ratio = neutral_ratio = 0.0
            
            # Score financeiro composto
            financial_score = positive_ratio - negative_ratio
            
            # Determinar sentimento financeiro
            if financial_score > 0.2:
                financial_sentiment = 'bullish'
            elif financial_score < -0.2:
                financial_sentiment = 'bearish'
            else:
                financial_sentiment = 'neutral'
            
            return {
                'financial_score': financial_score,
                'financial_sentiment': financial_sentiment,
                'positive_keywords': positive_count,
                'negative_keywords': negative_count,
                'neutral_keywords': neutral_count,
                'total_financial_keywords': total_financial_words,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento financeiro: {str(e)}")
            return {
                'financial_score': 0.0,
                'financial_sentiment': 'neutral',
                'positive_keywords': 0,
                'negative_keywords': 0,
                'neutral_keywords': 0,
                'total_financial_keywords': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
    
    def comprehensive_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Análise de sentimento abrangente combinando múltiplos métodos
        
        Args:
            text: Texto para análise
            
        Returns:
            Análise completa de sentimento
        """
        try:
            # Análises individuais
            textblob_result = self.analyze_sentiment_textblob(text)
            vader_result = self.analyze_sentiment_vader(text)
            financial_result = self.analyze_financial_sentiment(text)
            
            # Score combinado
            combined_score = (
                textblob_result['polarity'] * 0.3 +
                vader_result['compound'] * 0.4 +
                financial_result['financial_score'] * 0.3
            )
            
            # Sentimento final baseado no score combinado
            if combined_score > 0.1:
                final_sentiment = 'positive'
            elif combined_score < -0.1:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            # Confiança baseada na concordância entre métodos
            sentiments = [
                textblob_result['sentiment'],
                vader_result['sentiment'],
                financial_result['financial_sentiment']
            ]
            
            # Calcular concordância
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
            max_agreement = max(sentiment_counts.values())
            confidence = max_agreement / len(sentiments)
            
            return {
                'text': text,
                'final_sentiment': final_sentiment,
                'combined_score': combined_score,
                'confidence': confidence,
                'textblob': textblob_result,
                'vader': vader_result,
                'financial': financial_result,
                'agreement_ratio': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise abrangente: {str(e)}")
            return {
                'text': text,
                'final_sentiment': 'neutral',
                'combined_score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def analyze_batch_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """
        Análise de sentimento em lote
        
        Args:
            texts: Lista de textos para análise
            
        Returns:
            DataFrame com resultados
        """
        try:
            results = []
            
            for i, text in enumerate(texts):
                if pd.isna(text) or text == "":
                    continue
                
                analysis = self.comprehensive_sentiment_analysis(text)
                
                # Extrair informações principais
                result = {
                    'text_id': i,
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'final_sentiment': analysis['final_sentiment'],
                    'combined_score': analysis['combined_score'],
                    'confidence': analysis['confidence'],
                    'textblob_polarity': analysis['textblob']['polarity'],
                    'textblob_subjectivity': analysis['textblob']['subjectivity'],
                    'vader_compound': analysis['vader']['compound'],
                    'vader_positive': analysis['vader']['positive'],
                    'vader_negative': analysis['vader']['negative'],
                    'financial_score': analysis['financial']['financial_score'],
                    'financial_sentiment': analysis['financial']['financial_sentiment'],
                    'financial_keywords': analysis['financial']['total_financial_keywords']
                }
                
                results.append(result)
            
            df = pd.DataFrame(results)
            logger.info(f"Análise de sentimento concluída para {len(results)} textos")
            return df
            
        except Exception as e:
            logger.error(f"Erro na análise em lote: {str(e)}")
            return pd.DataFrame()
    
    def sentiment_time_series(self, data: pd.DataFrame,
                            text_column: str,
                            date_column: str,
                            aggregation: str = 'daily') -> pd.DataFrame:
        """
        Cria série temporal de sentimento
        
        Args:
            data: DataFrame com textos e datas
            text_column: Nome da coluna de texto
            date_column: Nome da coluna de data
            aggregation: Tipo de agregação ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame com série temporal de sentimento
        """
        try:
            # Análise de sentimento
            sentiment_results = self.analyze_batch_sentiment(data[text_column].tolist())
            
            # Adicionar datas
            sentiment_results['date'] = data[date_column].values[:len(sentiment_results)]
            sentiment_results['date'] = pd.to_datetime(sentiment_results['date'])
            
            # Definir frequência de agregação
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M'
            }
            
            freq = freq_map.get(aggregation, 'D')
            
            # Agrupar por período
            grouped = sentiment_results.groupby(pd.Grouper(key='date', freq=freq)).agg({
                'combined_score': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'textblob_polarity': 'mean',
                'vader_compound': 'mean',
                'financial_score': 'mean'
            }).reset_index()
            
            # Simplificar nomes das colunas
            grouped.columns = [
                'date', 'sentiment_mean', 'sentiment_std', 'text_count',
                'confidence_mean', 'textblob_mean', 'vader_mean', 'financial_mean'
            ]
            
            # Classificar sentimento médio
            grouped['sentiment_category'] = grouped['sentiment_mean'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
            
            logger.info(f"Série temporal de sentimento criada com agregação {aggregation}")
            return grouped
            
        except Exception as e:
            logger.error(f"Erro na criação da série temporal: {str(e)}")
            return pd.DataFrame()
    
    def sentiment_impact_correlation(self, sentiment_data: pd.DataFrame,
                                   price_data: pd.DataFrame,
                                   sentiment_column: str = 'sentiment_mean',
                                   price_column: str = 'close') -> Dict[str, Any]:
        """
        Analisa correlação entre sentimento e preços
        
        Args:
            sentiment_data: DataFrame com dados de sentimento
            price_data: DataFrame com dados de preços
            sentiment_column: Nome da coluna de sentimento
            price_column: Nome da coluna de preços
            
        Returns:
            Análise de correlação
        """
        try:
            # Garantir que ambos DataFrames têm coluna de data como index
            if 'date' in sentiment_data.columns:
                sentiment_data = sentiment_data.set_index('date')
            if 'date' in price_data.columns:
                price_data = price_data.set_index('date')
            
            # Calcular retornos dos preços
            price_data['returns'] = price_data[price_column].pct_change()
            
            # Merge dos dados por data
            merged_data = pd.merge(
                sentiment_data[[sentiment_column]],
                price_data[['returns']],
                left_index=True,
                right_index=True,
                how='inner'
            ).dropna()
            
            if merged_data.empty:
                logger.warning("Nenhum dado comum encontrado para correlação")
                return {}
            
            # Calcular correlações
            correlation = merged_data[sentiment_column].corr(merged_data['returns'])
            
            # Correlação com lag (sentimento hoje vs retorno amanhã)
            merged_data['returns_next'] = merged_data['returns'].shift(-1)
            lag_correlation = merged_data[sentiment_column].corr(merged_data['returns_next'])
            
            # Análise por quartis de sentimento
            merged_data['sentiment_quartile'] = pd.qcut(
                merged_data[sentiment_column], 
                q=4, 
                labels=['Very Negative', 'Negative', 'Positive', 'Very Positive']
            )
            
            quartile_returns = merged_data.groupby('sentiment_quartile')['returns'].agg([
                'mean', 'std', 'count'
            ]).round(4)
            
            results = {
                'correlation': correlation,
                'lag_correlation': lag_correlation,
                'sample_size': len(merged_data),
                'quartile_analysis': quartile_returns.to_dict(),
                'data_period': {
                    'start': merged_data.index.min(),
                    'end': merged_data.index.max()
                }
            }
            
            logger.info(f"Correlação sentimento-preço calculada: {correlation:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise de correlação: {str(e)}")
            return {}
    
    def generate_sentiment_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Gera relatório de análise de sentimento
        
        Args:
            analysis_results: Resultados das análises
            
        Returns:
            Relatório em texto
        """
        try:
            report = []
            report.append("=" * 60)
            report.append("RELATÓRIO DE ANÁLISE DE SENTIMENTO")
            report.append("=" * 60)
            report.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Estatísticas gerais
            if 'batch_results' in analysis_results:
                df = analysis_results['batch_results']
                report.append("ESTATÍSTICAS GERAIS")
                report.append("-" * 30)
                report.append(f"Total de textos analisados: {len(df)}")
                
                sentiment_counts = df['final_sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    pct = (count / len(df)) * 100
                    report.append(f"{sentiment.capitalize()}: {count} ({pct:.1f}%)")
                
                report.append(f"Score médio: {df['combined_score'].mean():.4f}")
                report.append(f"Confiança média: {df['confidence'].mean():.4f}")
                report.append("")
            
            # Correlação com preços
            if 'correlation' in analysis_results:
                corr = analysis_results['correlation']
                report.append("CORRELAÇÃO COM PREÇOS")
                report.append("-" * 30)
                report.append(f"Correlação contemporânea: {corr['correlation']:.4f}")
                report.append(f"Correlação com lag: {corr['lag_correlation']:.4f}")
                report.append(f"Tamanho da amostra: {corr['sample_size']}")
                report.append("")
            
            # Análise temporal
            if 'time_series' in analysis_results:
                ts = analysis_results['time_series']
                report.append("ANÁLISE TEMPORAL")
                report.append("-" * 30)
                report.append(f"Período: {ts['date'].min()} a {ts['date'].max()}")
                report.append(f"Sentimento médio: {ts['sentiment_mean'].mean():.4f}")
                report.append(f"Volatilidade do sentimento: {ts['sentiment_std'].mean():.4f}")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            return "Erro na geração do relatório"

# Exemplo de uso
if __name__ == "__main__":
    # Textos de exemplo
    sample_texts = [
        "The stock market rallied today on positive economic news",
        "Concerns about inflation are weighing on investor sentiment",
        "Company reports strong quarterly earnings, beating expectations",
        "Market volatility increases amid geopolitical tensions"
    ]
    
    # Inicializar analisador
    analyzer = SentimentAnalyzer()
    
    # Análise individual
    result = analyzer.comprehensive_sentiment_analysis(sample_texts[0])
    print(f"Sentimento: {result['final_sentiment']}")
    print(f"Score: {result['combined_score']:.4f}")
    
    # Análise em lote
    batch_results = analyzer.analyze_batch_sentiment(sample_texts)
    print(f"\nAnálise em lote concluída para {len(batch_results)} textos")
    print(batch_results[['final_sentiment', 'combined_score', 'confidence']].head())