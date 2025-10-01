# Statistical Analyzer
# Responsável: Pedro Silva
# Análise estatística e exploratória de dados financeiros

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Analisador estatístico para dados financeiros e eventos mundiais
    """
    
    def __init__(self):
        """
        Inicializa o analisador estatístico
        """
        self.results = {}
        
    def descriptive_statistics(self, data: pd.DataFrame, 
                             columns: List[str] = None) -> Dict[str, Any]:
        """
        Calcula estatísticas descritivas básicas
        
        Args:
            data: DataFrame com dados financeiros
            columns: Colunas para análise (se None, usa todas numéricas)
            
        Returns:
            Dicionário com estatísticas descritivas
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            stats_dict = {}
            
            for col in columns:
                if col in data.columns:
                    series = data[col].dropna()
                    
                    stats_dict[col] = {
                        'count': len(series),
                        'mean': series.mean(),
                        'median': series.median(),
                        'std': series.std(),
                        'var': series.var(),
                        'min': series.min(),
                        'max': series.max(),
                        'q25': series.quantile(0.25),
                        'q75': series.quantile(0.75),
                        'skewness': stats.skew(series),
                        'kurtosis': stats.kurtosis(series),
                        'range': series.max() - series.min(),
                        'iqr': series.quantile(0.75) - series.quantile(0.25),
                        'cv': series.std() / series.mean() if series.mean() != 0 else np.nan
                    }
            
            logger.info(f"Estatísticas descritivas calculadas para {len(columns)} variáveis")
            return stats_dict
            
        except Exception as e:
            logger.error(f"Erro no cálculo de estatísticas descritivas: {str(e)}")
            return {}
    
    def normality_tests(self, data: pd.DataFrame, 
                       columns: List[str] = None) -> Dict[str, Dict]:
        """
        Executa testes de normalidade
        
        Args:
            data: DataFrame com dados
            columns: Colunas para testar
            
        Returns:
            Resultados dos testes de normalidade
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            normality_results = {}
            
            for col in columns:
                if col in data.columns:
                    series = data[col].dropna()
                    
                    if len(series) < 3:
                        continue
                    
                    results = {}
                    
                    # Teste Shapiro-Wilk (para amostras pequenas)
                    if len(series) <= 5000:
                        shapiro_stat, shapiro_p = shapiro(series)
                        results['shapiro'] = {
                            'statistic': shapiro_stat,
                            'p_value': shapiro_p,
                            'is_normal': shapiro_p > 0.05
                        }
                    
                    # Teste Jarque-Bera
                    jb_stat, jb_p = jarque_bera(series)
                    results['jarque_bera'] = {
                        'statistic': jb_stat,
                        'p_value': jb_p,
                        'is_normal': jb_p > 0.05
                    }
                    
                    # Teste D'Agostino-Pearson
                    da_stat, da_p = normaltest(series)
                    results['dagostino_pearson'] = {
                        'statistic': da_stat,
                        'p_value': da_p,
                        'is_normal': da_p > 0.05
                    }
                    
                    normality_results[col] = results
            
            logger.info(f"Testes de normalidade executados para {len(normality_results)} variáveis")
            return normality_results
            
        except Exception as e:
            logger.error(f"Erro nos testes de normalidade: {str(e)}")
            return {}
    
    def correlation_analysis(self, data: pd.DataFrame, 
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Análise de correlação entre variáveis
        
        Args:
            data: DataFrame com dados
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            
        Returns:
            Matriz de correlação e estatísticas
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                logger.warning("Nenhuma variável numérica encontrada")
                return {}
            
            # Matriz de correlação
            corr_matrix = numeric_data.corr(method=method)
            
            # Correlações mais fortes
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(corr_value):
                        corr_pairs.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value)
                        })
            
            # Ordenar por correlação absoluta
            corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            results = {
                'correlation_matrix': corr_matrix,
                'top_correlations': corr_pairs[:10],
                'method': method,
                'strong_positive': [p for p in corr_pairs if p['correlation'] > 0.7],
                'strong_negative': [p for p in corr_pairs if p['correlation'] < -0.7]
            }
            
            logger.info(f"Análise de correlação concluída usando método {method}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise de correlação: {str(e)}")
            return {}
    
    def volatility_analysis(self, price_data: pd.DataFrame, 
                          price_column: str = 'close',
                          window: int = 30) -> Dict[str, Any]:
        """
        Análise de volatilidade dos preços
        
        Args:
            price_data: DataFrame com dados de preços
            price_column: Nome da coluna de preços
            window: Janela para volatilidade móvel
            
        Returns:
            Métricas de volatilidade
        """
        try:
            if price_column not in price_data.columns:
                logger.error(f"Coluna {price_column} não encontrada")
                return {}
            
            prices = price_data[price_column].dropna()
            
            # Retornos logarítmicos
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Volatilidade histórica (anualizada)
            historical_vol = log_returns.std() * np.sqrt(252)
            
            # Volatilidade móvel
            rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
            
            # Métricas de volatilidade
            vol_metrics = {
                'historical_volatility': historical_vol,
                'mean_rolling_volatility': rolling_vol.mean(),
                'max_rolling_volatility': rolling_vol.max(),
                'min_rolling_volatility': rolling_vol.min(),
                'volatility_of_volatility': rolling_vol.std(),
                'current_volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.nan,
                'rolling_volatility': rolling_vol,
                'log_returns': log_returns
            }
            
            logger.info("Análise de volatilidade concluída")
            return vol_metrics
            
        except Exception as e:
            logger.error(f"Erro na análise de volatilidade: {str(e)}")
            return {}
    
    def event_impact_analysis(self, price_data: pd.DataFrame,
                            event_dates: List[datetime],
                            price_column: str = 'close',
                            window_before: int = 5,
                            window_after: int = 5) -> Dict[str, Any]:
        """
        Análise do impacto de eventos nos preços
        
        Args:
            price_data: DataFrame com dados de preços (index deve ser datetime)
            event_dates: Lista de datas dos eventos
            price_column: Nome da coluna de preços
            window_before: Dias antes do evento para análise
            window_after: Dias após o evento para análise
            
        Returns:
            Análise do impacto dos eventos
        """
        try:
            if price_column not in price_data.columns:
                logger.error(f"Coluna {price_column} não encontrada")
                return {}
            
            # Garantir que o index é datetime
            if not isinstance(price_data.index, pd.DatetimeIndex):
                logger.error("Index do DataFrame deve ser datetime")
                return {}
            
            prices = price_data[price_column].dropna()
            returns = prices.pct_change().dropna()
            
            event_impacts = []
            
            for event_date in event_dates:
                try:
                    # Definir janelas
                    start_date = event_date - timedelta(days=window_before)
                    end_date = event_date + timedelta(days=window_after)
                    
                    # Filtrar dados da janela
                    window_prices = prices[start_date:end_date]
                    window_returns = returns[start_date:end_date]
                    
                    if len(window_prices) < 3:
                        continue
                    
                    # Preço antes e depois do evento
                    event_idx = window_prices.index.get_indexer([event_date], method='nearest')[0]
                    
                    if event_idx > 0 and event_idx < len(window_prices) - 1:
                        price_before = window_prices.iloc[event_idx - 1]
                        price_event = window_prices.iloc[event_idx]
                        price_after = window_prices.iloc[event_idx + 1] if event_idx + 1 < len(window_prices) else price_event
                        
                        # Calcular impactos
                        immediate_impact = (price_event - price_before) / price_before
                        next_day_impact = (price_after - price_event) / price_event
                        total_impact = (price_after - price_before) / price_before
                        
                        # Volatilidade na janela
                        window_volatility = window_returns.std()
                        
                        event_impacts.append({
                            'event_date': event_date,
                            'price_before': price_before,
                            'price_event': price_event,
                            'price_after': price_after,
                            'immediate_impact': immediate_impact,
                            'next_day_impact': next_day_impact,
                            'total_impact': total_impact,
                            'window_volatility': window_volatility,
                            'abnormal_return': abs(immediate_impact) > 2 * returns.std()
                        })
                
                except Exception as e:
                    logger.warning(f"Erro ao processar evento {event_date}: {str(e)}")
                    continue
            
            if not event_impacts:
                logger.warning("Nenhum impacto de evento calculado")
                return {}
            
            # Estatísticas agregadas
            impacts_df = pd.DataFrame(event_impacts)
            
            summary = {
                'total_events': len(event_impacts),
                'mean_immediate_impact': impacts_df['immediate_impact'].mean(),
                'mean_total_impact': impacts_df['total_impact'].mean(),
                'positive_events': len(impacts_df[impacts_df['immediate_impact'] > 0]),
                'negative_events': len(impacts_df[impacts_df['immediate_impact'] < 0]),
                'abnormal_returns': len(impacts_df[impacts_df['abnormal_return'] == True]),
                'max_positive_impact': impacts_df['immediate_impact'].max(),
                'max_negative_impact': impacts_df['immediate_impact'].min(),
                'impact_volatility': impacts_df['immediate_impact'].std()
            }
            
            results = {
                'event_impacts': event_impacts,
                'summary': summary,
                'impacts_dataframe': impacts_df
            }
            
            logger.info(f"Análise de impacto concluída para {len(event_impacts)} eventos")
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise de impacto de eventos: {str(e)}")
            return {}
    
    def time_series_analysis(self, data: pd.Series) -> Dict[str, Any]:
        """
        Análise básica de séries temporais
        
        Args:
            data: Série temporal
            
        Returns:
            Análise da série temporal
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Teste de estacionariedade (Augmented Dickey-Fuller)
            adf_result = adfuller(data.dropna())
            
            # Teste de autocorrelação (Ljung-Box)
            lb_result = acorr_ljungbox(data.dropna(), lags=10, return_df=True)
            
            # Tendência simples
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
            
            results = {
                'stationarity': {
                    'adf_statistic': adf_result[0],
                    'adf_pvalue': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05,
                    'critical_values': adf_result[4]
                },
                'autocorrelation': {
                    'ljung_box_results': lb_result,
                    'has_autocorrelation': any(lb_result['lb_pvalue'] < 0.05)
                },
                'trend': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'has_trend': p_value < 0.05
                }
            }
            
            logger.info("Análise de séries temporais concluída")
            return results
            
        except ImportError:
            logger.warning("statsmodels não disponível para análise avançada de séries temporais")
            return {}
        except Exception as e:
            logger.error(f"Erro na análise de séries temporais: {str(e)}")
            return {}
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Gera relatório textual das análises
        
        Args:
            analysis_results: Resultados das análises
            
        Returns:
            Relatório em texto
        """
        try:
            report = []
            report.append("=" * 60)
            report.append("RELATÓRIO DE ANÁLISE ESTATÍSTICA")
            report.append("=" * 60)
            report.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Estatísticas descritivas
            if 'descriptive_stats' in analysis_results:
                report.append("ESTATÍSTICAS DESCRITIVAS")
                report.append("-" * 30)
                for var, stats in analysis_results['descriptive_stats'].items():
                    report.append(f"\n{var}:")
                    report.append(f"  Média: {stats['mean']:.4f}")
                    report.append(f"  Desvio Padrão: {stats['std']:.4f}")
                    report.append(f"  Assimetria: {stats['skewness']:.4f}")
                    report.append(f"  Curtose: {stats['kurtosis']:.4f}")
                report.append("")
            
            # Correlações
            if 'correlations' in analysis_results:
                report.append("PRINCIPAIS CORRELAÇÕES")
                report.append("-" * 30)
                for corr in analysis_results['correlations']['top_correlations'][:5]:
                    report.append(f"{corr['variable1']} vs {corr['variable2']}: {corr['correlation']:.4f}")
                report.append("")
            
            # Volatilidade
            if 'volatility' in analysis_results:
                vol = analysis_results['volatility']
                report.append("ANÁLISE DE VOLATILIDADE")
                report.append("-" * 30)
                report.append(f"Volatilidade Histórica: {vol['historical_volatility']:.4f}")
                report.append(f"Volatilidade Atual: {vol['current_volatility']:.4f}")
                report.append("")
            
            # Impacto de eventos
            if 'event_impact' in analysis_results:
                impact = analysis_results['event_impact']['summary']
                report.append("IMPACTO DE EVENTOS")
                report.append("-" * 30)
                report.append(f"Total de eventos analisados: {impact['total_events']}")
                report.append(f"Impacto médio imediato: {impact['mean_immediate_impact']:.4f}")
                report.append(f"Eventos positivos: {impact['positive_events']}")
                report.append(f"Eventos negativos: {impact['negative_events']}")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            return "Erro na geração do relatório"

# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    df.set_index('date', inplace=True)
    
    # Inicializar analisador
    analyzer = StatisticalAnalyzer()
    
    # Executar análises
    desc_stats = analyzer.descriptive_statistics(df)
    correlations = analyzer.correlation_analysis(df)
    volatility = analyzer.volatility_analysis(df)
    
    print("Análise estatística concluída!")
    print(f"Variáveis analisadas: {list(desc_stats.keys())}")
    print(f"Volatilidade histórica: {volatility['historical_volatility']:.4f}")