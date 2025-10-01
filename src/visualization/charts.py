# Financial Charts
# Responsável: Ricardo Areas
# Gráficos especializados para análise financeira

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de visualização
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Bibliotecas para análise técnica
try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FinancialCharts:
    """
    Classe para criação de gráficos financeiros especializados
    """
    
    def __init__(self):
        """
        Inicializa a classe de gráficos financeiros
        """
        # Configurações de estilo
        self.style_config = {
            'figure_size': (12, 8),
            'dpi': 100,
            'color_palette': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8',
                'bullish': '#00ff00',
                'bearish': '#ff0000',
                'neutral': '#808080'
            },
            'font_size': 10,
            'line_width': 1.5
        }
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_action(self, data: pd.DataFrame,
                         price_col: str = 'close',
                         volume_col: Optional[str] = 'volume',
                         ma_periods: List[int] = [20, 50],
                         title: str = "Price Action Analysis",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria gráfico de ação do preço com médias móveis
        
        Args:
            data: DataFrame com dados de preços
            price_col: Nome da coluna de preços
            volume_col: Nome da coluna de volume
            ma_periods: Períodos das médias móveis
            title: Título do gráfico
            save_path: Caminho para salvar o gráfico
            
        Returns:
            Figura matplotlib
        """
        try:
            fig, axes = plt.subplots(2, 1, figsize=self.style_config['figure_size'],
                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Gráfico de preços
            ax1 = axes[0]
            ax1.plot(data.index, data[price_col], 
                    color=self.style_config['color_palette']['primary'],
                    linewidth=self.style_config['line_width'], label='Preço')
            
            # Médias móveis
            for period in ma_periods:
                ma_col = f'ma_{period}'
                if ma_col not in data.columns:
                    data[ma_col] = data[price_col].rolling(window=period).mean()
                
                ax1.plot(data.index, data[ma_col], 
                        linewidth=1, alpha=0.7, label=f'MA {period}')
            
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_ylabel('Preço', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de volume
            if volume_col and volume_col in data.columns:
                ax2 = axes[1]
                
                # Cores baseadas na direção do preço
                colors = []
                for i in range(len(data)):
                    if i == 0:
                        colors.append(self.style_config['color_palette']['neutral'])
                    else:
                        if data[price_col].iloc[i] > data[price_col].iloc[i-1]:
                            colors.append(self.style_config['color_palette']['bullish'])
                        else:
                            colors.append(self.style_config['color_palette']['bearish'])
                
                ax2.bar(data.index, data[volume_col], color=colors, alpha=0.6)
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Data', fontsize=12)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Gráfico salvo em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de preços: {str(e)}")
            return plt.figure()
    
    def plot_candlestick_mplfinance(self, data: pd.DataFrame,
                                  volume: bool = True,
                                  ma_periods: List[int] = [20, 50],
                                  title: str = "Candlestick Chart",
                                  save_path: Optional[str] = None) -> None:
        """
        Cria gráfico de candlestick usando mplfinance
        
        Args:
            data: DataFrame com dados OHLCV
            volume: Se deve incluir volume
            ma_periods: Períodos das médias móveis
            title: Título do gráfico
            save_path: Caminho para salvar
        """
        try:
            if not MPLFINANCE_AVAILABLE:
                logger.warning("mplfinance não disponível, usando método alternativo")
                return self.plot_price_action(data, title=title, save_path=save_path)
            
            # Preparar dados para mplfinance
            ohlc_data = data.copy()
            
            # Verificar se tem todas as colunas OHLC
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in ohlc_data.columns for col in required_cols):
                logger.warning("Colunas OHLC não encontradas, usando gráfico de linha")
                return self.plot_price_action(data, title=title, save_path=save_path)
            
            # Adicionar médias móveis
            ma_plots = []
            for period in ma_periods:
                ma_col = f'ma_{period}'
                ohlc_data[ma_col] = ohlc_data['close'].rolling(window=period).mean()
                ma_plots.append(mpf.make_addplot(ohlc_data[ma_col], type='line'))
            
            # Configurações do estilo
            mc = mpf.make_marketcolors(
                up='g', down='r',
                edge='inherit',
                wick={'up': 'green', 'down': 'red'},
                volume='in'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                y_on_right=False
            )
            
            # Criar gráfico
            mpf.plot(
                ohlc_data,
                type='candle',
                style=style,
                volume=volume,
                addplot=ma_plots,
                title=title,
                ylabel='Preço',
                ylabel_lower='Volume',
                figsize=self.style_config['figure_size'],
                savefig=save_path if save_path else None
            )
            
            if save_path:
                logger.info(f"Gráfico candlestick salvo em: {save_path}")
                
        except Exception as e:
            logger.error(f"Erro na criação do candlestick: {str(e)}")
    
    def plot_technical_indicators(self, data: pd.DataFrame,
                                price_col: str = 'close',
                                indicators: List[str] = ['rsi', 'macd', 'bb'],
                                title: str = "Technical Indicators",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria gráfico com indicadores técnicos
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            indicators: Lista de indicadores a plotar
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            # Calcular indicadores se não existirem
            df = self._calculate_technical_indicators(data, price_col)
            
            # Determinar número de subplots
            n_plots = 1 + len(indicators)
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
            
            if n_plots == 1:
                axes = [axes]
            
            # Gráfico de preços
            ax_price = axes[0]
            ax_price.plot(df.index, df[price_col], 
                         color=self.style_config['color_palette']['primary'],
                         linewidth=self.style_config['line_width'], label='Preço')
            
            # Bollinger Bands se solicitado
            if 'bb' in indicators and all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                ax_price.plot(df.index, df['bb_upper'], 'r--', alpha=0.7, label='BB Superior')
                ax_price.plot(df.index, df['bb_lower'], 'r--', alpha=0.7, label='BB Inferior')
                ax_price.plot(df.index, df['bb_middle'], 'b--', alpha=0.7, label='BB Média')
                ax_price.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
            
            ax_price.set_title(title, fontsize=14, fontweight='bold')
            ax_price.set_ylabel('Preço', fontsize=12)
            ax_price.legend()
            ax_price.grid(True, alpha=0.3)
            
            # Indicadores individuais
            plot_idx = 1
            
            # RSI
            if 'rsi' in indicators and 'rsi' in df.columns:
                ax_rsi = axes[plot_idx]
                ax_rsi.plot(df.index, df['rsi'], 
                           color=self.style_config['color_palette']['warning'],
                           linewidth=self.style_config['line_width'])
                ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.7)
                ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.7)
                ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
                ax_rsi.set_ylabel('RSI', fontsize=12)
                ax_rsi.set_ylim(0, 100)
                ax_rsi.grid(True, alpha=0.3)
                plot_idx += 1
            
            # MACD
            if 'macd' in indicators and all(col in df.columns for col in ['macd', 'macd_signal']):
                ax_macd = axes[plot_idx]
                ax_macd.plot(df.index, df['macd'], 
                            color=self.style_config['color_palette']['primary'],
                            linewidth=self.style_config['line_width'], label='MACD')
                ax_macd.plot(df.index, df['macd_signal'], 
                            color=self.style_config['color_palette']['danger'],
                            linewidth=self.style_config['line_width'], label='Signal')
                
                if 'macd_histogram' in df.columns:
                    colors = ['g' if x >= 0 else 'r' for x in df['macd_histogram']]
                    ax_macd.bar(df.index, df['macd_histogram'], color=colors, alpha=0.6)
                
                ax_macd.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax_macd.set_ylabel('MACD', fontsize=12)
                ax_macd.legend()
                ax_macd.grid(True, alpha=0.3)
                plot_idx += 1
            
            plt.xlabel('Data', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Gráfico de indicadores salvo em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação dos indicadores técnicos: {str(e)}")
            return plt.figure()
    
    def plot_correlation_matrix(self, data: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              method: str = 'pearson',
                              title: str = "Correlation Matrix",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria matriz de correlação
        
        Args:
            data: DataFrame com dados
            columns: Colunas para análise
            method: Método de correlação
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            # Selecionar colunas
            if columns is None:
                numeric_data = data.select_dtypes(include=[np.number])
            else:
                numeric_data = data[columns]
            
            # Calcular correlação
            corr_matrix = numeric_data.corr(method=method)
            
            # Criar gráfico
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Matriz de correlação salva em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação da matriz de correlação: {str(e)}")
            return plt.figure()
    
    def plot_returns_distribution(self, data: pd.DataFrame,
                                price_col: str = 'close',
                                title: str = "Returns Distribution",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria gráfico de distribuição de retornos
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            # Calcular retornos
            returns = data[price_col].pct_change().dropna()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histograma
            ax1 = axes[0, 0]
            ax1.hist(returns, bins=50, alpha=0.7, color=self.style_config['color_palette']['primary'])
            ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Média: {returns.mean():.4f}')
            ax1.axvline(returns.median(), color='green', linestyle='--', label=f'Mediana: {returns.median():.4f}')
            ax1.set_title('Distribuição de Retornos')
            ax1.set_xlabel('Retorno')
            ax1.set_ylabel('Frequência')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q Plot
            from scipy import stats
            ax2 = axes[0, 1]
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal)')
            ax2.grid(True, alpha=0.3)
            
            # Box Plot
            ax3 = axes[1, 0]
            ax3.boxplot(returns, vert=True)
            ax3.set_title('Box Plot de Retornos')
            ax3.set_ylabel('Retorno')
            ax3.grid(True, alpha=0.3)
            
            # Série temporal de retornos
            ax4 = axes[1, 1]
            ax4.plot(returns.index, returns, alpha=0.7, 
                    color=self.style_config['color_palette']['primary'])
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_title('Série Temporal de Retornos')
            ax4.set_xlabel('Data')
            ax4.set_ylabel('Retorno')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Gráfico de distribuição salvo em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação da distribuição de retornos: {str(e)}")
            return plt.figure()
    
    def plot_volatility_analysis(self, data: pd.DataFrame,
                               price_col: str = 'close',
                               windows: List[int] = [20, 60],
                               title: str = "Volatility Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria análise de volatilidade
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            windows: Janelas para cálculo de volatilidade
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            # Calcular retornos e volatilidades
            returns = data[price_col].pct_change()
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Gráfico de preços
            ax1 = axes[0]
            ax1.plot(data.index, data[price_col], 
                    color=self.style_config['color_palette']['primary'],
                    linewidth=self.style_config['line_width'])
            ax1.set_title('Preços')
            ax1.set_ylabel('Preço')
            ax1.grid(True, alpha=0.3)
            
            # Retornos
            ax2 = axes[1]
            ax2.plot(returns.index, returns, alpha=0.7,
                    color=self.style_config['color_palette']['secondary'])
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_title('Retornos Diários')
            ax2.set_ylabel('Retorno')
            ax2.grid(True, alpha=0.3)
            
            # Volatilidades
            ax3 = axes[2]
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, window in enumerate(windows):
                volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
                ax3.plot(volatility.index, volatility, 
                        color=colors[i % len(colors)],
                        linewidth=self.style_config['line_width'],
                        label=f'Vol {window}d')
            
            ax3.set_title('Volatilidade Anualizada')
            ax3.set_xlabel('Data')
            ax3.set_ylabel('Volatilidade')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Análise de volatilidade salva em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na análise de volatilidade: {str(e)}")
            return plt.figure()
    
    def plot_event_impact(self, price_data: pd.DataFrame,
                         events: pd.DataFrame,
                         price_col: str = 'close',
                         event_date_col: str = 'date',
                         event_desc_col: str = 'description',
                         window_days: int = 30,
                         title: str = "Event Impact Analysis",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria gráfico de impacto de eventos
        
        Args:
            price_data: DataFrame com dados de preços
            events: DataFrame com eventos
            price_col: Nome da coluna de preços
            event_date_col: Nome da coluna de data dos eventos
            event_desc_col: Nome da coluna de descrição
            window_days: Janela de análise em dias
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            
            # Plotar preços
            ax.plot(price_data.index, price_data[price_col],
                   color=self.style_config['color_palette']['primary'],
                   linewidth=self.style_config['line_width'], label='Preço')
            
            # Marcar eventos
            for _, event in events.iterrows():
                event_date = pd.to_datetime(event[event_date_col])
                
                # Verificar se a data está no range dos dados
                if event_date in price_data.index:
                    event_price = price_data.loc[event_date, price_col]
                    
                    # Linha vertical no evento
                    ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
                    
                    # Anotação do evento
                    ax.annotate(event[event_desc_col][:30] + "...",
                              xy=(event_date, event_price),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                              fontsize=8)
                    
                    # Análise de janela
                    start_date = event_date - timedelta(days=window_days//2)
                    end_date = event_date + timedelta(days=window_days//2)
                    
                    window_data = price_data.loc[start_date:end_date]
                    if not window_data.empty:
                        # Destacar período
                        ax.axvspan(start_date, end_date, alpha=0.1, color='red')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Preço', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Análise de impacto de eventos salva em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na análise de impacto de eventos: {str(e)}")
            return plt.figure()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame,
                                      price_col: str = 'close') -> pd.DataFrame:
        """
        Calcula indicadores técnicos básicos
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            
        Returns:
            DataFrame com indicadores
        """
        try:
            df = data.copy()
            
            # RSI
            if 'rsi' not in df.columns:
                delta = df[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if 'macd' not in df.columns:
                ema_12 = df[price_col].ewm(span=12).mean()
                ema_26 = df[price_col].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            if 'bb_middle' not in df.columns:
                df['bb_middle'] = df[price_col].rolling(window=20).mean()
                bb_std = df[price_col].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores: {str(e)}")
            return data
    
    def create_multi_asset_comparison(self, data_dict: Dict[str, pd.DataFrame],
                                    price_col: str = 'close',
                                    normalize: bool = True,
                                    title: str = "Multi-Asset Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria comparação entre múltiplos ativos
        
        Args:
            data_dict: Dicionário com DataFrames de diferentes ativos
            price_col: Nome da coluna de preços
            normalize: Se deve normalizar os preços
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
            
            for i, (asset_name, asset_data) in enumerate(data_dict.items()):
                if price_col in asset_data.columns:
                    prices = asset_data[price_col].dropna()
                    
                    if normalize:
                        # Normalizar para base 100
                        normalized_prices = (prices / prices.iloc[0]) * 100
                        ax.plot(normalized_prices.index, normalized_prices,
                               color=colors[i], linewidth=self.style_config['line_width'],
                               label=asset_name)
                    else:
                        ax.plot(prices.index, prices,
                               color=colors[i], linewidth=self.style_config['line_width'],
                               label=asset_name)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Preço Normalizado (Base 100)' if normalize else 'Preço', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.style_config['dpi'], bbox_inches='tight')
                logger.info(f"Comparação multi-ativo salva em: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na comparação multi-ativo: {str(e)}")
            return plt.figure()

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simular dados OHLCV
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': close_prices + np.random.randn(len(dates)) * 0.5,
        'high': close_prices + np.abs(np.random.randn(len(dates)) * 1.0),
        'low': close_prices - np.abs(np.random.randn(len(dates)) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }).set_index('date')
    
    # Inicializar classe de gráficos
    charts = FinancialCharts()
    
    # Criar gráficos
    print("Criando gráficos de exemplo...")
    
    # Gráfico de preços
    price_fig = charts.plot_price_action(sample_data, title="Análise de Preços - Exemplo")
    
    # Indicadores técnicos
    tech_fig = charts.plot_technical_indicators(sample_data, title="Indicadores Técnicos - Exemplo")
    
    # Distribuição de retornos
    dist_fig = charts.plot_returns_distribution(sample_data, title="Distribuição de Retornos - Exemplo")
    
    # Análise de volatilidade
    vol_fig = charts.plot_volatility_analysis(sample_data, title="Análise de Volatilidade - Exemplo")
    
    print("Gráficos criados com sucesso!")
    plt.show()