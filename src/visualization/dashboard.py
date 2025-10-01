# Interactive Dashboard
# Responsável: Ricardo Areas
# Dashboard interativo para visualização de dados financeiros e análises

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de visualização
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Streamlit para dashboard web
import streamlit as st
from streamlit_option_menu import option_menu

# Bibliotecas auxiliares
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class FinancialDashboard:
    """
    Dashboard interativo para análise financeira
    """
    
    def __init__(self):
        """
        Inicializa o dashboard
        """
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.layout_config = {
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    
    def create_price_chart(self, data: pd.DataFrame,
                          price_column: str = 'close',
                          volume_column: Optional[str] = 'volume',
                          title: str = "Análise de Preços") -> go.Figure:
        """
        Cria gráfico de preços com volume
        
        Args:
            data: DataFrame com dados
            price_column: Nome da coluna de preços
            volume_column: Nome da coluna de volume
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Criar subplots
            if volume_column and volume_column in data.columns:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Preço', 'Volume'),
                    row_width=[0.7, 0.3]
                )
            else:
                fig = go.Figure()
            
            # Gráfico de preços
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_column],
                    mode='lines',
                    name='Preço',
                    line=dict(color=self.color_palette['primary'], width=2),
                    hovertemplate='<b>Data:</b> %{x}<br><b>Preço:</b> %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Adicionar médias móveis se disponíveis
            for ma_period in [20, 50]:
                ma_col = f'ma_{ma_period}'
                if ma_col in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[ma_col],
                            mode='lines',
                            name=f'MA {ma_period}',
                            line=dict(width=1, dash='dash'),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
            
            # Gráfico de volume
            if volume_column and volume_column in data.columns:
                colors = ['red' if data[price_column].iloc[i] < data[price_column].iloc[i-1] 
                         else 'green' for i in range(1, len(data))]
                colors.insert(0, 'green')  # Primeiro valor
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[volume_column],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.6,
                        hovertemplate='<b>Data:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Configurar layout
            fig.update_layout(
                title=title,
                xaxis_title="Data",
                yaxis_title="Preço",
                **self.layout_config,
                showlegend=True,
                height=600
            )
            
            if volume_column and volume_column in data.columns:
                fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de preços: {str(e)}")
            return go.Figure()
    
    def create_candlestick_chart(self, data: pd.DataFrame,
                               open_col: str = 'open',
                               high_col: str = 'high',
                               low_col: str = 'low',
                               close_col: str = 'close',
                               volume_col: Optional[str] = 'volume',
                               title: str = "Gráfico de Candlestick") -> go.Figure:
        """
        Cria gráfico de candlestick
        
        Args:
            data: DataFrame com dados OHLCV
            open_col: Nome da coluna de abertura
            high_col: Nome da coluna de máxima
            low_col: Nome da coluna de mínima
            close_col: Nome da coluna de fechamento
            volume_col: Nome da coluna de volume
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Verificar se todas as colunas existem
            required_cols = [open_col, high_col, low_col, close_col]
            if not all(col in data.columns for col in required_cols):
                logger.warning("Colunas OHLC não encontradas, usando gráfico de linha")
                return self.create_price_chart(data, close_col, volume_col, title)
            
            # Criar subplots
            if volume_col and volume_col in data.columns:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Preço', 'Volume'),
                    row_width=[0.7, 0.3]
                )
            else:
                fig = go.Figure()
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data[open_col],
                    high=data[high_col],
                    low=data[low_col],
                    close=data[close_col],
                    name="OHLC",
                    increasing_line_color=self.color_palette['success'],
                    decreasing_line_color=self.color_palette['danger']
                ),
                row=1, col=1
            )
            
            # Volume
            if volume_col and volume_col in data.columns:
                colors = ['red' if data[close_col].iloc[i] < data[open_col].iloc[i] 
                         else 'green' for i in range(len(data))]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[volume_col],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            
            # Layout
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                **self.layout_config,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de candlestick: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, data: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 title: str = "Matriz de Correlação") -> go.Figure:
        """
        Cria heatmap de correlação
        
        Args:
            data: DataFrame com dados
            columns: Colunas para correlação
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Selecionar colunas numéricas
            if columns is None:
                numeric_data = data.select_dtypes(include=[np.number])
            else:
                numeric_data = data[columns]
            
            # Calcular correlação
            corr_matrix = numeric_data.corr()
            
            # Criar heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlação: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                **self.layout_config,
                height=500,
                width=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do heatmap: {str(e)}")
            return go.Figure()
    
    def create_sentiment_chart(self, sentiment_data: pd.DataFrame,
                             sentiment_col: str = 'sentiment_score',
                             price_data: Optional[pd.DataFrame] = None,
                             price_col: str = 'close',
                             title: str = "Análise de Sentimento") -> go.Figure:
        """
        Cria gráfico de análise de sentimento
        
        Args:
            sentiment_data: DataFrame com dados de sentimento
            sentiment_col: Nome da coluna de sentimento
            price_data: DataFrame com dados de preços (opcional)
            price_col: Nome da coluna de preços
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Criar subplots se tiver dados de preço
            if price_data is not None:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Sentimento', 'Preço'),
                    row_width=[0.5, 0.5]
                )
            else:
                fig = go.Figure()
            
            # Gráfico de sentimento
            colors = sentiment_data[sentiment_col].apply(
                lambda x: self.color_palette['success'] if x > 0 else 
                         (self.color_palette['danger'] if x < 0 else self.color_palette['info'])
            )
            
            fig.add_trace(
                go.Bar(
                    x=sentiment_data.index,
                    y=sentiment_data[sentiment_col],
                    name='Sentimento',
                    marker_color=colors,
                    hovertemplate='<b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Linha zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # Gráfico de preços se disponível
            if price_data is not None and price_col in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data[price_col],
                        mode='lines',
                        name='Preço',
                        line=dict(color=self.color_palette['primary'], width=2)
                    ),
                    row=2, col=1
                )
            
            # Layout
            fig.update_layout(
                title=title,
                **self.layout_config,
                height=600 if price_data is not None else 400
            )
            
            fig.update_yaxes(title_text="Score de Sentimento", row=1, col=1)
            if price_data is not None:
                fig.update_yaxes(title_text="Preço", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de sentimento: {str(e)}")
            return go.Figure()
    
    def create_volatility_chart(self, data: pd.DataFrame,
                              price_col: str = 'close',
                              window: int = 20,
                              title: str = "Análise de Volatilidade") -> go.Figure:
        """
        Cria gráfico de volatilidade
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            window: Janela para cálculo da volatilidade
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Calcular retornos e volatilidade
            returns = data[price_col].pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
            
            # Criar subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Preço', 'Volatilidade'),
                row_width=[0.6, 0.4]
            )
            
            # Gráfico de preços
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name='Preço',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Gráfico de volatilidade
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=volatility,
                    mode='lines',
                    name=f'Volatilidade ({window}d)',
                    line=dict(color=self.color_palette['warning'], width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            # Adicionar linha de volatilidade média
            vol_mean = volatility.mean()
            fig.add_hline(
                y=vol_mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {vol_mean:.2%}",
                row=2, col=1
            )
            
            # Layout
            fig.update_layout(
                title=title,
                **self.layout_config,
                height=600
            )
            
            fig.update_yaxes(title_text="Preço", row=1, col=1)
            fig.update_yaxes(title_text="Volatilidade Anualizada", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de volatilidade: {str(e)}")
            return go.Figure()
    
    def create_performance_metrics_chart(self, data: pd.DataFrame,
                                       price_col: str = 'close',
                                       benchmark_col: Optional[str] = None,
                                       title: str = "Métricas de Performance") -> go.Figure:
        """
        Cria gráfico de métricas de performance
        
        Args:
            data: DataFrame com dados
            price_col: Nome da coluna de preços
            benchmark_col: Nome da coluna de benchmark
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            # Calcular retornos cumulativos
            returns = data[price_col].pct_change().fillna(0)
            cumulative_returns = (1 + returns).cumprod() - 1
            
            fig = go.Figure()
            
            # Retornos cumulativos do ativo
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=cumulative_returns * 100,
                    mode='lines',
                    name='Retorno Cumulativo',
                    line=dict(color=self.color_palette['primary'], width=2),
                    hovertemplate='<b>Data:</b> %{x}<br><b>Retorno:</b> %{y:.2f}%<extra></extra>'
                )
            )
            
            # Benchmark se disponível
            if benchmark_col and benchmark_col in data.columns:
                benchmark_returns = data[benchmark_col].pct_change().fillna(0)
                benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=benchmark_cumulative * 100,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.color_palette['secondary'], width=2, dash='dash')
                    )
                )
            
            # Linha zero
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            # Layout
            fig.update_layout(
                title=title,
                xaxis_title="Data",
                yaxis_title="Retorno Cumulativo (%)",
                **self.layout_config,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de performance: {str(e)}")
            return go.Figure()
    
    def create_distribution_chart(self, data: pd.DataFrame,
                                column: str,
                                title: str = "Distribuição de Retornos") -> go.Figure:
        """
        Cria gráfico de distribuição
        
        Args:
            data: DataFrame com dados
            column: Nome da coluna para análise
            title: Título do gráfico
            
        Returns:
            Figura Plotly
        """
        try:
            values = data[column].dropna()
            
            # Criar subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Histograma', 'Box Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histograma
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=50,
                    name='Distribuição',
                    marker_color=self.color_palette['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=values,
                    name='Box Plot',
                    marker_color=self.color_palette['secondary']
                ),
                row=1, col=2
            )
            
            # Estatísticas
            mean_val = values.mean()
            std_val = values.std()
            
            # Adicionar linhas de referência no histograma
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {mean_val:.4f}",
                row=1, col=1
            )
            
            # Layout
            fig.update_layout(
                title=title,
                **self.layout_config,
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro na criação do gráfico de distribuição: {str(e)}")
            return go.Figure()
    
    def export_chart_to_html(self, fig: go.Figure, filename: str) -> str:
        """
        Exporta gráfico para HTML
        
        Args:
            fig: Figura Plotly
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        try:
            filepath = f"{filename}.html"
            fig.write_html(filepath)
            logger.info(f"Gráfico exportado para {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erro na exportação: {str(e)}")
            return ""
    
    def create_dashboard_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Cria resumo para dashboard
        
        Args:
            data: Dicionário com DataFrames
            
        Returns:
            Dicionário com métricas resumidas
        """
        try:
            summary = {}
            
            for name, df in data.items():
                if df.empty:
                    continue
                
                # Métricas básicas
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    summary[name] = {
                        'total_records': len(df),
                        'date_range': {
                            'start': df.index.min() if hasattr(df.index, 'min') else None,
                            'end': df.index.max() if hasattr(df.index, 'max') else None
                        },
                        'numeric_columns': len(numeric_cols),
                        'missing_values': df.isnull().sum().sum(),
                        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
                    }
                    
                    # Métricas específicas para dados de preços
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().dropna()
                        summary[name].update({
                            'price_metrics': {
                                'current_price': df['close'].iloc[-1],
                                'price_change': df['close'].iloc[-1] - df['close'].iloc[0],
                                'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                                'volatility': returns.std() * np.sqrt(252) * 100,  # Anualizada
                                'max_drawdown': ((df['close'] / df['close'].cummax()) - 1).min() * 100
                            }
                        })
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro na criação do resumo: {str(e)}")
            return {}

# Streamlit Dashboard Application
def create_streamlit_dashboard():
    """
    Cria aplicação Streamlit para dashboard
    """
    st.set_page_config(
        page_title="Financial Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS customizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 Financial Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Menu de navegação
        selected = option_menu(
            menu_title="Navegação",
            options=["Visão Geral", "Análise de Preços", "Sentimento", "Correlações", "Performance"],
            icons=["house", "graph-up", "emoji-smile", "diagram-3", "trophy"],
            menu_icon="cast",
            default_index=0,
        )
        
        # Upload de dados
        st.subheader("📁 Upload de Dados")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=['csv'],
            help="Upload de dados financeiros em formato CSV"
        )
    
    # Inicializar dashboard
    dashboard = FinancialDashboard()
    
    # Dados de exemplo se não houver upload
    if uploaded_file is None:
        st.info("💡 Usando dados de exemplo. Faça upload de um arquivo CSV para usar seus próprios dados.")
        
        # Gerar dados de exemplo
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        
        sample_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'open': prices + np.random.randn(len(dates)) * 0.5,
            'high': prices + np.abs(np.random.randn(len(dates)) * 1.0),
            'low': prices - np.abs(np.random.randn(len(dates)) * 1.0)
        }).set_index('date')
        
        data = sample_data
    else:
        # Carregar dados do upload
        try:
            data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            st.success(f"✅ Dados carregados: {len(data)} registros")
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return
    
    # Conteúdo baseado na seleção
    if selected == "Visão Geral":
        st.header("📊 Visão Geral")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total de Registros",
                value=f"{len(data):,}",
                delta=None
            )
        
        with col2:
            if 'close' in data.columns:
                current_price = data['close'].iloc[-1]
                price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                st.metric(
                    label="Preço Atual",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:.2f}"
                )
        
        with col3:
            if 'close' in data.columns:
                total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
                st.metric(
                    label="Retorno Total",
                    value=f"{total_return:.2f}%",
                    delta=None
                )
        
        with col4:
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric(
                    label="Volatilidade Anual",
                    value=f"{volatility:.2f}%",
                    delta=None
                )
        
        # Gráfico principal
        st.subheader("📈 Evolução dos Preços")
        price_chart = dashboard.create_price_chart(data, title="Evolução dos Preços")
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Informações dos dados
        st.subheader("ℹ️ Informações dos Dados")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Período dos Dados:**")
            st.write(f"Início: {data.index.min().strftime('%Y-%m-%d')}")
            st.write(f"Fim: {data.index.max().strftime('%Y-%m-%d')}")
            st.write(f"Duração: {(data.index.max() - data.index.min()).days} dias")
        
        with col2:
            st.write("**Estatísticas:**")
            st.write(f"Colunas: {len(data.columns)}")
            st.write(f"Valores faltantes: {data.isnull().sum().sum()}")
            st.write(f"Tamanho: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    elif selected == "Análise de Preços":
        st.header("💹 Análise de Preços")
        
        # Opções de visualização
        chart_type = st.selectbox(
            "Tipo de Gráfico",
            ["Linha", "Candlestick", "Volatilidade"]
        )
        
        if chart_type == "Linha":
            chart = dashboard.create_price_chart(data, title="Análise de Preços - Linha")
        elif chart_type == "Candlestick":
            chart = dashboard.create_candlestick_chart(data, title="Análise de Preços - Candlestick")
        else:
            chart = dashboard.create_volatility_chart(data, title="Análise de Volatilidade")
        
        st.plotly_chart(chart, use_container_width=True)
        
        # Distribuição de retornos
        if 'close' in data.columns:
            st.subheader("📊 Distribuição de Retornos")
            returns_data = data.copy()
            returns_data['returns'] = data['close'].pct_change()
            
            dist_chart = dashboard.create_distribution_chart(
                returns_data, 'returns', "Distribuição de Retornos Diários"
            )
            st.plotly_chart(dist_chart, use_container_width=True)
    
    elif selected == "Sentimento":
        st.header("😊 Análise de Sentimento")
        st.info("💡 Esta seção mostraria análise de sentimento de notícias e eventos.")
        
        # Placeholder para dados de sentimento
        sentiment_data = pd.DataFrame({
            'sentiment_score': np.random.randn(len(data)) * 0.5,
            'date': data.index
        }).set_index('date')
        
        sentiment_chart = dashboard.create_sentiment_chart(
            sentiment_data, price_data=data, title="Análise de Sentimento vs Preços"
        )
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    elif selected == "Correlações":
        st.header("🔗 Análise de Correlações")
        
        # Selecionar colunas para correlação
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Selecione as variáveis para análise de correlação",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        )
        
        if selected_cols:
            corr_chart = dashboard.create_correlation_heatmap(
                data, selected_cols, "Matriz de Correlação"
            )
            st.plotly_chart(corr_chart, use_container_width=True)
    
    elif selected == "Performance":
        st.header("🏆 Análise de Performance")
        
        if 'close' in data.columns:
            perf_chart = dashboard.create_performance_metrics_chart(
                data, title="Métricas de Performance"
            )
            st.plotly_chart(perf_chart, use_container_width=True)
            
            # Métricas de performance
            st.subheader("📈 Métricas Detalhadas")
            
            returns = data['close'].pct_change().dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Retorno Médio Diário", f"{returns.mean():.4f}")
                st.metric("Volatilidade Diária", f"{returns.std():.4f}")
            
            with col2:
                st.metric("Sharpe Ratio", f"{returns.mean() / returns.std():.4f}")
                st.metric("Skewness", f"{returns.skew():.4f}")
            
            with col3:
                st.metric("Kurtosis", f"{returns.kurtosis():.4f}")
                st.metric("Max Drawdown", f"{((data['close'] / data['close'].cummax()) - 1).min():.4f}")

if __name__ == "__main__":
    # Para executar o dashboard Streamlit:
    # streamlit run dashboard.py
    create_streamlit_dashboard()