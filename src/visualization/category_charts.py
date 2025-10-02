"""
Módulo de Visualização por Categorias
Desenvolvido por: Ricardo Areas & Equipe Big Data Finance
Visualização: Ricardo Areas
Infraestrutura: Ana Luiza Pazze
Gestão: Fabio
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import json

class CategoryCharts:
    """
    Gerador de gráficos e visualizações para análise por categorias
    """
    
    def __init__(self):
        self.colors = {
            'petroleo': '#FF6B35',
            'tecnologia': '#004E89',
            'agronegocio': '#2E8B57',
            'fiis': '#8B4513',
            'bancos': '#FFD700',
            'varejo': '#9932CC'
        }
    
    def create_category_performance_chart(self, analysis: Dict, save_path: str = None) -> go.Figure:
        """
        Cria gráfico de performance de uma categoria
        """
        metrics = analysis['metricas_individuais']
        
        symbols = list(metrics.keys())
        returns = [metrics[s]['retorno_total'] for s in symbols]
        volatilities = [metrics[s]['volatilidade'] for s in symbols]
        
        fig = go.Figure()
        
        # Gráfico de dispersão Retorno vs Volatilidade
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=symbols,
            textposition='top center',
            marker=dict(
                size=15,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Retorno (%)")
            ),
            name='Ativos'
        ))
        
        fig.update_layout(
            title=f"Performance da Categoria: {analysis['categoria']}",
            xaxis_title="Volatilidade (%)",
            yaxis_title="Retorno Total (%)",
            template="plotly_white",
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_category_comparison_chart(self, comparison: Dict, save_path: str = None) -> go.Figure:
        """
        Cria gráfico comparativo entre categorias
        """
        categories = []
        returns = []
        volatilities = []
        
        for cat, data in comparison['resumo_categorias'].items():
            categories.append(data['nome'])
            returns.append(data['retorno_medio'])
            volatilities.append(data['volatilidade_media'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Retorno por Categoria', 'Volatilidade por Categoria'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico de retornos
        fig.add_trace(
            go.Bar(
                x=categories,
                y=returns,
                name='Retorno (%)',
                marker_color=[self.colors.get(cat, '#1f77b4') for cat in comparison['resumo_categorias'].keys()]
            ),
            row=1, col=1
        )
        
        # Gráfico de volatilidade
        fig.add_trace(
            go.Bar(
                x=categories,
                y=volatilities,
                name='Volatilidade (%)',
                marker_color=[self.colors.get(cat, '#ff7f0e') for cat in comparison['resumo_categorias'].keys()]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparação entre Categorias de Ativos",
            template="plotly_white",
            width=1200,
            height=500,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_category_correlation_heatmap(self, analysis: Dict, save_path: str = None) -> go.Figure:
        """
        Cria mapa de calor de correlação para uma categoria
        """
        correlation_data = analysis['correlacao']
        symbols = list(correlation_data.keys())
        
        # Converter para matriz
        correlation_matrix = []
        for symbol1 in symbols:
            row = []
            for symbol2 in symbols:
                row.append(correlation_data[symbol1][symbol2])
            correlation_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlação")
        ))
        
        fig.update_layout(
            title=f"Matriz de Correlação - {analysis['categoria']}",
            template="plotly_white",
            width=600,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_price_evolution_chart(self, analysis: Dict, save_path: str = None) -> go.Figure:
        """
        Cria gráfico de evolução de preços da categoria
        """
        df = analysis['dados_precos']
        
        fig = go.Figure()
        
        for symbol in df.columns:
            # Normalizar preços para base 100
            normalized_prices = (df[symbol] / df[symbol].iloc[0]) * 100
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_prices,
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Evolução de Preços - {analysis['categoria']} (Base 100)",
            xaxis_title="Data",
            yaxis_title="Preço Normalizado (Base 100)",
            template="plotly_white",
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_risk_return_dashboard(self, comparison: Dict, save_path: str = None) -> go.Figure:
        """
        Cria dashboard completo de risco vs retorno
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Retorno por Categoria',
                'Volatilidade por Categoria', 
                'Risco vs Retorno',
                'Ranking de Performance'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        categories = []
        returns = []
        volatilities = []
        names = []
        
        for cat, data in comparison['resumo_categorias'].items():
            categories.append(cat)
            names.append(data['nome'])
            returns.append(data['retorno_medio'])
            volatilities.append(data['volatilidade_media'])
        
        # Gráfico 1: Retornos
        fig.add_trace(
            go.Bar(x=names, y=returns, name='Retorno (%)', 
                   marker_color=[self.colors.get(cat, '#1f77b4') for cat in categories]),
            row=1, col=1
        )
        
        # Gráfico 2: Volatilidade
        fig.add_trace(
            go.Bar(x=names, y=volatilities, name='Volatilidade (%)',
                   marker_color=[self.colors.get(cat, '#ff7f0e') for cat in categories]),
            row=1, col=2
        )
        
        # Gráfico 3: Risco vs Retorno
        fig.add_trace(
            go.Scatter(
                x=volatilities, y=returns, mode='markers+text',
                text=names, textposition='top center',
                marker=dict(size=15, color=[self.colors.get(cat, '#2ca02c') for cat in categories]),
                name='Categorias'
            ),
            row=2, col=1
        )
        
        # Gráfico 4: Ranking
        sorted_data = sorted(zip(names, returns), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_returns = zip(*sorted_data)
        
        fig.add_trace(
            go.Bar(x=list(sorted_names), y=list(sorted_returns), name='Ranking',
                   marker_color='lightblue'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Dashboard de Análise por Categorias",
            template="plotly_white",
            width=1400,
            height=800,
            showlegend=False
        )
        
        # Atualizar eixos
        fig.update_xaxes(title_text="Categorias", row=1, col=1)
        fig.update_xaxes(title_text="Categorias", row=1, col=2)
        fig.update_xaxes(title_text="Volatilidade (%)", row=2, col=1)
        fig.update_xaxes(title_text="Categorias", row=2, col=2)
        
        fig.update_yaxes(title_text="Retorno (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatilidade (%)", row=1, col=2)
        fig.update_yaxes(title_text="Retorno (%)", row=2, col=1)
        fig.update_yaxes(title_text="Retorno (%)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_sector_summary_table(self, comparison: Dict, save_path: str = None) -> go.Figure:
        """
        Cria tabela resumo dos setores
        """
        data = []
        for cat, info in comparison['resumo_categorias'].items():
            data.append([
                info['nome'],
                f"{info['retorno_medio']:.2f}%",
                f"{info['volatilidade_media']:.2f}%",
                info['melhor_ativo'],
                info['pior_ativo']
            ])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Categoria', 'Retorno Médio', 'Volatilidade', 'Melhor Ativo', 'Pior Ativo'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=list(zip(*data)),
                fill_color='white',
                align='center',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Resumo por Categoria de Ativos",
            width=800,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

# Exemplo de uso
if __name__ == "__main__":
    # Este arquivo será usado em conjunto com o CategoryAnalyzer
    print("📊 Módulo de visualização por categorias carregado!")
    print("🎨 Cores disponíveis por categoria:")
    charts = CategoryCharts()
    for cat, color in charts.colors.items():
        print(f"  • {cat}: {color}")