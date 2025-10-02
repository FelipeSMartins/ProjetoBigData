"""
Módulo de Visualização de Eventos Mundiais
Projeto Big Data Finance - Visualizações de Impactos de Grandes Eventos

Desenvolvido por: Ricardo Areas & Equipe Big Data Finance
Visualização: Ricardo Areas
Análise de Dados: Felipe Martins & Pedro Silva
Machine Learning: Anny Caroline Sousa
Infraestrutura: Ana Luiza Pazze
Gestão: Fabio

Objetivo: Criar visualizações claras e impactantes dos efeitos de eventos mundiais
nos mercados financeiros, com foco na comparação ANTES/DURANTE/DEPOIS.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class EventsCharts:
    """
    Gerador de gráficos especializados para análise de eventos mundiais
    """
    
    def __init__(self):
        # Cores por categoria de evento
        self.event_colors = {
            'Político': '#FF6B6B',      # Vermelho
            'Geopolítico': '#4ECDC4',   # Turquesa
            'Pandemia': '#45B7D1',      # Azul
            'Catástrofe Natural': '#96CEB4',  # Verde
            'Econômico': '#FFEAA7'      # Amarelo
        }
        
        # Cores para períodos
        self.period_colors = {
            'antes': '#95A5A6',         # Cinza
            'durante': '#E74C3C',       # Vermelho intenso
            'depois': '#3498DB'         # Azul
        }
        
        # Configurações padrão (sem title para evitar conflitos)
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'showlegend': True,
            'hovermode': 'x unified'
        }
    
    def create_event_timeline_chart(self, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de linha temporal mostrando preços antes/durante/depois do evento
        """
        evento = analysis_data['evento']
        df_precos = analysis_data['dados_precos']
        
        # Datas dos períodos
        event_date = pd.to_datetime(evento['date'])
        impact_days = evento['impact_period']
        
        before_start = event_date - timedelta(days=impact_days)
        during_end = event_date + timedelta(days=min(impact_days//3, 15))
        after_end = event_date + timedelta(days=impact_days)
        
        # Filtrar dados para mostrar apenas período relevante ao evento
        df_filtered = df_precos[(df_precos.index >= before_start) & (df_precos.index <= after_end)]
        
        fig = make_subplots(
            rows=len(df_filtered.columns), cols=1,
            subplot_titles=[f'{asset} - Evolução de Preços' for asset in df_filtered.columns],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        for i, asset in enumerate(df_filtered.columns, 1):
            prices = df_filtered[asset].dropna()
            
            # Normalizar preços para melhor visualização (base 100)
            if len(prices) > 0:
                base_price = prices.iloc[0]
                normalized_prices = (prices / base_price) * 100
            else:
                continue
            
            # Adicionar linha principal
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=normalized_prices,
                    mode='lines',
                    name=f'{asset}',
                    line=dict(width=2, color='#2C3E50'),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Destacar períodos com cores de fundo
            fig.add_vrect(
                x0=before_start, x1=event_date,
                fillcolor=self.period_colors['antes'], opacity=0.2,
                layer="below", line_width=0,
                annotation_text="ANTES", annotation_position="top left",
                row=i, col=1
            )
            
            fig.add_vrect(
                x0=event_date, x1=during_end,
                fillcolor=self.period_colors['durante'], opacity=0.3,
                layer="below", line_width=0,
                annotation_text="DURANTE", annotation_position="top",
                row=i, col=1
            )
            
            fig.add_vrect(
                x0=during_end, x1=after_end,
                fillcolor=self.period_colors['depois'], opacity=0.2,
                layer="below", line_width=0,
                annotation_text="DEPOIS", annotation_position="top right",
                row=i, col=1
            )
            
            # Linha vertical no dia do evento
            fig.add_vline(
                x=event_date,
                line=dict(color='red', width=3, dash='dash'),
                annotation_text=f"📅 {evento['name']}",
                annotation_position="top",
                row=i, col=1
            )
        
        fig.update_layout(
            title=f"📊 Evolução de Preços: {evento['name']}<br><sub>Análise Antes/Durante/Depois do Evento ({evento['date']})</sub>",
            height=300 * len(df_precos.columns),
            **self.default_layout
        )
        
        fig.update_yaxes(title_text="Preço Normalizado (Base 100)")
        fig.update_xaxes(title_text="Data", row=len(df_precos.columns), col=1)
        
        return fig
    
    def create_impact_comparison_chart(self, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de barras comparando retornos antes/durante/depois
        """
        resultados = analysis_data['resultados_por_ativo']
        
        assets = list(resultados.keys())
        antes_returns = [resultados[asset]['antes']['retorno'] for asset in assets]
        durante_returns = [resultados[asset]['durante']['retorno'] for asset in assets]
        depois_returns = [resultados[asset]['depois']['retorno'] for asset in assets]
        
        fig = go.Figure()
        
        # Barras para cada período
        fig.add_trace(go.Bar(
            name='Antes do Evento',
            x=assets,
            y=antes_returns,
            marker_color=self.period_colors['antes'],
            text=[f'{ret:.1f}%' for ret in antes_returns],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Durante o Evento',
            x=assets,
            y=durante_returns,
            marker_color=self.period_colors['durante'],
            text=[f'{ret:.1f}%' for ret in durante_returns],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Depois do Evento',
            x=assets,
            y=depois_returns,
            marker_color=self.period_colors['depois'],
            text=[f'{ret:.1f}%' for ret in depois_returns],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"📊 Comparação de Retornos: {analysis_data['evento']['name']}<br><sub>Impacto nos Diferentes Períodos</sub>",
            xaxis_title="Ativos",
            yaxis_title="Retorno (%)",
            barmode='group',
            **self.default_layout
        )
        
        # Linha horizontal no zero
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def create_volatility_impact_chart(self, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de volatilidade antes/durante/depois do evento
        """
        resultados = analysis_data['resultados_por_ativo']
        
        assets = list(resultados.keys())
        antes_vol = [resultados[asset]['antes']['volatilidade'] for asset in assets]
        durante_vol = [resultados[asset]['durante']['volatilidade'] for asset in assets]
        depois_vol = [resultados[asset]['depois']['volatilidade'] for asset in assets]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=assets,
            y=antes_vol,
            mode='markers+lines',
            name='Antes do Evento',
            marker=dict(size=10, color=self.period_colors['antes']),
            line=dict(width=2, color=self.period_colors['antes'])
        ))
        
        fig.add_trace(go.Scatter(
            x=assets,
            y=durante_vol,
            mode='markers+lines',
            name='Durante o Evento',
            marker=dict(size=12, color=self.period_colors['durante']),
            line=dict(width=3, color=self.period_colors['durante'])
        ))
        
        fig.add_trace(go.Scatter(
            x=assets,
            y=depois_vol,
            mode='markers+lines',
            name='Depois do Evento',
            marker=dict(size=10, color=self.period_colors['depois']),
            line=dict(width=2, color=self.period_colors['depois'])
        ))
        
        fig.update_layout(
            title=f"📈 Evolução da Volatilidade: {analysis_data['evento']['name']}<br><sub>Mudanças na Volatilidade dos Ativos</sub>",
            xaxis_title="Ativos",
            yaxis_title="Volatilidade Anualizada (%)",
            **self.default_layout
        )
        
        return fig
    
    def create_event_impact_heatmap(self, analysis_data: Dict) -> go.Figure:
        """
        Cria heatmap mostrando intensidade dos impactos
        """
        resultados = analysis_data['resultados_por_ativo']
        
        assets = list(resultados.keys())
        periods = ['Antes', 'Durante', 'Depois']
        
        # Matriz de retornos
        returns_matrix = []
        for period in ['antes', 'durante', 'depois']:
            period_returns = [resultados[asset][period]['retorno'] for asset in assets]
            returns_matrix.append(period_returns)
        
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix,
            x=assets,
            y=periods,
            colorscale='RdYlBu_r',
            text=[[f'{val:.1f}%' for val in row] for row in returns_matrix],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Retorno (%)")
        ))
        
        fig.update_layout(
            title=f"🔥 Mapa de Calor dos Impactos: {analysis_data['evento']['name']}<br><sub>Intensidade dos Retornos por Período</sub>",
            xaxis_title="Ativos",
            yaxis_title="Períodos",
            **self.default_layout
        )
        
        return fig
    
    def create_multiple_events_comparison(self, comparison_data: Dict) -> go.Figure:
        """
        Cria gráfico comparando múltiplos eventos
        """
        resumo = comparison_data['resumo_impactos']
        
        events = list(resumo.keys())
        event_names = [resumo[event]['nome'] for event in events]
        impact_returns = [resumo[event]['impacto_medio_retorno'] for event in events]
        impact_vols = [resumo[event]['impacto_medio_volatilidade'] for event in events]
        categories = [resumo[event]['categoria'] for event in events]
        
        # Cores por categoria
        colors = [self.event_colors.get(cat, '#95A5A6') for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=impact_returns,
            y=impact_vols,
            mode='markers+text',
            text=event_names,
            textposition='top center',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Impacto Retorno: %{x:.1f}%<br>' +
                         'Impacto Volatilidade: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="🌍 Comparação de Impactos de Eventos Mundiais<br><sub>Intensidade dos Efeitos nos Mercados Financeiros</sub>",
            xaxis_title="Impacto Médio no Retorno (%)",
            yaxis_title="Impacto Médio na Volatilidade (%)",
            **self.default_layout
        )
        
        # Linhas de referência
        fig.add_hline(y=np.mean(impact_vols), line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=np.mean(impact_returns), line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_individual_event_dashboard(self, analysis_data: Dict) -> go.Figure:
        """
        Cria dashboard individual para um evento específico com múltiplas visualizações
        """
        evento = analysis_data['evento']
        df_precos = analysis_data['dados_precos']
        resultados = analysis_data['resultados_por_ativo']
        
        # Criar subplots: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"📈 Evolução de Preços - {evento['name']}",
                f"📊 Impacto por Ativo - {evento['name']}",
                f"📉 Volatilidade por Período",
                f"🎯 Ranking de Impactos"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Gráfico de linha - Evolução de preços
        event_date = pd.to_datetime(evento['date'])
        colors = px.colors.qualitative.Set1
        
        for i, asset in enumerate(df_precos.columns):
            prices = df_precos[asset].dropna()
            if len(prices) > 0:
                base_price = prices.iloc[0]
                normalized_prices = (prices / base_price) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=normalized_prices,
                        mode='lines+markers',
                        name=asset,
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{asset}</b><br>Data: %{{x}}<br>Preço: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Linha vertical do evento
        fig.add_vline(
            x=event_date,
            line=dict(color='red', width=3, dash='dash'),
            annotation_text=f"📅 {evento['name']}",
            row=1, col=1
        )
        
        # 2. Gráfico de barras - Impacto por ativo
        assets = list(resultados.keys())
        antes_returns = [resultados[asset]['antes']['retorno'] * 100 for asset in assets]
        durante_returns = [resultados[asset]['durante']['retorno'] * 100 for asset in assets]
        depois_returns = [resultados[asset]['depois']['retorno'] * 100 for asset in assets]
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=antes_returns,
                name='Antes',
                marker_color=self.period_colors['antes'],
                hovertemplate='<b>%{x}</b><br>Retorno Antes: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=durante_returns,
                name='Durante',
                marker_color=self.period_colors['durante'],
                hovertemplate='<b>%{x}</b><br>Retorno Durante: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=depois_returns,
                name='Depois',
                marker_color=self.period_colors['depois'],
                hovertemplate='<b>%{x}</b><br>Retorno Depois: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Gráfico de volatilidade
        volatilidades = []
        periodos = []
        ativos_vol = []
        
        for asset in assets:
            for periodo in ['antes', 'durante', 'depois']:
                volatilidades.append(resultados[asset][periodo]['volatilidade'] * 100)
                periodos.append(periodo.title())
                ativos_vol.append(asset)
        
        fig.add_trace(
            go.Scatter(
                x=ativos_vol,
                y=volatilidades,
                mode='markers',
                marker=dict(
                    size=12,
                    color=[self.period_colors[p.lower()] for p in periodos],
                    line=dict(width=2, color='white')
                ),
                text=periodos,
                hovertemplate='<b>%{x}</b><br>Período: %{text}<br>Volatilidade: %{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Ranking de impactos (impacto total)
        impactos_totais = []
        for asset in assets:
            impacto_total = abs(resultados[asset]['durante']['retorno']) * 100
            impactos_totais.append(impacto_total)
        
        # Ordenar por impacto
        sorted_data = sorted(zip(assets, impactos_totais), key=lambda x: x[1], reverse=True)
        sorted_assets, sorted_impacts = zip(*sorted_data)
        
        colors_ranking = ['#E74C3C' if impact > 5 else '#F39C12' if impact > 2 else '#27AE60' 
                         for impact in sorted_impacts]
        
        fig.add_trace(
            go.Bar(
                x=list(sorted_assets),
                y=list(sorted_impacts),
                marker_color=colors_ranking,
                hovertemplate='<b>%{x}</b><br>Impacto: %{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Atualizar layouts
        fig.update_xaxes(title_text="Data", row=1, col=1)
        fig.update_yaxes(title_text="Preço (Base 100)", row=1, col=1)
        
        fig.update_xaxes(title_text="Ativos", row=1, col=2)
        fig.update_yaxes(title_text="Retorno (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Ativos", row=2, col=1)
        fig.update_yaxes(title_text="Volatilidade (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Ativos", row=2, col=2)
        fig.update_yaxes(title_text="Impacto Absoluto (%)", row=2, col=2)
        
        fig.update_layout(
            title=f"🌍 Dashboard Completo: {evento['name']}<br><sub>📅 {evento['date']} | 📊 Análise Detalhada de Impactos</sub>",
            height=800,
            **self.default_layout
        )
        
        return fig

    def create_events_selector_dashboard(self, events_data: Dict) -> go.Figure:
        """
        Cria dashboard com seletor de eventos para comparação interativa
        """
        # Criar figura com dropdown para seleção de eventos
        fig = go.Figure()
        
        # Lista de eventos disponíveis
        event_names = list(events_data.keys())
        
        # Criar traces para cada evento (inicialmente invisível)
        for i, event_name in enumerate(event_names):
            event_data = events_data[event_name]
            df_precos = event_data['dados_precos']
            evento = event_data['evento']
            
            # Debug: Verificar dados recebidos
            print(f"🔍 DEBUG DASHBOARD - {event_name}:")
            print(f"   Range de dados: {df_precos.index.min()} a {df_precos.index.max()}")
            print(f"   Total de registros: {len(df_precos)}")
            
            # Filtrar dados para mostrar apenas período relevante ao evento (60 dias antes, durante e 30 dias depois)
            event_date = pd.to_datetime(evento['date'])
            impact_days = evento['impact_period']
            start_date = event_date - timedelta(days=60)  # 60 dias antes
            end_date = event_date + timedelta(days=30)    # 30 dias depois
            df_filtered = df_precos[(df_precos.index >= start_date) & (df_precos.index <= end_date)]
            
            print(f"   Após filtro adicional: {df_filtered.index.min()} a {df_filtered.index.max()} ({len(df_filtered)} registros)")
            
            # Adicionar traces para cada ativo do evento
            for j, asset in enumerate(df_filtered.columns):
                prices = df_filtered[asset].dropna()
                if len(prices) > 0:
                    base_price = prices.iloc[0]
                    normalized_prices = (prices / base_price) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=prices.index,
                            y=normalized_prices,
                            mode='lines+markers',
                            name=f'{asset}',
                            line=dict(width=3),
                            marker=dict(size=4),
                            visible=(i == 0),  # Apenas o primeiro evento visível inicialmente
                            hovertemplate=f'<b>{asset}</b><br>Data: %{{x}}<br>Preço: %{{y:.2f}}<extra></extra>'
                        )
                    )
        
        # Criar botões dropdown para seleção de eventos
        dropdown_buttons = []
        
        for i, event_name in enumerate(event_names):
            event_data = events_data[event_name]
            num_assets = len(event_data['dados_precos'].columns)
            
            # Calcular range de datas para este evento (60 dias antes e 30 dias depois)
            event_date = pd.to_datetime(event_data['evento']['date'])
            impact_days = event_data['evento']['impact_period']
            start_date = event_date - timedelta(days=60)  # 60 dias antes
            end_date = event_date + timedelta(days=30)    # 30 dias depois
            
            # Criar lista de visibilidade para este evento
            visibility = [False] * len(fig.data)
            start_idx = sum(len(events_data[event_names[j]]['dados_precos'].columns) 
                          for j in range(i))
            end_idx = start_idx + num_assets
            
            for idx in range(start_idx, end_idx):
                if idx < len(visibility):
                    visibility[idx] = True
            
            dropdown_buttons.append(
                dict(
                    label=f"🌍 {event_name}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": f"📊 Análise de Evento: {event_name}<br><sub>📅 {event_data['evento']['date']} | Selecione outros eventos no menu</sub>",
                            "xaxis.range": [start_date, end_date]
                        }
                    ]
                )
            )
        
        # Configurar range inicial do eixo X para o primeiro evento
        first_event_data = events_data[event_names[0]]
        first_event_date = pd.to_datetime(first_event_data['evento']['date'])
        first_impact_days = first_event_data['evento']['impact_period']
        first_start_date = first_event_date - timedelta(days=first_impact_days)
        first_end_date = first_event_date + timedelta(days=first_impact_days)
        
        # Configurar layout com dropdown
        fig.update_layout(
            title=f"📊 Análise de Evento: {event_names[0]}<br><sub>📅 {events_data[event_names[0]]['evento']['date']} | Selecione outros eventos no menu</sub>",
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                )
            ],
            annotations=[
                dict(
                    text="Selecionar Evento:",
                    x=0.02, xanchor="left",
                    y=1.18, yanchor="bottom",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )
            ],
            height=600,
            **self.default_layout
        )
        
        fig.update_xaxes(
            title_text="Data",
            range=[first_start_date, first_end_date]
        )
        fig.update_yaxes(title_text="Preço Normalizado (Base 100)")
        
        return fig

    def create_simple_line_chart_by_event(self, event_name: str, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de linha simples para um evento específico
        """
        evento = analysis_data['evento']
        df_precos = analysis_data['dados_precos']
        
        # Filtrar dados para mostrar apenas período relevante ao evento
        event_date = pd.to_datetime(evento['date'])
        impact_days = evento['impact_period']
        
        # Definir janela de visualização (período antes/durante/depois do evento)
        start_date = event_date - timedelta(days=impact_days)
        end_date = event_date + timedelta(days=impact_days)
        
        # Filtrar DataFrame para o período relevante
        df_filtered = df_precos[(df_precos.index >= start_date) & (df_precos.index <= end_date)]
        
        fig = go.Figure()
        
        # Cores para diferentes ativos
        colors = px.colors.qualitative.Set1
        
        # Adicionar linha para cada ativo
        for i, asset in enumerate(df_filtered.columns):
            prices = df_filtered[asset].dropna()
            if len(prices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices,
                        mode='lines+markers',
                        name=asset,
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{asset}</b><br>Data: %{{x}}<br>Preço: $%{{y:.2f}}<extra></extra>'
                    )
                )
        
        # Linha vertical do evento
        event_date = pd.to_datetime(evento['date'])
        fig.add_vline(
            x=event_date,
            line=dict(color='red', width=3, dash='dash'),
            annotation_text=f"📅 {evento['name']}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=f"📈 {event_name}<br><sub>📅 {evento['date']} | Evolução dos Preços dos Ativos</sub>",
            xaxis_title="Data",
            yaxis_title="Preço ($)",
            height=500,
            **self.default_layout
        )
        
        return fig

    def create_simple_bar_chart_by_event(self, event_name: str, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de barras simples para impactos de um evento específico
        """
        evento = analysis_data['evento']
        resultados = analysis_data['resultados_por_ativo']
        
        fig = go.Figure()
        
        # Preparar dados
        assets = list(resultados.keys())
        impactos = [resultados[asset]['durante']['retorno'] * 100 for asset in assets]
        
        # Cores baseadas no impacto (positivo/negativo)
        colors = ['#27AE60' if imp > 0 else '#E74C3C' for imp in impactos]
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=impactos,
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Impacto: %{y:.2f}%<extra></extra>',
                text=[f'{imp:+.1f}%' for imp in impactos],
                textposition='outside'
            )
        )
        
        # Linha horizontal no zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f"📊 Impacto do Evento: {event_name}<br><sub>📅 {evento['date']} | Retorno Durante o Evento (%)</sub>",
            xaxis_title="Ativos",
            yaxis_title="Retorno (%)",
            height=400,
            showlegend=False,
            **self.default_layout
        )
        
        return fig

    def create_simple_bar_chart_by_event(self, event_name: str, analysis_data: Dict) -> go.Figure:
        """
        Cria gráfico de barras simples para impactos de um evento específico
        """
        evento = analysis_data['evento']
        resultados = analysis_data['resultados_por_ativo']
        
        fig = go.Figure()
        
        # Preparar dados
        assets = list(resultados.keys())
        impactos = [resultados[asset]['durante']['retorno'] * 100 for asset in assets]
        
        # Cores baseadas no impacto (positivo/negativo)
        colors = ['#27AE60' if imp > 0 else '#E74C3C' for imp in impactos]
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=impactos,
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Impacto: %{y:.2f}%<extra></extra>',
                text=[f'{imp:+.1f}%' for imp in impactos],
                textposition='outside'
            )
        )
        
        # Linha horizontal no zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f"📊 Impacto do Evento: {event_name}<br><sub>📅 {evento['date']} | Retorno Durante o Evento (%)</sub>",
            xaxis_title="Ativos",
            yaxis_title="Retorno (%)",
            height=400,
            showlegend=False,
            **self.default_layout
        )
        
        return fig

    def save_all_charts(self, analysis_data: Dict, output_dir: str = ".") -> List[str]:
        """
        Salva todos os gráficos de um evento em arquivos HTML
        """
        evento_name = analysis_data['evento']['name'].replace(' ', '_').replace('/', '_')
        saved_files = []
        
        charts = {
            'timeline': self.create_event_timeline_chart(analysis_data),
            'impact_comparison': self.create_impact_comparison_chart(analysis_data),
            'volatility': self.create_volatility_impact_chart(analysis_data),
            'heatmap': self.create_event_impact_heatmap(analysis_data),
            'dashboard': self.create_event_severity_dashboard(analysis_data)
        }
        
        for chart_name, fig in charts.items():
            filename = f"{output_dir}/evento_{evento_name}_{chart_name}.html"
            fig.write_html(filename)
            saved_files.append(filename)
            print(f"✅ Gráfico salvo: {filename}")
        
        return saved_files

# Exemplo de uso
if __name__ == "__main__":
    from events_analyzer import EventsAnalyzer
    
    # Criar instâncias
    analyzer = EventsAnalyzer()
    charts = EventsCharts()
    
    # Analisar um evento
    try:
        analysis = analyzer.analyze_event_impact('covid19_pandemia')
        
        # Criar visualizações
        timeline_chart = charts.create_event_timeline_chart(analysis)
        impact_chart = charts.create_impact_comparison_chart(analysis)
        dashboard = charts.create_event_severity_dashboard(analysis)
        
        # Salvar gráficos
        charts.save_all_charts(analysis)
        
        print("✅ Visualizações criadas com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")