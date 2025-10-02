"""
Módulo de Análise de Eventos Mundiais
Projeto Big Data Finance - Análise de Impactos de Grandes Eventos

Desenvolvido por: Felipe Martins & Equipe Big Data Finance
Coleta de Dados: Felipe Martins
Infraestrutura: Ana Luiza Pazze
Análise Estatística: Pedro Silva
Machine Learning: Anny Caroline Sousa
Visualização: Ricardo Areas
Gestão: Fabio

Objetivo: Analisar o impacto de grandes eventos mundiais nos mercados financeiros
através de análises ANTES, DURANTE e DEPOIS dos eventos.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class EventsAnalyzer:
    """
    Analisador de impactos de eventos mundiais nos mercados financeiros
    """
    
    def __init__(self):
        self.major_events = {
            # 1. EVENTOS POLÍTICOS
            'eleicoes_eua_2016': {
                'name': 'Eleições Presidenciais EUA 2016',
                'category': 'Político',
                'date': '2016-11-08',
                'description': 'Eleição de Donald Trump como presidente dos EUA',
                'impact_period': 30,  # dias de análise antes/depois
                'severity': 'Alto'
            },
            'eleicoes_eua_2020': {
                'name': 'Eleições Presidenciais EUA 2020',
                'category': 'Político',
                'date': '2020-11-03',
                'description': 'Eleição de Joe Biden como presidente dos EUA',
                'impact_period': 30,
                'severity': 'Alto'
            },
            'eleicoes_eua_2024': {
                'name': 'Eleições Presidenciais EUA 2024',
                'category': 'Político',
                'date': '2024-11-05',
                'description': 'Eleições presidenciais americanas de 2024',
                'impact_period': 30,
                'severity': 'Alto'
            },
            'brexit_referendum': {
                'name': 'Referendo do Brexit',
                'category': 'Político',
                'date': '2016-06-23',
                'description': 'Reino Unido vota para sair da União Europeia',
                'impact_period': 45,
                'severity': 'Muito Alto'
            },
            'brexit_efetivacao': {
                'name': 'Brexit - Saída Efetiva da UE',
                'category': 'Político',
                'date': '2020-01-31',
                'description': 'Reino Unido oficialmente deixa a União Europeia',
                'impact_period': 30,
                'severity': 'Alto'
            },
            
            # 2. CONFLITOS GEOPOLÍTICOS
            'guerra_russia_ucrania': {
                'name': 'Invasão da Ucrânia pela Rússia',
                'category': 'Geopolítico',
                'date': '2022-02-24',
                'description': 'Início da guerra entre Rússia e Ucrânia',
                'impact_period': 60,
                'severity': 'Muito Alto'
            },
            'guerra_comercial_eua_china': {
                'name': 'Guerra Comercial EUA-China',
                'category': 'Geopolítico',
                'date': '2018-07-06',
                'description': 'Início das tarifas comerciais entre EUA e China',
                'impact_period': 45,
                'severity': 'Alto'
            },
            'tensao_oriente_medio_2023': {
                'name': 'Conflito Israel-Hamas 2023',
                'category': 'Geopolítico',
                'date': '2023-10-07',
                'description': 'Escalada do conflito no Oriente Médio',
                'impact_period': 30,
                'severity': 'Alto'
            },
            
            # 3. CATÁSTROFES E PANDEMIAS
            'covid19_pandemia': {
                'name': 'Declaração da Pandemia COVID-19',
                'category': 'Pandemia',
                'date': '2020-03-11',
                'description': 'OMS declara COVID-19 como pandemia global',
                'impact_period': 90,
                'severity': 'Extremo'
            },
            'covid19_lockdown_global': {
                'name': 'Lockdowns Globais COVID-19',
                'category': 'Pandemia',
                'date': '2020-03-20',
                'description': 'Início dos lockdowns em massa mundial',
                'impact_period': 120,
                'severity': 'Extremo'
            },
            'terremoto_japao_2011': {
                'name': 'Terremoto e Tsunami no Japão',
                'category': 'Catástrofe Natural',
                'date': '2011-03-11',
                'description': 'Grande terremoto e tsunami no Japão, acidente nuclear de Fukushima',
                'impact_period': 60,
                'severity': 'Muito Alto'
            },
            'terremoto_turquia_2023': {
                'name': 'Terremotos na Turquia e Síria',
                'category': 'Catástrofe Natural',
                'date': '2023-02-06',
                'description': 'Terremotos devastadores na Turquia e Síria',
                'impact_period': 30,
                'severity': 'Alto'
            },
            
            # 4. EVENTOS ECONÔMICOS
            'fed_rate_hike_2022': {
                'name': 'Início do Ciclo de Alta do Fed 2022',
                'category': 'Econômico',
                'date': '2022-03-16',
                'description': 'Federal Reserve inicia agressivo ciclo de alta de juros',
                'impact_period': 45,
                'severity': 'Alto'
            },
            'svb_collapse': {
                'name': 'Colapso do Silicon Valley Bank',
                'category': 'Econômico',
                'date': '2023-03-10',
                'description': 'Falência do SVB e crise bancária regional nos EUA',
                'impact_period': 30,
                'severity': 'Alto'
            },
            'credit_suisse_collapse': {
                'name': 'Colapso do Credit Suisse',
                'category': 'Econômico',
                'date': '2023-03-19',
                'description': 'Aquisição emergencial do Credit Suisse pelo UBS',
                'impact_period': 30,
                'severity': 'Alto'
            },
            'lehman_brothers': {
                'name': 'Falência do Lehman Brothers',
                'category': 'Econômico',
                'date': '2008-09-15',
                'description': 'Colapso do Lehman Brothers e início da crise financeira global',
                'impact_period': 90,
                'severity': 'Extremo'
            }
        }
        
        # Ativos para análise por categoria de evento
        self.event_assets = {
            'Político': ['SPY', 'VIX', 'GLD', 'TLT', 'DXY'],
            'Geopolítico': ['XLE', 'GLD', 'VIX', 'SPY', 'EWZ', 'FXI'],
            'Pandemia': ['SPY', 'QQQ', 'VIX', 'ZOOM', 'NFLX', 'AMZN'],
            'Catástrofe Natural': ['SPY', 'VIX', 'GLD', 'XLU', 'RE'],
            'Econômico': ['SPY', 'QQQ', 'XLF', 'TLT', 'VIX', 'GLD']
        }
    
    def get_available_events(self) -> Dict:
        """Retorna eventos disponíveis organizados por categoria"""
        events_by_category = {}
        for event_id, event_data in self.major_events.items():
            category = event_data['category']
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append({
                'id': event_id,
                'name': event_data['name'],
                'date': event_data['date'],
                'severity': event_data['severity']
            })
        return events_by_category
    
    def analyze_event_impact(self, event_id: str, custom_assets: List[str] = None) -> Dict:
        """
        Analisa o impacto de um evento específico nos mercados
        
        Args:
            event_id: ID do evento a ser analisado
            custom_assets: Lista customizada de ativos (opcional)
        """
        if event_id not in self.major_events:
            raise ValueError(f"Evento '{event_id}' não encontrado")
        
        event = self.major_events[event_id]
        event_date = pd.to_datetime(event['date'])
        impact_days = event['impact_period']
        
        # Definir períodos de análise (remover timezone para compatibilidade)
        event_date = event_date.tz_localize(None) if event_date.tz is not None else event_date
        
        # Período de análise: 60 dias antes, durante o evento, e 30 dias depois
        before_start = event_date - timedelta(days=60)  # 60 dias antes
        before_end = event_date - timedelta(days=1)
        during_start = event_date
        during_end = event_date + timedelta(days=min(impact_days, 15))  # Período "durante" baseado no impact_period
        after_start = during_end + timedelta(days=1)
        after_end = event_date + timedelta(days=30)  # 30 dias depois
        
        # Selecionar ativos para análise
        assets = custom_assets if custom_assets else self.event_assets.get(event['category'], ['SPY', 'VIX', 'GLD'])
        
        print(f"📊 Analisando evento: {event['name']}")
        print(f"📅 Data do evento: {event['date']}")
        print(f"🎯 Ativos analisados: {', '.join(assets)}")
        print(f"⏰ Período de análise: {impact_days} dias antes/depois")
        
        # Coletar dados
        extended_start = before_start - timedelta(days=10)  # Buffer para cálculos
        extended_end = after_end + timedelta(days=5)
        
        asset_data = {}
        for asset in assets:
            try:
                ticker = yf.Ticker(asset)
                hist = ticker.history(start=extended_start, end=extended_end)
                if not hist.empty:
                    asset_data[asset] = hist['Close']
                    print(f"✅ {asset}: {len(hist)} registros coletados")
                else:
                    print(f"⚠️ {asset}: Nenhum dado encontrado")
            except Exception as e:
                print(f"❌ Erro ao coletar {asset}: {str(e)}")
        
        if not asset_data:
            raise ValueError("Nenhum dado foi coletado para análise")
        
        # Criar DataFrame consolidado
        df = pd.DataFrame(asset_data).dropna()
        
        # Garantir que o índice não tenha timezone
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Análise por períodos
        analysis_results = {}
        
        for asset in df.columns:
            # Filtrar dados por período
            before_data = df[asset][(df.index >= before_start) & (df.index <= before_end)]
            during_data = df[asset][(df.index >= during_start) & (df.index <= during_end)]
            after_data = df[asset][(df.index >= after_start) & (df.index <= after_end)]
            
            if len(before_data) == 0 or len(after_data) == 0:
                continue
            
            # Calcular métricas
            before_return = (before_data.iloc[-1] / before_data.iloc[0] - 1) * 100
            during_return = (during_data.iloc[-1] / during_data.iloc[0] - 1) * 100 if len(during_data) > 0 else 0
            after_return = (after_data.iloc[-1] / after_data.iloc[0] - 1) * 100
            
            # Volatilidade (desvio padrão dos retornos diários)
            before_vol = before_data.pct_change().std() * np.sqrt(252) * 100
            during_vol = during_data.pct_change().std() * np.sqrt(252) * 100 if len(during_data) > 1 else 0
            after_vol = after_data.pct_change().std() * np.sqrt(252) * 100
            
            # Impacto do evento (comparação antes vs depois)
            event_impact = after_return - before_return
            volatility_change = after_vol - before_vol
            
            analysis_results[asset] = {
                'antes': {
                    'retorno': round(before_return, 2),
                    'volatilidade': round(before_vol, 2),
                    'preco_inicial': round(before_data.iloc[0], 2),
                    'preco_final': round(before_data.iloc[-1], 2)
                },
                'durante': {
                    'retorno': round(during_return, 2),
                    'volatilidade': round(during_vol, 2),
                    'preco_inicial': round(during_data.iloc[0], 2) if len(during_data) > 0 else 0,
                    'preco_final': round(during_data.iloc[-1], 2) if len(during_data) > 0 else 0
                },
                'depois': {
                    'retorno': round(after_return, 2),
                    'volatilidade': round(after_vol, 2),
                    'preco_inicial': round(after_data.iloc[0], 2),
                    'preco_final': round(after_data.iloc[-1], 2)
                },
                'impacto_evento': {
                    'mudanca_retorno': round(event_impact, 2),
                    'mudanca_volatilidade': round(volatility_change, 2),
                    'severidade_impacto': self._classify_impact(event_impact, volatility_change)
                }
            }
        
        # Filtrar DataFrame para retornar apenas o período relevante do evento
        df_filtered = df[(df.index >= before_start) & (df.index <= after_end)]
        
        # Debug: Verificar o range de datas
        print(f"🔍 DEBUG - Dados originais: {df.index.min()} a {df.index.max()} ({len(df)} registros)")
        print(f"🔍 DEBUG - Dados filtrados: {df_filtered.index.min()} a {df_filtered.index.max()} ({len(df_filtered)} registros)")
        print(f"🔍 DEBUG - Período esperado: {before_start} a {after_end}")
        
        # Resumo do evento
        event_summary = {
            'evento': event,
            'periodos': {
                'antes': f"{before_start.strftime('%Y-%m-%d')} a {before_end.strftime('%Y-%m-%d')}",
                'durante': f"{during_start.strftime('%Y-%m-%d')} a {during_end.strftime('%Y-%m-%d')}",
                'depois': f"{after_start.strftime('%Y-%m-%d')} a {after_end.strftime('%Y-%m-%d')}"
            },
            'ativos_analisados': len(analysis_results),
            'data_analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resultados_por_ativo': analysis_results,
            'dados_precos': df_filtered
        }
        
        return event_summary
    
    def _classify_impact(self, return_change: float, volatility_change: float) -> str:
        """Classifica a severidade do impacto do evento"""
        if abs(return_change) > 20 or abs(volatility_change) > 15:
            return "Extremo"
        elif abs(return_change) > 10 or abs(volatility_change) > 10:
            return "Alto"
        elif abs(return_change) > 5 or abs(volatility_change) > 5:
            return "Moderado"
        else:
            return "Baixo"
    
    def compare_multiple_events(self, event_ids: List[str], assets: List[str] = None) -> Dict:
        """
        Compara o impacto de múltiplos eventos
        """
        print(f"🔄 Comparando {len(event_ids)} eventos...")
        
        events_analysis = {}
        for event_id in event_ids:
            try:
                analysis = self.analyze_event_impact(event_id, assets)
                events_analysis[event_id] = analysis
                print(f"✅ Análise concluída para {analysis['evento']['name']}")
            except Exception as e:
                print(f"❌ Erro na análise de {event_id}: {str(e)}")
        
        # Comparação resumida
        comparison = {
            'data_comparacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eventos_analisados': len(events_analysis),
            'resumo_impactos': {},
            'ranking_impacto': [],
            'analises_detalhadas': events_analysis
        }
        
        # Calcular impactos médios por evento
        for event_id, analysis in events_analysis.items():
            impacts = []
            vol_changes = []
            
            for asset_data in analysis['resultados_por_ativo'].values():
                impacts.append(abs(asset_data['impacto_evento']['mudanca_retorno']))
                vol_changes.append(abs(asset_data['impacto_evento']['mudanca_volatilidade']))
            
            avg_impact = np.mean(impacts) if impacts else 0
            avg_vol_change = np.mean(vol_changes) if vol_changes else 0
            
            comparison['resumo_impactos'][event_id] = {
                'nome': analysis['evento']['name'],
                'categoria': analysis['evento']['category'],
                'data': analysis['evento']['date'],
                'impacto_medio_retorno': round(avg_impact, 2),
                'impacto_medio_volatilidade': round(avg_vol_change, 2),
                'severidade': analysis['evento']['severity']
            }
        
        # Ranking por impacto
        comparison['ranking_impacto'] = sorted(
            comparison['resumo_impactos'].items(),
            key=lambda x: x[1]['impacto_medio_retorno'],
            reverse=True
        )
        
        return comparison
    
    def generate_event_report(self, event_id: str) -> str:
        """
        Gera relatório textual detalhado de um evento
        """
        analysis = self.analyze_event_impact(event_id)
        
        report = f"""
🌍 RELATÓRIO DE IMPACTO DE EVENTO MUNDIAL
{'='*70}
📋 Evento: {analysis['evento']['name']}
📅 Data: {analysis['evento']['date']}
🏷️ Categoria: {analysis['evento']['category']}
⚠️ Severidade: {analysis['evento']['severity']}
📝 Descrição: {analysis['evento']['description']}

⏰ PERÍODOS DE ANÁLISE:
• Antes: {analysis['periodos']['antes']}
• Durante: {analysis['periodos']['durante']}
• Depois: {analysis['periodos']['depois']}

📊 IMPACTOS POR ATIVO:
"""
        
        for asset, data in analysis['resultados_por_ativo'].items():
            impact = data['impacto_evento']
            report += f"""
{asset}:
  📈 ANTES DO EVENTO:
    • Retorno: {data['antes']['retorno']}%
    • Volatilidade: {data['antes']['volatilidade']}%
    • Preço: ${data['antes']['preco_inicial']} → ${data['antes']['preco_final']}
  
  ⚡ DURANTE O EVENTO:
    • Retorno: {data['durante']['retorno']}%
    • Volatilidade: {data['durante']['volatilidade']}%
    • Preço: ${data['durante']['preco_inicial']} → ${data['durante']['preco_final']}
  
  📉 DEPOIS DO EVENTO:
    • Retorno: {data['depois']['retorno']}%
    • Volatilidade: {data['depois']['volatilidade']}%
    • Preço: ${data['depois']['preco_inicial']} → ${data['depois']['preco_final']}
  
  🎯 IMPACTO DO EVENTO:
    • Mudança no retorno: {impact['mudanca_retorno']}%
    • Mudança na volatilidade: {impact['mudanca_volatilidade']}%
    • Severidade do impacto: {impact['severidade_impacto']}
"""
        
        return report

# Exemplo de uso
if __name__ == "__main__":
    analyzer = EventsAnalyzer()
    
    # Listar eventos disponíveis
    print("🌍 Eventos mundiais disponíveis para análise:")
    events = analyzer.get_available_events()
    for category, event_list in events.items():
        print(f"\n📋 {category}:")
        for event in event_list:
            print(f"  • {event['name']} ({event['date']}) - Severidade: {event['severity']}")
    
    # Exemplo de análise
    try:
        print("\n" + "="*50)
        analysis = analyzer.analyze_event_impact('covid19_pandemia')
        print(analyzer.generate_event_report('covid19_pandemia'))
    except Exception as e:
        print(f"Erro na análise: {e}")