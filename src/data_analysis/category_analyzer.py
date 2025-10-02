"""
Módulo de Análise por Categorias de Ativos
Desenvolvido por: Felipe Martins & Equipe Big Data Finance
Responsável por Categorização: Felipe Martins
Infraestrutura: Ana Luiza Pazze
Gestão: Fabio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta

class CategoryAnalyzer:
    """
    Analisador de ativos por categorias específicas do mercado brasileiro e internacional
    """
    
    def __init__(self):
        self.categories = {
            'petroleo': {
                'name': 'Petróleo e Energia',
                'symbols': ['PETR4.SA', 'VALE3.SA', 'PRIO3.SA', 'XOM', 'CVX', 'BP'],
                'description': 'Empresas do setor de petróleo, gás e energia'
            },
            'tecnologia': {
                'name': 'Tecnologia',
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'MGLU3.SA', 'B3SA3.SA'],
                'description': 'Empresas de tecnologia e inovação'
            },
            'agronegocio': {
                'name': 'Agronegócio',
                'symbols': ['JBS.SA', 'BEEF3.SA', 'MRFG3.SA', 'SLCE3.SA', 'ADM', 'CTVA'],
                'description': 'Empresas do setor agrícola e alimentício'
            },
            'fiis': {
                'name': 'Fundos Imobiliários',
                'symbols': ['HGLG11.SA', 'XPML11.SA', 'KNRI11.SA', 'MXRF11.SA', 'BCFF11.SA'],
                'description': 'Fundos de Investimento Imobiliário'
            },
            'bancos': {
                'name': 'Setor Bancário',
                'symbols': ['ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'SANB11.SA', 'JPM', 'BAC'],
                'description': 'Instituições financeiras e bancos'
            },
            'varejo': {
                'name': 'Varejo e Consumo',
                'symbols': ['MGLU3.SA', 'LREN3.SA', 'AMER3.SA', 'AMZN', 'WMT', 'TGT'],
                'description': 'Empresas de varejo e bens de consumo'
            }
        }
        
    def get_available_categories(self) -> Dict:
        """Retorna as categorias disponíveis"""
        return {k: v['name'] for k, v in self.categories.items()}
    
    def get_category_symbols(self, category: str) -> List[str]:
        """Retorna os símbolos de uma categoria específica"""
        if category not in self.categories:
            raise ValueError(f"Categoria '{category}' não encontrada. Disponíveis: {list(self.categories.keys())}")
        return self.categories[category]['symbols']
    
    def collect_category_data(self, category: str, period: str = "1y") -> pd.DataFrame:
        """
        Coleta dados de uma categoria específica
        
        Args:
            category: Nome da categoria
            period: Período dos dados (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        symbols = self.get_category_symbols(category)
        
        print(f"📊 Coletando dados da categoria: {self.categories[category]['name']}")
        print(f"🎯 Ativos: {', '.join(symbols)}")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
                    print(f"✅ {symbol}: {len(hist)} registros coletados")
                else:
                    print(f"⚠️ {symbol}: Nenhum dado encontrado")
            except Exception as e:
                print(f"❌ Erro ao coletar {symbol}: {str(e)}")
        
        if not data:
            raise ValueError(f"Nenhum dado foi coletado para a categoria {category}")
        
        df = pd.DataFrame(data)
        df.index.name = 'Date'
        return df.dropna()
    
    def analyze_category_performance(self, category: str, period: str = "1y") -> Dict:
        """
        Analisa a performance de uma categoria
        """
        df = self.collect_category_data(category, period)
        
        # Calcular retornos
        returns = df.pct_change().dropna()
        
        # Métricas por ativo
        metrics = {}
        for symbol in df.columns:
            start_price = df[symbol].iloc[0]
            end_price = df[symbol].iloc[-1]
            total_return = (end_price / start_price - 1) * 100
            volatility = returns[symbol].std() * np.sqrt(252) * 100  # Anualizada
            
            metrics[symbol] = {
                'preco_inicial': round(start_price, 2),
                'preco_atual': round(end_price, 2),
                'retorno_total': round(total_return, 2),
                'volatilidade': round(volatility, 2),
                'sharpe_ratio': round(total_return / volatility if volatility > 0 else 0, 3)
            }
        
        # Métricas da categoria
        category_returns = returns.mean(axis=1)  # Média dos retornos diários
        category_total_return = (1 + category_returns).prod() - 1
        category_volatility = category_returns.std() * np.sqrt(252)
        
        # Correlação entre ativos
        correlation_matrix = returns.corr()
        
        analysis = {
            'categoria': self.categories[category]['name'],
            'periodo': period,
            'data_analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ativos_analisados': len(df.columns),
            'metricas_individuais': metrics,
            'metricas_categoria': {
                'retorno_medio': round(category_total_return * 100, 2),
                'volatilidade_media': round(category_volatility * 100, 2),
                'melhor_ativo': max(metrics.keys(), key=lambda x: metrics[x]['retorno_total']),
                'pior_ativo': min(metrics.keys(), key=lambda x: metrics[x]['retorno_total']),
                'ativo_menos_volatil': min(metrics.keys(), key=lambda x: metrics[x]['volatilidade']),
                'ativo_mais_volatil': max(metrics.keys(), key=lambda x: metrics[x]['volatilidade'])
            },
            'correlacao': correlation_matrix.round(3).to_dict(),
            'dados_precos': df
        }
        
        return analysis
    
    def compare_categories(self, categories: List[str], period: str = "1y") -> Dict:
        """
        Compara performance entre diferentes categorias
        """
        print(f"🔄 Comparando categorias: {', '.join(categories)}")
        
        category_analyses = {}
        for category in categories:
            try:
                analysis = self.analyze_category_performance(category, period)
                category_analyses[category] = analysis
                print(f"✅ Análise concluída para {self.categories[category]['name']}")
            except Exception as e:
                print(f"❌ Erro na análise de {category}: {str(e)}")
        
        # Comparação resumida
        comparison = {
            'periodo': period,
            'data_comparacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categorias_analisadas': len(category_analyses),
            'resumo_categorias': {},
            'ranking_retorno': [],
            'ranking_volatilidade': [],
            'analises_detalhadas': category_analyses
        }
        
        # Resumo por categoria
        for cat, analysis in category_analyses.items():
            cat_name = self.categories[cat]['name']
            metrics = analysis['metricas_categoria']
            
            comparison['resumo_categorias'][cat] = {
                'nome': cat_name,
                'retorno_medio': metrics['retorno_medio'],
                'volatilidade_media': metrics['volatilidade_media'],
                'melhor_ativo': metrics['melhor_ativo'],
                'pior_ativo': metrics['pior_ativo']
            }
        
        # Rankings
        comparison['ranking_retorno'] = sorted(
            comparison['resumo_categorias'].items(),
            key=lambda x: x[1]['retorno_medio'],
            reverse=True
        )
        
        comparison['ranking_volatilidade'] = sorted(
            comparison['resumo_categorias'].items(),
            key=lambda x: x[1]['volatilidade_media']
        )
        
        return comparison
    
    def generate_category_report(self, category: str, period: str = "1y") -> str:
        """
        Gera relatório textual para uma categoria
        """
        analysis = self.analyze_category_performance(category, period)
        
        report = f"""
🏷️ RELATÓRIO DE CATEGORIA: {analysis['categoria'].upper()}
{'='*60}
📅 Período: {period}
📊 Data da análise: {analysis['data_analise']}
🎯 Ativos analisados: {analysis['ativos_analisados']}

📈 PERFORMANCE DA CATEGORIA:
• Retorno médio: {analysis['metricas_categoria']['retorno_medio']}%
• Volatilidade média: {analysis['metricas_categoria']['volatilidade_media']}%
• Melhor ativo: {analysis['metricas_categoria']['melhor_ativo']}
• Pior ativo: {analysis['metricas_categoria']['pior_ativo']}
• Menos volátil: {analysis['metricas_categoria']['ativo_menos_volatil']}
• Mais volátil: {analysis['metricas_categoria']['ativo_mais_volatil']}

💰 PERFORMANCE INDIVIDUAL:
"""
        
        for symbol, metrics in analysis['metricas_individuais'].items():
            report += f"""
{symbol}:
  💵 Preço inicial: ${metrics['preco_inicial']}
  💵 Preço atual: ${metrics['preco_atual']}
  📈 Retorno: {metrics['retorno_total']}%
  📊 Volatilidade: {metrics['volatilidade']}%
  ⚡ Sharpe Ratio: {metrics['sharpe_ratio']}
"""
        
        return report

# Exemplo de uso
if __name__ == "__main__":
    analyzer = CategoryAnalyzer()
    
    # Listar categorias disponíveis
    print("📋 Categorias disponíveis:")
    for key, name in analyzer.get_available_categories().items():
        print(f"  • {key}: {name}")
    
    # Analisar uma categoria específica
    try:
        analysis = analyzer.analyze_category_performance('tecnologia', '6mo')
        print(analyzer.generate_category_report('tecnologia', '6mo'))
    except Exception as e:
        print(f"Erro na análise: {e}")