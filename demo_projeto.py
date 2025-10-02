#!/usr/bin/env python3
"""
Demonstração Completa do Projeto Big Data Finance
Desenvolvido por: Equipe Big Data Finance
Gestor: Fabio | APIs: Felipe Martins | Infraestrutura: Ana Luiza Pazze
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BigDataFinanceDemo:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.data = {}
        print("🚀 Iniciando Demonstração do Projeto Big Data Finance")
        print("=" * 60)
        
    def collect_financial_data(self):
        """Coleta dados financeiros - Módulo de Felipe Martins"""
        print("\n📊 1. COLETA DE DADOS FINANCEIROS (Felipe Martins)")
        print("-" * 50)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for symbol in self.symbols:
            try:
                print(f"Coletando dados de {symbol}...")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                info = ticker.info
                
                self.data[symbol] = {
                    'history': hist,
                    'info': info,
                    'current_price': hist['Close'].iloc[-1],
                    'volume': hist['Volume'].mean(),
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252)
                }
                print(f"✅ {symbol}: Preço atual ${self.data[symbol]['current_price']:.2f}")
                
            except Exception as e:
                print(f"❌ Erro ao coletar {symbol}: {e}")
        
        print(f"\n✅ Dados coletados para {len(self.data)} ativos")
        return self.data
    
    def process_data_spark_simulation(self):
        """Simula processamento Spark - Módulo de Ana Luiza Pazze"""
        print("\n⚡ 2. PROCESSAMENTO DISTRIBUÍDO (Ana Luiza Pazze)")
        print("-" * 50)
        
        processed_data = {}
        
        for symbol, data in self.data.items():
            print(f"Processando {symbol} com Spark (simulado)...")
            
            df = data['history'].copy()
            
            # Indicadores técnicos
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Sinais de trading
            df['Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
            
            processed_data[symbol] = df
            print(f"✅ {symbol}: Calculados SMA, volatilidade e sinais")
        
        self.processed_data = processed_data
        print("\n✅ Processamento Spark concluído")
        return processed_data
    
    def statistical_analysis(self):
        """Análise estatística - Módulo de Pedro Silva"""
        print("\n📈 3. ANÁLISE ESTATÍSTICA (Pedro Silva)")
        print("-" * 50)
        
        # Matriz de correlação
        returns_data = {}
        for symbol, df in self.processed_data.items():
            returns_data[symbol] = df['Daily_Return'].dropna()
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        print("Matriz de Correlação:")
        print(correlation_matrix.round(3))
        
        # Estatísticas descritivas
        print("\nEstatísticas dos Retornos Diários:")
        stats = returns_df.describe()
        print(stats.round(4))
        
        # Volatilidade anualizada
        print("\nVolatilidade Anualizada:")
        annual_vol = returns_df.std() * np.sqrt(252)
        for symbol, vol in annual_vol.items():
            print(f"{symbol}: {vol:.2%}")
        
        self.correlation_matrix = correlation_matrix
        self.returns_df = returns_df
        return correlation_matrix, returns_df
    
    def machine_learning_analysis(self):
        """Análise de Machine Learning - Módulo de Anny Caroline Sousa"""
        print("\n🤖 4. MACHINE LEARNING (Anny Caroline Sousa)")
        print("-" * 50)
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        
        ml_results = {}
        
        for symbol, df in self.processed_data.items():
            print(f"Treinando modelo para {symbol}...")
            
            # Preparar features
            features_df = df[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'Volatility']].dropna()
            target = df['Close'].loc[features_df.index]
            
            if len(features_df) > 50:  # Verificar se há dados suficientes
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, target, test_size=0.2, random_state=42
                )
                
                # Treinar modelo
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Predições
                y_pred = model.predict(X_test)
                
                # Métricas
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                ml_results[symbol] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'feature_importance': dict(zip(features_df.columns, model.feature_importances_))
                }
                
                print(f"✅ {symbol}: R² = {r2:.3f}, MSE = {mse:.2f}")
            else:
                print(f"❌ {symbol}: Dados insuficientes")
        
        self.ml_results = ml_results
        return ml_results
    
    def create_visualizations(self):
        """Criar visualizações - Módulo de Ricardo Areas"""
        print("\n📊 5. VISUALIZAÇÕES INTERATIVAS (Ricardo Areas)")
        print("-" * 50)
        
        # 1. Gráfico de preços
        fig1 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Preços das Ações', 'Volume de Negociação'),
            vertical_spacing=0.1
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (symbol, df) in enumerate(self.processed_data.items()):
            fig1.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name=symbol,
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
            
            fig1.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Volume'],
                    name=f'{symbol} Volume',
                    line=dict(color=colors[i % len(colors)], dash='dot'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig1.update_layout(
            title="Dashboard Financeiro - Big Data Finance",
            height=600,
            showlegend=True
        )
        
        # 2. Matriz de correlação
        fig2 = px.imshow(
            self.correlation_matrix,
            title="Matriz de Correlação dos Retornos",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        # 3. Gráfico de volatilidade
        volatilities = []
        symbols_list = []
        
        for symbol, data in self.data.items():
            volatilities.append(data['volatility'])
            symbols_list.append(symbol)
        
        fig3 = px.bar(
            x=symbols_list,
            y=volatilities,
            title="Volatilidade Anualizada por Ativo",
            labels={'x': 'Ativo', 'y': 'Volatilidade'}
        )
        
        # Salvar gráficos
        fig1.write_html("dashboard_precos.html")
        fig2.write_html("matriz_correlacao.html")
        fig3.write_html("volatilidade.html")
        
        print("✅ Gráficos salvos:")
        print("  - dashboard_precos.html")
        print("  - matriz_correlacao.html") 
        print("  - volatilidade.html")
        
        return fig1, fig2, fig3
    
    def generate_report(self):
        """Gerar relatório final - Coordenação Fabio"""
        print("\n📋 6. RELATÓRIO EXECUTIVO (Fabio - Gestão)")
        print("-" * 50)
        
        report = {
            'data_coleta': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ativos_analisados': len(self.data),
            'periodo_analise': '12 meses',
            'metricas_principais': {}
        }
        
        print("RESUMO EXECUTIVO:")
        print(f"📅 Data da análise: {report['data_coleta']}")
        print(f"📊 Ativos analisados: {report['ativos_analisados']}")
        print(f"⏰ Período: {report['periodo_analise']}")
        
        print("\nPERFORMANCE DOS ATIVOS:")
        for symbol, data in self.data.items():
            hist = data['history']
            total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            
            print(f"{symbol}:")
            print(f"  💰 Preço atual: ${data['current_price']:.2f}")
            print(f"  📈 Retorno total: {total_return:.2f}%")
            print(f"  📊 Volatilidade: {data['volatility']:.2%}")
            
            report['metricas_principais'][symbol] = {
                'preco_atual': data['current_price'],
                'retorno_total': total_return,
                'volatilidade': data['volatility']
            }
        
        if hasattr(self, 'ml_results'):
            print("\nMODELOS DE MACHINE LEARNING:")
            for symbol, results in self.ml_results.items():
                print(f"{symbol}: R² = {results['r2']:.3f}")
        
        # Salvar relatório
        import json
        with open('relatorio_executivo.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n✅ Relatório salvo em: relatorio_executivo.json")
        return report
    
    def run_complete_demo(self):
        """Executar demonstração completa"""
        print("🎯 PROJETO BIG DATA FINANCE - DEMONSTRAÇÃO COMPLETA")
        print("👥 Equipe: Fabio (Gestão) | Felipe (APIs) | Ana Luiza (Infra) | Pedro (Stats) | Anny (ML) | Ricardo (Viz)")
        print("=" * 80)
        
        try:
            # 1. Coleta de dados
            self.collect_financial_data()
            
            # 2. Processamento
            self.process_data_spark_simulation()
            
            # 3. Análise estatística
            self.statistical_analysis()
            
            # 4. Machine Learning
            self.machine_learning_analysis()
            
            # 5. Visualizações
            self.create_visualizations()
            
            # 6. Relatório
            self.generate_report()
            
            print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
            print("=" * 60)
            print("📁 Arquivos gerados:")
            print("  - dashboard_precos.html")
            print("  - matriz_correlacao.html")
            print("  - volatilidade.html")
            print("  - relatorio_executivo.json")
            print("\n🌐 Abra os arquivos HTML no navegador para ver as visualizações!")
            
        except Exception as e:
            print(f"\n❌ Erro durante a demonstração: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo = BigDataFinanceDemo()
    demo.run_complete_demo()