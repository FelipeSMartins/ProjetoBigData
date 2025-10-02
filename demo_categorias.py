"""
DEMONSTRAÇÃO DO SISTEMA DE ANÁLISE POR CATEGORIAS
Projeto Big Data Finance - Análise Setorial Avançada

Equipe:
- Gestão: Fabio
- APIs e Coleta de Dados: Felipe Martins  
- Arquitetura e Infraestrutura: Ana Luiza Pazze
- Análise Estatística: Pedro Silva
- Machine Learning: Anny Caroline Sousa
- Visualização: Ricardo Areas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_analysis.category_analyzer import CategoryAnalyzer
from visualization.category_charts import CategoryCharts
import json
from datetime import datetime

def print_header(title):
    """Imprime cabeçalho formatado"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_section(title, icon="📊"):
    """Imprime seção formatada"""
    print(f"\n{icon} {title}")
    print("-" * 50)

def main():
    """Demonstração completa do sistema de categorias"""
    
    print_header("SISTEMA DE ANÁLISE POR CATEGORIAS DE ATIVOS")
    print("🏢 Projeto Big Data Finance")
    print("👥 Equipe completa trabalhando em conjunto")
    print(f"📅 Executado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    
    # Inicializar analisadores
    analyzer = CategoryAnalyzer()
    charts = CategoryCharts()
    
    print_section("CATEGORIAS DISPONÍVEIS", "📋")
    categories = analyzer.get_available_categories()
    
    for i, (key, name) in enumerate(categories.items(), 1):
        symbols = analyzer.get_category_symbols(key)
        print(f"{i}. {name} ({key})")
        print(f"   🎯 Ativos: {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}")
        print(f"   📊 Total de ativos: {len(symbols)}")
    
    # Demonstrar análise individual de categorias
    demo_categories = ['tecnologia', 'petroleo', 'fiis', 'agronegocio']
    
    print_section("ANÁLISES INDIVIDUAIS POR CATEGORIA", "🔍")
    
    category_results = {}
    
    for category in demo_categories:
        try:
            print(f"\n🔄 Analisando categoria: {categories[category]}")
            
            # Análise da categoria
            analysis = analyzer.analyze_category_performance(category, period='6mo')
            category_results[category] = analysis
            
            # Métricas resumidas
            metrics = analysis['metricas_categoria']
            print(f"✅ Análise concluída!")
            print(f"   📈 Retorno médio: {metrics['retorno_medio']}%")
            print(f"   📊 Volatilidade média: {metrics['volatilidade_media']}%")
            print(f"   🏆 Melhor ativo: {metrics['melhor_ativo']}")
            print(f"   📉 Pior ativo: {metrics['pior_ativo']}")
            
            # Gerar visualizações
            print(f"   🎨 Gerando gráficos...")
            
            # Gráfico de performance
            perf_chart = charts.create_category_performance_chart(
                analysis, 
                f"categoria_{category}_performance.html"
            )
            
            # Gráfico de correlação
            corr_chart = charts.create_category_correlation_heatmap(
                analysis,
                f"categoria_{category}_correlacao.html"
            )
            
            # Gráfico de evolução de preços
            price_chart = charts.create_price_evolution_chart(
                analysis,
                f"categoria_{category}_precos.html"
            )
            
            print(f"   ✅ Gráficos salvos para {categories[category]}")
            
        except Exception as e:
            print(f"   ❌ Erro na análise de {category}: {str(e)}")
    
    # Comparação entre categorias
    print_section("COMPARAÇÃO ENTRE CATEGORIAS", "⚖️")
    
    try:
        print("🔄 Comparando todas as categorias analisadas...")
        comparison = analyzer.compare_categories(list(category_results.keys()), period='6mo')
        
        print("✅ Comparação concluída!")
        
        # Mostrar ranking
        print("\n🏆 RANKING POR RETORNO:")
        for i, (cat, data) in enumerate(comparison['ranking_retorno'], 1):
            print(f"{i}. {data['nome']}: {data['retorno_medio']}%")
        
        print("\n📊 RANKING POR VOLATILIDADE (menor = melhor):")
        for i, (cat, data) in enumerate(comparison['ranking_volatilidade'], 1):
            print(f"{i}. {data['nome']}: {data['volatilidade_media']}%")
        
        # Gerar dashboard comparativo
        print("\n🎨 Gerando dashboard comparativo...")
        
        # Dashboard principal
        dashboard = charts.create_risk_return_dashboard(
            comparison,
            "dashboard_categorias_completo.html"
        )
        
        # Gráfico comparativo
        comp_chart = charts.create_category_comparison_chart(
            comparison,
            "comparacao_categorias.html"
        )
        
        # Tabela resumo
        summary_table = charts.create_sector_summary_table(
            comparison,
            "resumo_categorias.html"
        )
        
        print("✅ Dashboard comparativo gerado!")
        
        # Salvar relatório JSON
        with open('relatorio_categorias.json', 'w', encoding='utf-8') as f:
            # Remover dados de preços para reduzir tamanho do arquivo
            comparison_clean = comparison.copy()
            for cat in comparison_clean['analises_detalhadas']:
                if 'dados_precos' in comparison_clean['analises_detalhadas'][cat]:
                    del comparison_clean['analises_detalhadas'][cat]['dados_precos']
            
            json.dump(comparison_clean, f, indent=2, ensure_ascii=False, default=str)
        
        print("✅ Relatório JSON salvo!")
        
    except Exception as e:
        print(f"❌ Erro na comparação: {str(e)}")
    
    # Relatório executivo
    print_section("RELATÓRIO EXECUTIVO", "📋")
    
    if category_results:
        print("📊 RESUMO EXECUTIVO - ANÁLISE SETORIAL")
        print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y')}")
        print(f"⏰ Período analisado: 6 meses")
        print(f"🎯 Categorias analisadas: {len(category_results)}")
        
        # Encontrar melhores e piores
        all_returns = []
        for cat, analysis in category_results.items():
            cat_return = analysis['metricas_categoria']['retorno_medio']
            all_returns.append((categories[cat], cat_return))
        
        all_returns.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 MELHOR CATEGORIA: {all_returns[0][0]} ({all_returns[0][1]}%)")
        print(f"📉 PIOR CATEGORIA: {all_returns[-1][0]} ({all_returns[-1][1]}%)")
        
        # Insights por categoria
        print(f"\n💡 INSIGHTS POR CATEGORIA:")
        for cat, analysis in category_results.items():
            metrics = analysis['metricas_categoria']
            individual = analysis['metricas_individuais']
            
            # Calcular dispersão de retornos
            returns = [m['retorno_total'] for m in individual.values()]
            dispersao = max(returns) - min(returns)
            
            print(f"\n• {categories[cat]}:")
            print(f"  📈 Retorno: {metrics['retorno_medio']}%")
            print(f"  📊 Volatilidade: {metrics['volatilidade_media']}%")
            print(f"  🎯 Dispersão: {dispersao:.1f}% (diferença entre melhor e pior ativo)")
            print(f"  ⭐ Destaque: {metrics['melhor_ativo']}")
    
    # Arquivos gerados
    print_section("ARQUIVOS GERADOS", "📁")
    
    generated_files = [
        "dashboard_categorias_completo.html",
        "comparacao_categorias.html", 
        "resumo_categorias.html",
        "relatorio_categorias.json"
    ]
    
    # Adicionar arquivos por categoria
    for category in category_results.keys():
        generated_files.extend([
            f"categoria_{category}_performance.html",
            f"categoria_{category}_correlacao.html",
            f"categoria_{category}_precos.html"
        ])
    
    print("📊 Dashboards e relatórios:")
    for i, file in enumerate(generated_files, 1):
        print(f"{i:2d}. {file}")
    
    print_section("CONCLUSÃO", "🎉")
    print("✅ Demonstração do sistema de categorias concluída com sucesso!")
    print("🌐 Abra os arquivos HTML no navegador para visualizar os dashboards")
    print("📊 Use o arquivo JSON para integrações com outros sistemas")
    print("🔄 O sistema está pronto para análises em tempo real!")
    
    print(f"\n👥 CONTRIBUIÇÕES DA EQUIPE:")
    print("• Fabio: Gestão e coordenação do projeto")
    print("• Felipe Martins: Desenvolvimento da coleta de dados por categoria")
    print("• Ana Luiza Pazze: Infraestrutura e arquitetura do sistema")
    print("• Pedro Silva: Algoritmos de análise estatística")
    print("• Anny Caroline Sousa: Modelos preditivos por setor")
    print("• Ricardo Areas: Dashboards e visualizações interativas")

if __name__ == "__main__":
    main()