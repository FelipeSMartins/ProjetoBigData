"""
Demonstração Melhorada de Análise de Eventos Mundiais
Projeto Big Data Finance - Visualizações Claras e Separadas por Eventos

Desenvolvido por: Ricardo Areas & Equipe Big Data Finance
Objetivo: Criar visualizações mais claras com gráficos de linha e barras separados por eventos
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_analysis.events_analyzer import EventsAnalyzer
from src.visualization.events_charts import EventsCharts
import json
from datetime import datetime

def main():
    print("🌍 DEMONSTRAÇÃO MELHORADA - ANÁLISE DE EVENTOS MUNDIAIS")
    print("=" * 70)
    print("📊 Visualizações Claras e Separadas por Eventos")
    print("🎯 Gráficos de Linha e Barras Individuais")
    print("🔍 Filtros Interativos para Seleção de Eventos")
    print("=" * 70)
    
    # Inicializar analisador e visualizador
    analyzer = EventsAnalyzer()
    charts = EventsCharts()
    
    # Lista de eventos para análise
    eventos_para_analisar = [
        'svb_collapse',
        'covid19_pandemia',
        'brexit_referendum',
        'eleicoes_eua_2016',
        'eleicoes_eua_2020',
        'guerra_russia_ucrania',
        'fed_rate_hike_2022',
        'terremoto_japao_2011',
        'lehman_brothers',
        'credit_suisse_collapse'
    ]
    
    print("\n📈 Coletando dados dos eventos...")
    
    # Dicionário para armazenar dados de todos os eventos
    all_events_data = {}
    
    # Analisar cada evento individualmente
    for evento_id in eventos_para_analisar:
        try:
            print(f"\n🔍 Analisando: {evento_id}")
            analysis_data = analyzer.analyze_event_impact(evento_id)
            
            if analysis_data:
                evento_nome = analysis_data['evento']['name']
                all_events_data[evento_nome] = analysis_data
                
                # 1. Criar gráfico de linha individual para o evento
                print(f"  📈 Criando gráfico de linha para {evento_nome}...")
                line_chart = charts.create_simple_line_chart_by_event(evento_nome, analysis_data)
                line_filename = f"linha_{evento_id}.html"
                line_chart.write_html(line_filename)
                
                # 2. Criar gráfico de barras individual para o evento
                print(f"  📊 Criando gráfico de barras para {evento_nome}...")
                bar_chart = charts.create_simple_bar_chart_by_event(evento_nome, analysis_data)
                bar_filename = f"barras_{evento_id}.html"
                bar_chart.write_html(bar_filename)
                
                # 3. Criar dashboard completo individual
                print(f"  🎯 Criando dashboard completo para {evento_nome}...")
                dashboard = charts.create_individual_event_dashboard(analysis_data)
                dashboard_filename = f"dashboard_{evento_id}.html"
                dashboard.write_html(dashboard_filename)
                
                print(f"  ✅ Visualizações criadas:")
                print(f"     - Linha: {line_filename}")
                print(f"     - Barras: {bar_filename}")
                print(f"     - Dashboard: {dashboard_filename}")
                
        except Exception as e:
            print(f"  ❌ Erro ao analisar {evento_id}: {e}")
    
    # 4. Criar dashboard com seletor de eventos (filtro interativo)
    if all_events_data:
        print(f"\n🎛️ Criando dashboard com seletor de eventos...")
        selector_dashboard = charts.create_events_selector_dashboard(all_events_data)
        
        # Adicionar timestamp para evitar cache do navegador
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selector_filename = f"seletor_eventos_interativo_{timestamp}.html"
        
        selector_dashboard.write_html(selector_filename)
        print(f"✅ Dashboard interativo criado: {selector_filename}")
        
        # Também criar versão sem timestamp para compatibilidade
        selector_dashboard.write_html("seletor_eventos_interativo.html")
        print(f"✅ Dashboard padrão atualizado: seletor_eventos_interativo.html")
    
    # 5. Gerar relatório resumo
    print(f"\n📋 Gerando relatório resumo...")
    
    resumo_relatorio = {
        "timestamp": datetime.now().isoformat(),
        "total_eventos_analisados": len(all_events_data),
        "eventos_processados": list(all_events_data.keys()),
        "arquivos_gerados": {
            "graficos_linha": [f"linha_{evento_id}.html" for evento_id in eventos_para_analisar if any(evento_nome in all_events_data for evento_nome in all_events_data.keys())],
            "graficos_barras": [f"barras_{evento_id}.html" for evento_id in eventos_para_analisar if any(evento_nome in all_events_data for evento_nome in all_events_data.keys())],
            "dashboards_individuais": [f"dashboard_{evento_id}.html" for evento_id in eventos_para_analisar if any(evento_nome in all_events_data for evento_nome in all_events_data.keys())],
            "dashboard_interativo": "seletor_eventos_interativo.html"
        },
        "melhorias_implementadas": [
            "Gráficos de linha separados por evento",
            "Gráficos de barras individuais por evento",
            "Dashboard completo para cada evento (2x2 grid)",
            "Seletor interativo de eventos com dropdown",
            "Visualizações mais claras e focadas",
            "Cores diferenciadas por período (antes/durante/depois)",
            "Tooltips informativos em todos os gráficos"
        ],
        "funcionalidades_dashboard": [
            "Evolução de preços normalizados",
            "Comparação de retornos por período",
            "Análise de volatilidade",
            "Ranking de impactos por ativo",
            "Filtro interativo para seleção de eventos"
        ]
    }
    
    with open("relatorio_eventos_melhorados.json", "w", encoding="utf-8") as f:
        json.dump(resumo_relatorio, f, indent=2, ensure_ascii=False)
    
    print("✅ Relatório salvo: relatorio_eventos_melhorados.json")
    
    # Resumo final
    print("\n" + "=" * 70)
    print("🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print(f"📊 Total de eventos analisados: {len(all_events_data)}")
    print(f"📈 Gráficos de linha criados: {len(all_events_data)}")
    print(f"📊 Gráficos de barras criados: {len(all_events_data)}")
    print(f"🎯 Dashboards individuais: {len(all_events_data)}")
    print(f"🎛️ Dashboard interativo: 1")
    
    print("\n🔍 COMO USAR AS VISUALIZAÇÕES:")
    print("1. 📈 Gráficos de Linha (linha_*.html):")
    print("   - Mostram evolução dos preços ao longo do tempo")
    print("   - Linha vertical vermelha marca o evento")
    print("   - Cada ativo tem cor diferente")
    
    print("\n2. 📊 Gráficos de Barras (barras_*.html):")
    print("   - Mostram impacto durante o evento")
    print("   - Verde = impacto positivo, Vermelho = negativo")
    print("   - Valores em percentual")
    
    print("\n3. 🎯 Dashboards Individuais (dashboard_*.html):")
    print("   - 4 visualizações em uma tela")
    print("   - Evolução + Impactos + Volatilidade + Ranking")
    
    print("\n4. 🎛️ Dashboard Interativo (seletor_eventos_interativo.html):")
    print("   - Use o menu dropdown para selecionar eventos")
    print("   - Visualização muda dinamicamente")
    print("   - Compare diferentes eventos facilmente")
    
    print("\n🌟 MELHORIAS IMPLEMENTADAS:")
    print("✅ Visualizações mais claras e focadas")
    print("✅ Gráficos separados por evento")
    print("✅ Filtro interativo para seleção")
    print("✅ Cores consistentes e intuitivas")
    print("✅ Tooltips informativos")
    print("✅ Layouts responsivos")
    
    print("\n🚀 Para visualizar, abra os arquivos HTML no navegador!")
    print("=" * 70)

if __name__ == "__main__":
    main()