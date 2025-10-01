# Exploratory Data Analyzer
# Responsável: Pedro Silva
# Análise exploratória de dados financeiros e eventos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExploratoryAnalyzer:
    """
    Analisador exploratório para dados financeiros e eventos mundiais
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Inicializa o analisador exploratório
        
        Args:
            figsize: Tamanho padrão das figuras
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Visão geral dos dados
        
        Args:
            data: DataFrame para análise
            
        Returns:
            Dicionário com informações gerais
        """
        try:
            overview = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'missing_values': data.isnull().sum().to_dict(),
                'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': data.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Estatísticas por tipo de coluna
            if overview['numeric_columns']:
                overview['numeric_summary'] = data[overview['numeric_columns']].describe().to_dict()
            
            if overview['categorical_columns']:
                overview['categorical_summary'] = {}
                for col in overview['categorical_columns']:
                    overview['categorical_summary'][col] = {
                        'unique_values': data[col].nunique(),
                        'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                        'value_counts': data[col].value_counts().head().to_dict()
                    }
            
            logger.info(f"Visão geral gerada para dataset com {data.shape[0]} linhas e {data.shape[1]} colunas")
            return overview
            
        except Exception as e:
            logger.error(f"Erro na visão geral dos dados: {str(e)}")
            return {}
    
    def plot_missing_values(self, data: pd.DataFrame, save_path: str = None) -> None:
        """
        Visualiza valores ausentes
        
        Args:
            data: DataFrame para análise
            save_path: Caminho para salvar o gráfico
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Heatmap de valores ausentes
            missing_data = data.isnull()
            sns.heatmap(missing_data, cbar=True, ax=axes[0,0], cmap='viridis')
            axes[0,0].set_title('Padrão de Valores Ausentes')
            
            # Contagem de valores ausentes por coluna
            missing_counts = data.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
            
            if not missing_counts.empty:
                missing_counts.plot(kind='bar', ax=axes[0,1])
                axes[0,1].set_title('Valores Ausentes por Coluna')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # Percentual de valores ausentes
            missing_pct = (data.isnull().sum() / len(data) * 100)
            missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
            
            if not missing_pct.empty:
                missing_pct.plot(kind='bar', ax=axes[1,0], color='orange')
                axes[1,0].set_title('Percentual de Valores Ausentes')
                axes[1,0].set_ylabel('Percentual (%)')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # Matriz de correlação de valores ausentes
            if len(data.columns) > 1:
                missing_corr = data.isnull().corr()
                sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
                axes[1,1].set_title('Correlação entre Valores Ausentes')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info("Visualização de valores ausentes criada")
            
        except Exception as e:
            logger.error(f"Erro na visualização de valores ausentes: {str(e)}")
    
    def plot_distributions(self, data: pd.DataFrame, 
                          columns: List[str] = None,
                          save_path: str = None) -> None:
        """
        Plota distribuições das variáveis numéricas
        
        Args:
            data: DataFrame com dados
            columns: Colunas para plotar (se None, usa todas numéricas)
            save_path: Caminho para salvar o gráfico
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns:
                logger.warning("Nenhuma coluna numérica encontrada")
                return
            
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(columns):
                if i < len(axes):
                    # Histograma com curva de densidade
                    data[col].hist(bins=30, alpha=0.7, ax=axes[i], density=True)
                    data[col].plot.density(ax=axes[i], color='red', linewidth=2)
                    
                    axes[i].set_title(f'Distribuição de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Densidade')
                    
                    # Adicionar estatísticas
                    mean_val = data[col].mean()
                    median_val = data[col].median()
                    axes[i].axvline(mean_val, color='green', linestyle='--', label=f'Média: {mean_val:.2f}')
                    axes[i].axvline(median_val, color='orange', linestyle='--', label=f'Mediana: {median_val:.2f}')
                    axes[i].legend()
            
            # Remover subplots vazios
            for i in range(len(columns), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info(f"Distribuições plotadas para {len(columns)} variáveis")
            
        except Exception as e:
            logger.error(f"Erro na plotagem de distribuições: {str(e)}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              method: str = 'pearson',
                              save_path: str = None) -> None:
        """
        Plota matriz de correlação
        
        Args:
            data: DataFrame com dados
            method: Método de correlação
            save_path: Caminho para salvar o gráfico
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                logger.warning("Nenhuma variável numérica para correlação")
                return
            
            corr_matrix = numeric_data.corr(method=method)
            
            plt.figure(figsize=self.figsize)
            
            # Criar máscara para triângulo superior
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Heatmap
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8})
            
            plt.title(f'Matriz de Correlação ({method.capitalize()})')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info("Matriz de correlação plotada")
            
        except Exception as e:
            logger.error(f"Erro na plotagem da matriz de correlação: {str(e)}")
    
    def plot_time_series(self, data: pd.DataFrame, 
                        date_column: str = None,
                        value_columns: List[str] = None,
                        save_path: str = None) -> None:
        """
        Plota séries temporais
        
        Args:
            data: DataFrame com dados
            date_column: Nome da coluna de data (se None, usa index)
            value_columns: Colunas de valores para plotar
            save_path: Caminho para salvar o gráfico
        """
        try:
            # Preparar dados
            if date_column and date_column in data.columns:
                plot_data = data.set_index(date_column)
            else:
                plot_data = data.copy()
            
            # Garantir que o index é datetime
            if not isinstance(plot_data.index, pd.DatetimeIndex):
                try:
                    plot_data.index = pd.to_datetime(plot_data.index)
                except:
                    logger.error("Não foi possível converter index para datetime")
                    return
            
            if value_columns is None:
                value_columns = plot_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not value_columns:
                logger.warning("Nenhuma coluna numérica encontrada")
                return
            
            n_cols = len(value_columns)
            fig, axes = plt.subplots(n_cols, 1, figsize=(self.figsize[0], 4*n_cols))
            
            if n_cols == 1:
                axes = [axes]
            
            for i, col in enumerate(value_columns):
                if col in plot_data.columns:
                    plot_data[col].plot(ax=axes[i], linewidth=1.5)
                    axes[i].set_title(f'Série Temporal - {col}')
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Adicionar média móvel se houver dados suficientes
                    if len(plot_data) > 30:
                        ma_30 = plot_data[col].rolling(window=30).mean()
                        ma_30.plot(ax=axes[i], color='red', alpha=0.7, label='MA 30')
                        axes[i].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info(f"Séries temporais plotadas para {len(value_columns)} variáveis")
            
        except Exception as e:
            logger.error(f"Erro na plotagem de séries temporais: {str(e)}")
    
    def plot_boxplots(self, data: pd.DataFrame, 
                     columns: List[str] = None,
                     save_path: str = None) -> None:
        """
        Plota boxplots para identificar outliers
        
        Args:
            data: DataFrame com dados
            columns: Colunas para plotar
            save_path: Caminho para salvar o gráfico
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns:
                logger.warning("Nenhuma coluna numérica encontrada")
                return
            
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(columns):
                if i < len(axes):
                    data.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Boxplot - {col}')
                    axes[i].set_ylabel(col)
            
            # Remover subplots vazios
            for i in range(len(columns), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info(f"Boxplots criados para {len(columns)} variáveis")
            
        except Exception as e:
            logger.error(f"Erro na criação de boxplots: {str(e)}")
    
    def plot_event_impact(self, price_data: pd.DataFrame,
                         event_dates: List[datetime],
                         price_column: str = 'close',
                         window: int = 10,
                         save_path: str = None) -> None:
        """
        Visualiza impacto de eventos nos preços
        
        Args:
            price_data: DataFrame com dados de preços
            event_dates: Lista de datas dos eventos
            price_column: Nome da coluna de preços
            window: Janela de dias ao redor do evento
            save_path: Caminho para salvar o gráfico
        """
        try:
            if price_column not in price_data.columns:
                logger.error(f"Coluna {price_column} não encontrada")
                return
            
            # Garantir que o index é datetime
            if not isinstance(price_data.index, pd.DatetimeIndex):
                logger.error("Index do DataFrame deve ser datetime")
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 10))
            
            # Gráfico 1: Preços com eventos marcados
            price_data[price_column].plot(ax=axes[0], linewidth=1, alpha=0.8)
            
            for event_date in event_dates:
                axes[0].axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
            
            axes[0].set_title('Preços com Eventos Marcados')
            axes[0].set_ylabel('Preço')
            axes[0].grid(True, alpha=0.3)
            
            # Gráfico 2: Retornos com eventos
            returns = price_data[price_column].pct_change()
            returns.plot(ax=axes[1], linewidth=1, alpha=0.8, color='green')
            
            for event_date in event_dates:
                axes[1].axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
            
            axes[1].set_title('Retornos com Eventos Marcados')
            axes[1].set_ylabel('Retorno')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info(f"Visualização de impacto criada para {len(event_dates)} eventos")
            
        except Exception as e:
            logger.error(f"Erro na visualização de impacto de eventos: {str(e)}")
    
    def plot_volatility_analysis(self, price_data: pd.DataFrame,
                               price_column: str = 'close',
                               window: int = 30,
                               save_path: str = None) -> None:
        """
        Visualiza análise de volatilidade
        
        Args:
            price_data: DataFrame com dados de preços
            price_column: Nome da coluna de preços
            window: Janela para volatilidade móvel
            save_path: Caminho para salvar o gráfico
        """
        try:
            if price_column not in price_data.columns:
                logger.error(f"Coluna {price_column} não encontrada")
                return
            
            # Calcular retornos e volatilidade
            prices = price_data[price_column]
            returns = prices.pct_change().dropna()
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], 12))
            
            # Preços
            prices.plot(ax=axes[0], linewidth=1.5)
            axes[0].set_title('Preços')
            axes[0].set_ylabel('Preço')
            axes[0].grid(True, alpha=0.3)
            
            # Retornos
            returns.plot(ax=axes[1], linewidth=1, alpha=0.8, color='green')
            axes[1].set_title('Retornos Diários')
            axes[1].set_ylabel('Retorno')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Volatilidade móvel
            rolling_vol.plot(ax=axes[2], linewidth=1.5, color='red')
            axes[2].set_title(f'Volatilidade Móvel ({window} dias)')
            axes[2].set_ylabel('Volatilidade Anualizada')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            logger.info("Análise de volatilidade visualizada")
            
        except Exception as e:
            logger.error(f"Erro na visualização de volatilidade: {str(e)}")
    
    def generate_eda_report(self, data: pd.DataFrame, 
                          output_dir: str = "reports") -> str:
        """
        Gera relatório completo de análise exploratória
        
        Args:
            data: DataFrame para análise
            output_dir: Diretório para salvar relatórios
            
        Returns:
            Caminho do relatório gerado
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"eda_report_{timestamp}.html")
            
            # Gerar visão geral
            overview = self.data_overview(data)
            
            # Criar HTML do relatório
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Relatório de Análise Exploratória</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>Relatório de Análise Exploratória de Dados</h1>
                <p><strong>Data de geração:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Visão Geral dos Dados</h2>
                <div class="metric">
                    <p><strong>Dimensões:</strong> {overview['shape'][0]} linhas × {overview['shape'][1]} colunas</p>
                    <p><strong>Uso de memória:</strong> {overview['memory_usage'] / 1024 / 1024:.2f} MB</p>
                    <p><strong>Linhas duplicadas:</strong> {overview['duplicate_rows']}</p>
                </div>
                
                <h2>Tipos de Dados</h2>
                <table>
                    <tr><th>Coluna</th><th>Tipo</th><th>Valores Ausentes</th><th>% Ausentes</th></tr>
            """
            
            for col in data.columns:
                missing_count = overview['missing_values'][col]
                missing_pct = overview['missing_percentage'][col]
                dtype = overview['dtypes'][col]
                
                html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{dtype}</td>
                        <td>{missing_count}</td>
                        <td>{missing_pct:.2f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Resumo Estatístico</h2>
            """
            
            if overview['numeric_columns']:
                html_content += "<h3>Variáveis Numéricas</h3><table>"
                html_content += "<tr><th>Variável</th><th>Média</th><th>Desvio Padrão</th><th>Mínimo</th><th>Máximo</th></tr>"
                
                for col in overview['numeric_columns']:
                    if col in overview['numeric_summary']:
                        stats = overview['numeric_summary'][col]
                        html_content += f"""
                            <tr>
                                <td>{col}</td>
                                <td>{stats['mean']:.4f}</td>
                                <td>{stats['std']:.4f}</td>
                                <td>{stats['min']:.4f}</td>
                                <td>{stats['max']:.4f}</td>
                            </tr>
                        """
                
                html_content += "</table>"
            
            html_content += """
                </body>
            </html>
            """
            
            # Salvar relatório
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Relatório EDA gerado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erro na geração do relatório EDA: {str(e)}")
            return ""

# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'price': 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))),
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'returns': np.random.normal(0.001, 0.02, len(dates))
    })
    
    # Inicializar analisador
    analyzer = ExploratoryAnalyzer()
    
    # Executar análises
    overview = analyzer.data_overview(df)
    print("Análise exploratória concluída!")
    print(f"Dataset: {overview['shape'][0]} linhas × {overview['shape'][1]} colunas")