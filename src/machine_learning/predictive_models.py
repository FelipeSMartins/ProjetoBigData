# Predictive Models
# Responsável: Anny Caroline Sousa
# Modelos de machine learning para predição de movimentos de mercado

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de machine learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Bibliotecas para séries temporais
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Bibliotecas para deep learning (opcional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engenharia de features para dados financeiros
    """
    
    def __init__(self):
        """
        Inicializa o engenheiro de features
        """
        self.scalers = {}
    
    def create_technical_indicators(self, data: pd.DataFrame,
                                  price_column: str = 'close') -> pd.DataFrame:
        """
        Cria indicadores técnicos
        
        Args:
            data: DataFrame com dados de preços
            price_column: Nome da coluna de preços
            
        Returns:
            DataFrame com indicadores técnicos
        """
        try:
            df = data.copy()
            
            # Médias móveis
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df[price_column].rolling(window=window).mean()
                df[f'ma_ratio_{window}'] = df[price_column] / df[f'ma_{window}']
            
            # Médias móveis exponenciais
            for span in [12, 26]:
                df[f'ema_{span}'] = df[price_column].ewm(span=span).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df[price_column].rolling(window=20).mean()
            bb_std = df[price_column].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df[price_column] - df['bb_lower']) / df['bb_width']
            
            # Volatilidade
            df['volatility_5'] = df[price_column].rolling(window=5).std()
            df['volatility_20'] = df[price_column].rolling(window=20).std()
            
            # Retornos
            df['return_1d'] = df[price_column].pct_change()
            df['return_5d'] = df[price_column].pct_change(periods=5)
            df['return_20d'] = df[price_column].pct_change(periods=20)
            
            # Volume indicators (se disponível)
            if 'volume' in df.columns:
                df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
                
                # On-Balance Volume
                df['obv'] = (np.sign(df['return_1d']) * df['volume']).cumsum()
            
            logger.info("Indicadores técnicos criados com sucesso")
            return df
            
        except Exception as e:
            logger.error(f"Erro na criação de indicadores técnicos: {str(e)}")
            return data
    
    def create_lag_features(self, data: pd.DataFrame,
                          columns: List[str],
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Cria features com lag temporal
        
        Args:
            data: DataFrame com dados
            columns: Colunas para criar lags
            lags: Lista de lags a criar
            
        Returns:
            DataFrame com features de lag
        """
        try:
            df = data.copy()
            
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            logger.info(f"Features de lag criadas para {len(columns)} colunas")
            return df
            
        except Exception as e:
            logger.error(f"Erro na criação de features de lag: {str(e)}")
            return data
    
    def create_rolling_features(self, data: pd.DataFrame,
                              columns: List[str],
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Cria features com janelas móveis
        
        Args:
            data: DataFrame com dados
            columns: Colunas para criar features
            windows: Tamanhos das janelas
            
        Returns:
            DataFrame com features de janela móvel
        """
        try:
            df = data.copy()
            
            for col in columns:
                if col in df.columns:
                    for window in windows:
                        df[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                        df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                        df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                        df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
            
            logger.info(f"Features de janela móvel criadas para {len(columns)} colunas")
            return df
            
        except Exception as e:
            logger.error(f"Erro na criação de features de janela móvel: {str(e)}")
            return data

class PredictiveModels:
    """
    Modelos preditivos para análise financeira
    """
    
    def __init__(self):
        """
        Inicializa os modelos preditivos
        """
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
    
    def prepare_data(self, data: pd.DataFrame,
                    target_column: str,
                    feature_columns: Optional[List[str]] = None,
                    create_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para modelagem
        
        Args:
            data: DataFrame com dados
            target_column: Nome da coluna target
            feature_columns: Colunas de features (None para auto-seleção)
            create_features: Se deve criar features automaticamente
            
        Returns:
            Tupla com features e target
        """
        try:
            df = data.copy()
            
            # Criar features técnicas se solicitado
            if create_features:
                df = self.feature_engineer.create_technical_indicators(df)
                
                # Criar features de lag para colunas numéricas
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numeric_cols:
                    numeric_cols.remove(target_column)
                
                df = self.feature_engineer.create_lag_features(df, numeric_cols[:5])  # Limitar para evitar muitas features
            
            # Selecionar features
            if feature_columns is None:
                # Auto-seleção: todas as colunas numéricas exceto target
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in feature_columns:
                    feature_columns.remove(target_column)
            
            # Remover colunas com muitos NaN
            valid_features = []
            for col in feature_columns:
                if col in df.columns and df[col].notna().sum() > len(df) * 0.5:
                    valid_features.append(col)
            
            self.feature_columns = valid_features
            
            # Preparar X e y
            X = df[valid_features].fillna(method='ffill').fillna(0)
            y = df[target_column].fillna(method='ffill')
            
            # Remover linhas com NaN no target
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            logger.info(f"Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Erro na preparação dos dados: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Dict[str, Any]:
        """
        Treina modelos de regressão
        
        Args:
            X: Features
            y: Target
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Resultados dos modelos
        """
        try:
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['regression'] = scaler
            
            # Definir modelos
            models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'svr': SVR(kernel='rbf', C=1.0)
            }
            
            results = {}
            
            # Treinar e avaliar cada modelo
            for name, model in models.items():
                try:
                    # Treinar modelo
                    if name in ['linear_regression', 'ridge', 'lasso', 'svr']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calcular métricas
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'model': model,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse),
                        'predictions': y_pred,
                        'actual': y_test.values
                    }
                    
                    logger.info(f"{name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
                    
                except Exception as e:
                    logger.error(f"Erro no treinamento do modelo {name}: {str(e)}")
                    continue
            
            # Salvar melhor modelo
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
                self.models['best_regression'] = results[best_model_name]['model']
                logger.info(f"Melhor modelo de regressão: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento de modelos de regressão: {str(e)}")
            return {}
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series,
                                  test_size: float = 0.2,
                                  random_state: int = 42) -> Dict[str, Any]:
        """
        Treina modelos de classificação
        
        Args:
            X: Features
            y: Target (classes)
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Resultados dos modelos
        """
        try:
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['classification'] = scaler
            
            # Definir modelos
            models = {
                'logistic_regression': LogisticRegression(random_state=random_state),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                'svc': SVC(kernel='rbf', C=1.0, random_state=random_state)
            }
            
            results = {}
            
            # Treinar e avaliar cada modelo
            for name, model in models.items():
                try:
                    # Treinar modelo
                    if name in ['logistic_regression', 'svc']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_scaled)
                        else:
                            y_proba = None
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                    
                    # Calcular métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'predictions': y_pred,
                        'probabilities': y_proba,
                        'actual': y_test.values,
                        'classification_report': classification_report(y_test, y_pred)
                    }
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                    
                except Exception as e:
                    logger.error(f"Erro no treinamento do modelo {name}: {str(e)}")
                    continue
            
            # Salvar melhor modelo
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
                self.models['best_classification'] = results[best_model_name]['model']
                logger.info(f"Melhor modelo de classificação: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento de modelos de classificação: {str(e)}")
            return {}
    
    def create_price_direction_target(self, data: pd.DataFrame,
                                    price_column: str = 'close',
                                    periods: int = 1) -> pd.Series:
        """
        Cria target para direção do preço (subida/descida)
        
        Args:
            data: DataFrame com dados
            price_column: Nome da coluna de preços
            periods: Períodos à frente para prever
            
        Returns:
            Série com direção do preço (1=subida, 0=descida)
        """
        try:
            future_return = data[price_column].pct_change(periods=periods).shift(-periods)
            direction = (future_return > 0).astype(int)
            return direction
            
        except Exception as e:
            logger.error(f"Erro na criação do target de direção: {str(e)}")
            return pd.Series()
    
    def create_volatility_target(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               window: int = 5) -> pd.Series:
        """
        Cria target para volatilidade futura
        
        Args:
            data: DataFrame com dados
            price_column: Nome da coluna de preços
            window: Janela para calcular volatilidade
            
        Returns:
            Série com volatilidade futura
        """
        try:
            returns = data[price_column].pct_change()
            future_volatility = returns.rolling(window=window).std().shift(-window)
            return future_volatility
            
        except Exception as e:
            logger.error(f"Erro na criação do target de volatilidade: {str(e)}")
            return pd.Series()
    
    def predict_with_model(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com modelo treinado
        
        Args:
            model_name: Nome do modelo
            X: Features para predição
            
        Returns:
            Array com predições
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Modelo {model_name} não encontrado")
            
            model = self.models[model_name]
            
            # Aplicar escalonamento se necessário
            scaler_key = 'regression' if 'regression' in model_name else 'classification'
            if scaler_key in self.scalers:
                X_scaled = self.scalers[scaler_key].transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            return np.array([])
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series,
                           model, cv_folds: int = 5) -> Dict[str, float]:
        """
        Validação cruzada temporal para séries temporais
        
        Args:
            X: Features
            y: Target
            model: Modelo para validar
            cv_folds: Número de folds
            
        Returns:
            Métricas de validação cruzada
        """
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            scores = {
                'mse': [],
                'mae': [],
                'r2': []
            }
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Treinar modelo
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
                
                # Calcular métricas
                scores['mse'].append(mean_squared_error(y_val_cv, y_pred_cv))
                scores['mae'].append(mean_absolute_error(y_val_cv, y_pred_cv))
                scores['r2'].append(r2_score(y_val_cv, y_pred_cv))
            
            # Calcular médias
            cv_results = {
                'mse_mean': np.mean(scores['mse']),
                'mse_std': np.std(scores['mse']),
                'mae_mean': np.mean(scores['mae']),
                'mae_std': np.std(scores['mae']),
                'r2_mean': np.mean(scores['r2']),
                'r2_std': np.std(scores['r2'])
            }
            
            logger.info(f"Validação cruzada - R² médio: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"Erro na validação cruzada: {str(e)}")
            return {}
    
    def generate_model_report(self, results: Dict[str, Any],
                            model_type: str = 'regression') -> str:
        """
        Gera relatório dos modelos
        
        Args:
            results: Resultados dos modelos
            model_type: Tipo de modelo ('regression' ou 'classification')
            
        Returns:
            Relatório em texto
        """
        try:
            report = []
            report.append("=" * 60)
            report.append(f"RELATÓRIO DE MODELOS - {model_type.upper()}")
            report.append("=" * 60)
            report.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            if model_type == 'regression':
                report.append("MÉTRICAS DE REGRESSÃO")
                report.append("-" * 30)
                for name, result in results.items():
                    report.append(f"\n{name.upper()}:")
                    report.append(f"  R²: {result['r2']:.4f}")
                    report.append(f"  RMSE: {result['rmse']:.4f}")
                    report.append(f"  MAE: {result['mae']:.4f}")
            
            elif model_type == 'classification':
                report.append("MÉTRICAS DE CLASSIFICAÇÃO")
                report.append("-" * 30)
                for name, result in results.items():
                    report.append(f"\n{name.upper()}:")
                    report.append(f"  Accuracy: {result['accuracy']:.4f}")
                    report.append(f"  Precision: {result['precision']:.4f}")
                    report.append(f"  Recall: {result['recall']:.4f}")
                    report.append(f"  F1-Score: {result['f1']:.4f}")
            
            report.append("")
            report.append(f"Total de features utilizadas: {len(self.feature_columns)}")
            report.append(f"Features principais: {', '.join(self.feature_columns[:10])}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            return "Erro na geração do relatório"

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simular dados de preços
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    volumes = np.random.randint(1000000, 10000000, len(dates))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # Inicializar modelos
    models = PredictiveModels()
    
    # Preparar dados para regressão (prever preço futuro)
    sample_data['future_price'] = sample_data['close'].shift(-1)
    X, y = models.prepare_data(sample_data.dropna(), 'future_price')
    
    if not X.empty:
        # Treinar modelos de regressão
        regression_results = models.train_regression_models(X, y)
        print("Modelos de regressão treinados")
        
        # Criar target de classificação (direção do preço)
        sample_data['price_direction'] = models.create_price_direction_target(sample_data)
        X_class, y_class = models.prepare_data(sample_data.dropna(), 'price_direction', create_features=False)
        
        if not X_class.empty:
            # Treinar modelos de classificação
            classification_results = models.train_classification_models(X_class, y_class)
            print("Modelos de classificação treinados")
            
            # Gerar relatórios
            reg_report = models.generate_model_report(regression_results, 'regression')
            class_report = models.generate_model_report(classification_results, 'classification')
            
            print("\n" + reg_report)
            print("\n" + class_report)