import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import nnls
import warnings
import sys
import os

# Adiciona src ao path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_selection import select_features
from statistical_tests import compare_models

warnings.filterwarnings("ignore")

# Constantes
DATA_PATH = "data/dados_macro_bruto.xlsx"
OUTPUT_PLOT = "forecast_plot.png"
OUTPUT_PLOT_LEVEL = "forecast_plot_level.png"

# Grids de hiperparâmetros para GridSearchCV (deixei pouco gridsearch para não demorar muito)
RF_PARAM_GRID = {
    'n_estimators': [150, 200, 300],
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt']
}

GB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 3],
    'subsample': [0.6, 0.8],
    'max_features': ['sqrt', 0.5],
    'min_samples_leaf': [5, 10]
}

# Série original do IBC-Br para reconstrução de nível
ORIGINAL_IBC_BR = None


# ============== Otimização Adaptativa de Hiperparâmetros ==============
# Limiares para comportamento adaptativo baseado no tamanho amostral
MIN_SAMPLES_FOR_GRIDSEARCH = 120  # Necessário para folds de CV robustos
MIN_SAMPLES_PER_FOLD = 20         # Mínimo por fold de CV

# Parâmetros conservadores de fallback (regularização para amostras pequenas)
CONSERVATIVE_RF_PARAMS = {'n_estimators': 100, 'max_depth': 7, 'min_samples_leaf': 4}
CONSERVATIVE_GB_PARAMS = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8}

# Rastreamento de uso de fallback
_gridsearch_stats = {'rf_gridsearch': 0, 'rf_fallback': 0, 'gb_gridsearch': 0, 'gb_fallback': 0}


def get_adaptive_cv_splits(n_samples: int) -> int:
    """
    Determina número de splits de CV adaptativamente baseado no tamanho amostral.
    Garante que cada fold tenha pelo menos MIN_SAMPLES_PER_FOLD amostras.
    """
    max_splits = n_samples // MIN_SAMPLES_PER_FOLD
    if max_splits >= 5:
        return 5
    elif max_splits >= 3:
        return max_splits
    else:
        return 2  # CV mínimo viável


def get_best_rf(X_train, y_train):
    """
    Encontra melhores parâmetros do Random Forest com comportamento adaptativo.
    Usa parâmetros conservadores automaticamente para amostras pequenas.
    """
    global _gridsearch_stats
    n_samples = len(X_train)
    
    if n_samples >= MIN_SAMPLES_FOR_GRIDSEARCH:
        n_splits = get_adaptive_cv_splits(n_samples)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            RF_PARAM_GRID,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        _gridsearch_stats['rf_gridsearch'] += 1
        return gs.best_estimator_
    else:
        # Regularização adaptativa: parâmetros fixos conservadores
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **CONSERVATIVE_RF_PARAMS)
        rf.fit(X_train, y_train)
        _gridsearch_stats['rf_fallback'] += 1
        return rf


def get_best_gb(X_train, y_train):
    """
    Encontra melhores parâmetros do Gradient Boosting com comportamento adaptativo.
    Usa parâmetros conservadores automaticamente para amostras pequenas.
    """
    global _gridsearch_stats
    n_samples = len(X_train)
    
    if n_samples >= MIN_SAMPLES_FOR_GRIDSEARCH:
        n_splits = get_adaptive_cv_splits(n_samples)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            GB_PARAM_GRID,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        _gridsearch_stats['gb_gridsearch'] += 1
        return gs.best_estimator_
    else:
        # Regularização adaptativa: parâmetros fixos conservadores
        gb = GradientBoostingRegressor(random_state=42, **CONSERVATIVE_GB_PARAMS)
        gb.fit(X_train, y_train)
        _gridsearch_stats['gb_fallback'] += 1
        return gb


# ============== Carregamento e Pré-processamento ==============
def load_and_preprocess_data(filepath):
    global ORIGINAL_IBC_BR
    print("CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS")

    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    # Tratamento da coluna de data
    if '' in df.columns:
        df.rename(columns={'': 'Date'}, inplace=True)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    print(f"Dimensão inicial: {df.shape}")
    
    # Seleção de features por correspondência parcial
    col_mapping = {}
    for c in df.columns:
        if "IBC-Br" in c: col_mapping["IBC-Br"] = c
        if "Preços das Exportações" in c: col_mapping["Preco_Exp"] = c
        if "Preços das Importações" in c: col_mapping["Preco_Imp"] = c
        if "Exportações de Mercadorias" in c: col_mapping["Exportacoes"] = c
        if "Importações de Mercadorias" in c: col_mapping["Importacoes"] = c
        if "IPCA" in c: col_mapping["IPCA"] = c
        if "Meios de Pagamento - M1" in c: col_mapping["M1"] = c
        if "Soja" in c: col_mapping["Soja"] = c
        if "Açúcar" in c: col_mapping["Acucar"] = c
        if "Energia Eletrica" in c: col_mapping["Energia"] = c
        if "Derivados do Petróleo" in c: col_mapping["Petrol"] = c
        if "Indústria Geral" in c: col_mapping["Industria"] = c
        if "IBOVESPA" in c: col_mapping["Ibovespa"] = c
        if "SELIC" in c: col_mapping["Selic"] = c
        if "Embi+" in c: col_mapping["Embi"] = c
        if "EUA - Taxa Básica" in c: col_mapping["US_Rate"] = c
        if "Veículos Automotores" in c: col_mapping["Veiculos"] = c
        if "Dólar Comercial" in c: col_mapping["Dolar"] = c
        if "Confiança do Consumidor" in c: col_mapping["Confianca"] = c
        if "Saldo - Total Brasil" in c: col_mapping["Job_Balance"] = c
    
    df = df[list(col_mapping.values())]
    df.columns = list(col_mapping.keys())
    
    # Tratamento de valores ausentes
    df = df.ffill().bfill()
    
    # Armazena IBC-Br original para reconstrução de nível
    ORIGINAL_IBC_BR = df['IBC-Br'].copy()
    
    # 1. Juro Real - impacto real no investimento
    if 'Selic' in df.columns and 'IPCA' in df.columns:
        df['Juro_Real'] = df['Selic'] - (df['IPCA'].pct_change() * 100)
    
    # 2. Termos de Troca - riqueza relativa do país
    if 'Preco_Exp' in df.columns:
        if 'Preco_Imp' in df.columns:
            df['Termos_Troca'] = df['Preco_Exp'] / df['Preco_Imp']
        elif 'Dolar' in df.columns:
            df['Termos_Troca'] = df['Preco_Exp'] / df['Dolar']
    elif 'Exportacoes' in df.columns and 'Importacoes' in df.columns:
        df['Balanca_Comercial_Ratio'] = df['Exportacoes'] / df['Importacoes'].replace(0, np.nan)

    # 2.1 Spread Externo e Risco
    if 'Selic' in df.columns and 'US_Rate' in df.columns:
        df['Spread_Ext'] = df['Selic'] - df['US_Rate']
    
    if 'Embi' in df.columns:
        df['Embi'] = df['Embi']

    # 2.2 Liquidez e Atividade Real
    if 'M1' in df.columns and 'IPCA' in df.columns:
        df['M1_Real'] = df['M1'] / (df['IPCA'] / 100) # Normalização simples para remover efeito preço grosso modo
    
    if 'Job_Balance' in df.columns:
        df['Job_Balance'] = df['Job_Balance']
    
    if 'Petrol' in df.columns:
        df['Petrol'] = df['Petrol']
    
    # 3. Volatilidade Financeira - incerteza do mercado
    vol_windows = [3, 6]
    if 'Dolar' in df.columns:
        for w in vol_windows:
            df[f'Dolar_Vol_{w}m'] = df['Dolar'].pct_change().rolling(window=w).std()
    
    if 'Ibovespa' in df.columns:
        for w in vol_windows:
            df[f'Ibovespa_Vol_{w}m'] = df['Ibovespa'].pct_change().rolling(window=w).std()
    
    # 4. Hiato do Produto - economia aquecida ou ociosa?
    if 'IBC-Br' in df.columns:
        df['IBC_Trend_12m'] = df['IBC-Br'].rolling(window=12).mean()
        df['IBC_Trend_24m'] = df['IBC-Br'].rolling(window=24).mean()
        df['Hiato_12m'] = df['IBC-Br'] - df['IBC_Trend_12m']
        df['Hiato_24m'] = df['IBC-Br'] - df['IBC_Trend_24m']
    
    # 5. Dummies temporais
    df['Mes'] = df.index.month
    df['Trimestre'] = df.index.quarter
    
    # Colunas que não devem sofrer pct_change padrão
    cols_no_pct = ['Selic', 'Juro_Real', 'Mes', 'Trimestre', 'Hiato_12m', 'Hiato_24m',
                   'US_Rate', 'Spread_Ext', 'Embi', 'Job_Balance']
    cols_for_pct = [c for c in df.columns if c not in cols_no_pct]
    
    df_trans = df[cols_for_pct].pct_change()
    
    # Adiciona colunas que não sofrem pct_change
    if 'Selic' in df.columns:
        df_trans['Selic'] = df['Selic'].diff()
    if 'Juro_Real' in df.columns:
        df_trans['Juro_Real'] = df['Juro_Real'].diff()
    if 'Confianca' in df.columns:
        df_trans['Confianca'] = df['Confianca'].pct_change() # Confiança é índice, pct_change faz sentido
    
    # Tratamento específico para novas features
    if 'US_Rate' in df.columns:
        df_trans['US_Rate'] = df['US_Rate'].diff()
    if 'Spread_Ext' in df.columns:
        df_trans['Spread_Ext'] = df['Spread_Ext'].diff()
    if 'Embi' in df.columns:
        df_trans['Embi'] = df['Embi'].diff()
    if 'Job_Balance' in df.columns:
        df_trans['Job_Balance'] = df['Job_Balance']
    if 'M1_Real' in df.columns:
        df_trans['M1_Real'] = df['M1_Real'].pct_change()
    
    # Features de ciclo (já em nível)
    for col in ['Hiato_12m', 'Hiato_24m']:
        if col in df.columns:
            df_trans[col] = df[col]
    
    # Sazonalidade
    df_trans['Mes'] = df['Mes']
    df_trans['Trimestre'] = df['Trimestre']
    
    # Features de lag
    print("\nAdicionando lags (3, 12 meses) e médias móveis...")
    base_cols = [c for c in df_trans.columns if c not in ['Mes', 'Trimestre']]
    for col in base_cols:
        df_trans[f'{col}_lag3'] = df_trans[col].shift(3)
        df_trans[f'{col}_lag12'] = df_trans[col].shift(12)
    
    df_trans['IBC_MA3'] = df_trans['IBC-Br'].rolling(window=3).mean()
    df_trans['IBC_MA6'] = df_trans['IBC-Br'].rolling(window=6).mean()
    
    # Prepara target e features
    target = df_trans['IBC-Br']
    features = df_trans.drop(columns=['IBC-Br']).shift(1)
    
    # Combina e limpa
    data = pd.concat([target, features], axis=1)
    data = data.dropna()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"\nApós transformações: {data.shape}")
    print(f"[Engenharia] Total de features criadas: {data.shape[1] - 1}")
    
    # Aplica seleção estatística de features
    data = select_features(
        data, 
        target_col='IBC-Br', 
        adf_alpha=0.05, 
        corr_threshold=0.85,
        verbose=True
    )
    
    print(f"Dimensão final processada: {data.shape}")
    return data

# ============== Treinamento do Ensemble ==============
def train_ensemble(data):
    X = data.drop(columns=['IBC-Br'])
    y = data['IBC-Br']
    
    preds_rf = []
    preds_gb = []
    actuals = []
    dates = []
    
    initial_window = 48
    total_iterations = len(data) - initial_window
    
    print("\n" + "=" * 60)
    print("TREINAMENTO COM OTIMIZAÇÃO DINÂMICA DE HIPERPARÂMETROS")
    print(f"Janelas de treino: {total_iterations}")
    print("=" * 60)
    
    for i in range(initial_window, len(data)):
        train_idx = range(0, i)
        test_idx = [i]
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        progress = ((i - initial_window + 1) / total_iterations) * 100
        if (i - initial_window) % 20 == 0 or i == len(data) - 1:
            print(f"  Progresso: {progress:.1f}% (Janela {i - initial_window + 1}/{total_iterations})")
        
        # Otimização dinâmica de hiperparâmetros
        rf = get_best_rf(X_train, y_train)
        p_rf = rf.predict(X_test)[0]
        
        gb = get_best_gb(X_train, y_train)
        p_gb = gb.predict(X_test)[0]
        
        preds_rf.append(p_rf)
        preds_gb.append(p_gb)
        actuals.append(y_test.values[0])
        dates.append(y_test.index[0])
    
    # Meta-modelo com NNLS (pesos não-negativos e interpretáveis)
    print("\n" + "=" * 60)
    print("TREINAMENTO DO META-MODELO (NNLS - Pesos Restritos)")
    print("=" * 60)
    
    X_meta = np.column_stack([preds_rf, preds_gb])
    y_meta = np.array(actuals)
    
    # NNLS: min ||Xw - y||^2 sujeito a w >= 0
    weights, residual = nnls(X_meta, y_meta)
    
    # Normaliza pesos para somar 1
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights_normalized = weights / weight_sum
    else:
        weights_normalized = np.array([0.5, 0.5])
    
    print(f"  Pesos NNLS brutos:     RF={weights[0]:.4f}, GB={weights[1]:.4f}")
    print(f"  Normalizados (soma=1): RF={weights_normalized[0]:.2%}, GB={weights_normalized[1]:.2%}")
    
    # Aplica pesos normalizados
    meta_preds = X_meta @ weights_normalized

    results = pd.DataFrame({
        'Actual': actuals,
        'RF': preds_rf,
        'GB': preds_gb,
        'Ensemble': meta_preds
    }, index=dates)
    
    return results


# ============== Reconstrução de Nível ==============
def reconstruct_levels(results):
    """Reconstrói os níveis do índice IBC-Br a partir das variações percentuais."""
    global ORIGINAL_IBC_BR
    
    actual_levels = []
    rf_levels = []
    gb_levels = []
    ensemble_levels = []
    
    for date in results.index:
        try:
            idx_loc = ORIGINAL_IBC_BR.index.get_loc(date)
            prev_value = ORIGINAL_IBC_BR.iloc[idx_loc - 1]
        except:
            continue
        
        actual_levels.append(ORIGINAL_IBC_BR.loc[date])
        rf_levels.append(prev_value * (1 + results.loc[date, 'RF']))
        gb_levels.append(prev_value * (1 + results.loc[date, 'GB']))
        ensemble_levels.append(prev_value * (1 + results.loc[date, 'Ensemble']))
    
    return pd.DataFrame({
        'Actual': actual_levels,
        'RF': rf_levels,
        'GB': gb_levels,
        'Ensemble': ensemble_levels
    }, index=results.index[:len(actual_levels)])


# ============== Avaliação ==============
def evaluate(results, results_level):
    n_var = len(results)
    n_level = len(results_level)
    
    print("\n" + "=" * 60)
    print(f"MÉTRICAS SOBRE VARIAÇÕES (% Mensal) | n={n_var}")
    print("=" * 60)
    for model in ['RF', 'GB', 'Ensemble']:
        rmse = np.sqrt(mean_squared_error(results['Actual'], results[model]))
        mae = mean_absolute_error(results['Actual'], results[model])
        print(f"  {model:10} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    print("\n" + "=" * 60)
    print(f"MÉTRICAS SOBRE NÍVEIS RECONSTRUÍDOS (Índice IBC-Br) | n={n_level}")
    print("=" * 60)
    for model in ['RF', 'GB', 'Ensemble']:
        rmse = np.sqrt(mean_squared_error(results_level['Actual'], results_level[model]))
        mae = mean_absolute_error(results_level['Actual'], results_level[model])
        mape = np.mean(np.abs((results_level['Actual'] - results_level[model]) / results_level['Actual'])) * 100
        print(f"  {model:10} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # Testes Diebold-Mariano
    compare_models(results_level)


# ============== Gráficos ==============
def plot_results(results, results_level):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Variações
    ax1 = axes[0]
    ax1.plot(results.index, results['Actual'], label='Real (Variação %)', color='black', linewidth=1.5)
    ax1.plot(results.index, results['Ensemble'], label='Ensemble', color='red', linestyle='--', alpha=0.8)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title("Variação Mensal do IBC-Br (%) - Real vs Ensemble", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Variação (%)")
    
    # Subplot 2: Níveis Reconstruídos
    ax2 = axes[1]
    ax2.plot(results_level.index, results_level['Actual'], label='Real (Índice IBC-Br)', color='black', linewidth=2)
    ax2.plot(results_level.index, results_level['Ensemble'], label='Stacking Ensemble', color='red', linestyle='--', linewidth=1.5)
    ax2.plot(results_level.index, results_level['RF'], label='Random Forest', color='blue', linestyle=':', alpha=0.6)
    ax2.plot(results_level.index, results_level['GB'], label='Gradient Boosting', color='green', linestyle=':', alpha=0.6)
    ax2.set_title("Nível do Índice IBC-Br - Real vs Previsões", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Índice IBC-Br (2022=100)")
    ax2.set_xlabel("Data")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nGráfico salvo em {OUTPUT_PLOT}")
    
    # Gráfico de nível para apresentação
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_level.index, results_level['Actual'], label='Real (IBC-Br)', color='black', linewidth=2)
    ax.plot(results_level.index, results_level['Ensemble'], label='Stacking Ensemble', color='crimson', linewidth=1.5)
    ax.fill_between(results_level.index, results_level['Actual'], results_level['Ensemble'], alpha=0.2, color='red')
    ax.set_title("Previsão: IBC-Br (Proxy do PIB) - Real vs Stacking Ensemble", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Índice IBC-Br (2022=100)", fontsize=11)
    ax.set_xlabel("Data", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_LEVEL, dpi=150)
    print(f"Gráfico de nível salvo em {OUTPUT_PLOT_LEVEL}")


# ============== Principal ==============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    
    data = load_and_preprocess_data(DATA_PATH)
    results = train_ensemble(data)
    results_level = reconstruct_levels(results)
    evaluate(results, results_level)
    plot_results(results, results_level)
    
    print("\n" + "=" * 60)
