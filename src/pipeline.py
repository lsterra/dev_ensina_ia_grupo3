import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
import warnings
import sys
import os

from xgboost import XGBRegressor

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
    'n_estimators': [150, 200],
    'max_depth': [5, 7],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt']
}

GB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 3],
    'subsample': [0.7],
    'min_samples_leaf': [5]
}

EN_PARAM_GRID = {
    'alpha': [0.01, 0.1, 1.0],
    'l1_ratio': [0.2, 0.5, 0.8]
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.7],
    'colsample_bytree': [0.7]
}

ORIGINAL_IBC_BR = None
MIN_SAMPLES_FOR_GRIDSEARCH = 120
MIN_SAMPLES_PER_FOLD = 20

CONSERVATIVE_RF_PARAMS = {'n_estimators': 100, 'max_depth': 7, 'min_samples_leaf': 4}
CONSERVATIVE_GB_PARAMS = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8}
CONSERVATIVE_EN_PARAMS = {'alpha': 0.1, 'l1_ratio': 0.5}
CONSERVATIVE_XGB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8}


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

def get_best_model(X_train, y_train, model_type='rf'):
    """Encontra melhores parâmetros com comportamento adaptativo."""
    n_samples = len(X_train)
    
    configs = {
        'rf': (RandomForestRegressor(random_state=42, n_jobs=-1), RF_PARAM_GRID, CONSERVATIVE_RF_PARAMS),
        'gb': (GradientBoostingRegressor(random_state=42), GB_PARAM_GRID, CONSERVATIVE_GB_PARAMS),
        'en': (ElasticNet(random_state=42, max_iter=2000), EN_PARAM_GRID, CONSERVATIVE_EN_PARAMS),
        'xgb': (XGBRegressor(random_state=42, n_jobs=-1, verbosity=0), XGB_PARAM_GRID, CONSERVATIVE_XGB_PARAMS)
    }
    
    base_model, param_grid, conservative_params = configs[model_type]
    
    if n_samples >= MIN_SAMPLES_FOR_GRIDSEARCH:
        n_splits = get_adaptive_cv_splits(n_samples)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    else:
        model = base_model.set_params(**conservative_params)
        model.fit(X_train, y_train)
        return model

# Carregamento e Pré-processamento
def load_and_preprocess_data(filepath, horizon=1):
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
    base_cols = [c for c in df_trans.columns if c not in ['Mes', 'Trimestre']]
    for col in base_cols:
        df_trans[f'{col}_lag3'] = df_trans[col].shift(3)
        df_trans[f'{col}_lag12'] = df_trans[col].shift(12)
    
    df_trans['IBC_MA3'] = df_trans['IBC-Br'].rolling(window=3).mean()
    df_trans['IBC_MA6'] = df_trans['IBC-Br'].rolling(window=6).mean()
    
    # Prepara target e features
    target = df_trans['IBC-Br']
    features = df_trans.drop(columns=['IBC-Br']).shift(horizon)
    
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

def train_ensemble(data):
    """Stacking com 4 modelos base + Ridge meta-learner usando blending OOF."""
    X = data.drop(columns=['IBC-Br'])
    y = data['IBC-Br']
    
    model_types = ['rf', 'gb', 'en', 'xgb']
    
    preds = {m: [] for m in model_types}
    actuals = []
    dates = []
    
    initial_window = 48
    n_folds_blend = 3
    
    print(f"\nTREINAMENTO STACKING ({len(model_types)} modelos base + Ridge meta)")
    
    for i in range(initial_window, len(data)):
        train_idx = range(0, i)
        test_idx = [i]
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for m in model_types:
            if m == 'en':
                model = get_best_model(X_train_scaled, y_train, m)
                preds[m].append(model.predict(X_test_scaled)[0])
            else:
                model = get_best_model(X_train, y_train, m)
                preds[m].append(model.predict(X_test)[0])
        
        actuals.append(y_test.values[0])
        dates.append(y_test.index[0])
    
    # NNLS: pesos não-negativos (zera modelos ruins automaticamente)
    X_meta = np.column_stack([preds[m] for m in model_types if len(preds[m]) > 0])
    y_meta = np.array(actuals)
    
    weights, _ = nnls(X_meta, y_meta)
    weight_sum = weights.sum()
    weights_norm = weights / weight_sum if weight_sum > 0 else np.ones(len(model_types)) / len(model_types)
    
    meta_preds = X_meta @ weights_norm
    
    results = pd.DataFrame({
        'Actual': actuals,
        'RF': preds['rf'],
        'GB': preds['gb'],
        'EN': preds['en'],
        'Ensemble': meta_preds
    }, index=dates)
    
    results['XGB'] = preds['xgb']
    
    return results


def reconstruct_levels(results):
    """Reconstrói níveis do IBC-Br a partir das variações."""
    global ORIGINAL_IBC_BR
    
    model_cols = [c for c in results.columns if c != 'Actual']
    levels = {c: [] for c in ['Actual'] + model_cols}
    valid_dates = []
    
    for date in results.index:
        try:
            idx_loc = ORIGINAL_IBC_BR.index.get_loc(date)
            prev_value = ORIGINAL_IBC_BR.iloc[idx_loc - 1]
        except:
            continue
        
        levels['Actual'].append(ORIGINAL_IBC_BR.loc[date])
        for col in model_cols:
            levels[col].append(prev_value * (1 + results.loc[date, col]))
        valid_dates.append(date)
    
    return pd.DataFrame(levels, index=valid_dates)


def evaluate(results, results_level):
    model_cols = [c for c in results.columns if c != 'Actual']
    
    print(f"\nMÉTRICAS VARIAÇÕES (n={len(results)})")
    for model in model_cols:
        rmse = np.sqrt(mean_squared_error(results['Actual'], results[model]))
        mae = mean_absolute_error(results['Actual'], results[model])
        print(f"  {model:10} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    print(f"\nMÉTRICAS NÍVEIS (n={len(results_level)})")
    for model in model_cols:
        if model in results_level.columns:
            rmse = np.sqrt(mean_squared_error(results_level['Actual'], results_level[model]))
            mae = mean_absolute_error(results_level['Actual'], results_level[model])
            mape = np.mean(np.abs((results_level['Actual'] - results_level[model]) / results_level['Actual'])) * 100
            print(f"  {model:10} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    compare_models(results_level)


def plot_results(results, results_level):
    colors = {'RF': '#3498db', 'GB': '#2ecc71', 'EN': '#f39c12', 'XGB': '#9b59b6', 'Ensemble': '#e74c3c'}
    model_cols = [c for c in results.columns if c != 'Actual']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 14})
    
    metrics = {}
    for m in model_cols:
        if m in results_level.columns:
            metrics[m] = {
                'RMSE': np.sqrt(mean_squared_error(results_level['Actual'], results_level[m])),
                'MAE': mean_absolute_error(results_level['Actual'], results_level[m])
            }
    
    # Gráfico 1: Real vs Ensemble
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(results_level.index, results_level['Actual'], label='Real', color='#2c3e50', linewidth=3)
    ax1.plot(results_level.index, results_level['Ensemble'], label='Stacking Ensemble', color=colors['Ensemble'], linewidth=2.5, linestyle='--')
    ax1.fill_between(results_level.index, results_level['Actual'], results_level['Ensemble'], alpha=0.2, color=colors['Ensemble'])
    ax1.set_title('Previsão IBC-Br: Real vs Stacking Ensemble', fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Índice IBC-Br')
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax1.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('plot_1_real_vs_ensemble.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Gráfico 2: Barras de métricas
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(metrics))
    width = 0.35
    rmse_vals = [metrics[m]['RMSE'] for m in metrics]
    mae_vals = [metrics[m]['MAE'] for m in metrics]
    
    bars1 = ax2.bar(x - width/2, rmse_vals, width, label='RMSE', color='#34495e', edgecolor='white', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, mae_vals, width, label='MAE', color='#95a5a6', edgecolor='white', linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(metrics.keys()), fontsize=13)
    ax2.set_title('Comparativo de Métricas por Modelo', fontweight='bold')
    ax2.set_ylabel('Erro')
    ax2.legend(fontsize=12)
    
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_2_metricas.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Gráfico 3: Erro acumulado
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    for m in model_cols:
        if m in results_level.columns:
            cumulative_error = np.abs(results_level['Actual'] - results_level[m]).cumsum()
            lw = 3 if m == 'Ensemble' else 2
            ax3.plot(results_level.index, cumulative_error, label=m, color=colors.get(m, 'gray'), linewidth=lw)
    ax3.set_title('Erro Absoluto Acumulado ao Longo do Tempo', fontweight='bold')
    ax3.set_xlabel('Data')
    ax3.set_ylabel('Erro Acumulado')
    ax3.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax3.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('plot_3_erro_acumulado.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Gráfico 4: Todos os modelos
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    ax4.plot(results_level.index, results_level['Actual'], label='Real', color='#2c3e50', linewidth=3)
    for m in model_cols:
        if m in results_level.columns:
            lw = 2.5 if m == 'Ensemble' else 1.5
            ls = '--' if m == 'Ensemble' else '-'
            alpha = 1.0 if m == 'Ensemble' else 0.7
            ax4.plot(results_level.index, results_level[m], label=m, color=colors.get(m, 'gray'), linewidth=lw, linestyle=ls, alpha=alpha)
    ax4.set_title('Comparativo: Todos os Modelos vs Real', fontweight='bold')
    ax4.set_xlabel('Data')
    ax4.set_ylabel('Índice IBC-Br')
    ax4.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    ax4.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('plot_4_todos_modelos.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURAÇÕES DE EXECUÇÃO ---
    # Coloque True se quiser treinar os modelos do zero (demora)
    # Coloque False se quiser apenas gerar gráficos e métricas usando os Excels já salvos (rápido)
    REPROCESSAR_TUDO = True 
    
    horizontes = [1, 3, 6, 12]
    
    # Cria pastas para organizar a bagunça
    PASTA_RESULTADOS = "resultados_excel"
    PASTA_GRAFICOS = "graficos_finais"
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"INICIANDO PIPELINE DE PREVISÃO DO PIB (IBC-Br)")
    print(f"Modo de Retreino: {'ATIVADO' if REPROCESSAR_TUDO else 'DESATIVADO (Lendo arquivos salvos)'}")
    print(f"{'#'*60}")

    for h in horizontes:
        print(f"\n>>> PROCESSANDO HORIZONTE H = {h} MESES")
        
        nome_arquivo_excel = os.path.join(PASTA_RESULTADOS, f"resultados_horizonte_{h}m.xlsx")
        
        if REPROCESSAR_TUDO:
            # 1. Carrega e processa (com o lag correto para o horizonte)
            data = load_and_preprocess_data(DATA_PATH, horizon=h)
            
            # 2. Treina os modelos
            results = train_ensemble(data)
            
            # 3. Reconstrói níveis (transforma variação % em índice absoluto)
            results_level = reconstruct_levels(results)
            
            # 4. Gera o Benchmark Naive (Ingênuo)
            # Lógica: A previsão para daqui H meses é igual ao valor de hoje (H meses atrás no target)
            naive_preds = []
            for date in results_level.index:
                try:
                    loc = ORIGINAL_IBC_BR.index.get_loc(date)
                    naive_val = ORIGINAL_IBC_BR.iloc[loc - h]
                    naive_preds.append(naive_val)
                except:
                    naive_preds.append(np.nan)
            results_level['Naive'] = naive_preds
            
            # 5. Salva o Excel na pasta organizada
            results_level.to_excel(nome_arquivo_excel)
            print(f"   -> Dados salvos em: {nome_arquivo_excel}")
            
        else:
            # MODO RÁPIDO: Apenas carrega o arquivo existente
            if os.path.exists(nome_arquivo_excel):
                print(f"   -> Lendo arquivo existente: {nome_arquivo_excel}")
                results_level = pd.read_excel(nome_arquivo_excel, index_col=0, parse_dates=True)
                # Recria o objeto 'results' (variação) apenas para a função evaluate não quebrar
                # (Isso é uma aproximação, mas para gráficos de nível serve)
                results = results_level.pct_change().dropna() 
            else:
                print(f"   [ERRO] Arquivo {nome_arquivo_excel} não encontrado. Mude REPROCESSAR_TUDO = True.")
                continue

        # 6. Avalia e Plota (Isso roda sempre, para você ver os números na tela)
        # O evaluate vai imprimir o RMSE e o Diebold-Mariano
        print(f"\n--- Métricas para Horizonte {h} ---")
        evaluate(results, results_level)
        
        # Gera os gráficos temporários
        plot_results(results, results_level)
        
        # 7. Move e Renomeia as Imagens para a pasta 'graficos'
        imagens_padrao = [
            'plot_1_real_vs_ensemble.png',
            'plot_2_metricas.png',
            'plot_3_erro_acumulado.png',
            'plot_4_todos_modelos.png'
        ]
        
        for img_nome in imagens_padrao:
            if os.path.exists(img_nome):
                # Novo nome ex: graficos_finais/plot_1_real_vs_ensemble_h3.png
                novo_nome = f"{img_nome.replace('.png', '')}_h{h}.png"
                caminho_destino = os.path.join(PASTA_GRAFICOS, novo_nome)
                
                # Move o arquivo (substitui se já existir)
                if os.path.exists(caminho_destino):
                    os.remove(caminho_destino)
                os.rename(img_nome, caminho_destino)
                
    print(f"\n{'-'*60}")
    print(f"CONCLUÍDO! Verifique as pastas '{PASTA_RESULTADOS}' e '{PASTA_GRAFICOS}'")