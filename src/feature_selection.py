import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings as _warnings

# Limiares adaptativos baseados no tamanho da amostra
MIN_OBS_ADF_ROBUST = 36   # 3 anos de dados mensais para ADF robusto
MIN_OBS_ADF_RELAXED = 24  # 2 anos de dados mensais para evitar remoção de features com poucas observações


def test_stationarity(series: pd.Series, alpha: float = 0.05, adaptive: bool = True) -> bool:
    """
    Aplica o teste ADF para verificar estacionariedade.
    Usa limiares adaptativos baseados no tamanho amostral.
    
    Args:
        series: Série temporal a ser testada.
        alpha: Nível de significância (padrão 5%).
        adaptive: Se True, relaxa alpha para amostras marginais.
    
    Returns:
        True se a série é estacionária (rejeita raiz unitária), False caso contrário.
    """
    try:
        clean_series = series.dropna()
        n_obs = len(clean_series)
        
        if n_obs < MIN_OBS_ADF_RELAXED:
            return False  # Observações insuficientes
        
        # Ajusta alpha para amostras marginais (regularização)
        effective_alpha = alpha
        if adaptive and n_obs < MIN_OBS_ADF_ROBUST:
            effective_alpha = alpha * 1.5  # Ex: 0.05 -> 0.075
        
        # Maxlag adaptativo baseado no tamanho amostral
        maxlag = min(12, int(n_obs / 4))
        
        result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')
        p_value = result[1]
        return p_value < effective_alpha
    except Exception:
        return False


def remove_nonstationary(df: pd.DataFrame, alpha: float = 0.05, verbose: bool = True, adaptive: bool = True) -> pd.DataFrame:
    """
    Remove colunas que falham no teste ADF de estacionariedade.
    Usa limiares adaptativos para amostras pequenas.
    
    Args:
        df: DataFrame com colunas de séries temporais.
        alpha: Nível de significância para o teste ADF.
        verbose: Exibe colunas removidas.
        adaptive: Usa níveis de significância adaptativos para amostras pequenas.
    
    Returns:
        DataFrame apenas com colunas estacionárias.
    """
    stationary_cols = []
    removed_cols = []
    marginal_count = 0
    
    n_obs = len(df)
    if verbose and n_obs < MIN_OBS_ADF_ROBUST:
        print(f"[Seleção] Aviso: {n_obs} obs. abaixo do limiar robusto ({MIN_OBS_ADF_ROBUST}). Modo adaptativo ativado.")
    
    for col in df.columns:
        if test_stationarity(df[col], alpha, adaptive=adaptive):
            stationary_cols.append(col)
            if adaptive and len(df[col].dropna()) < MIN_OBS_ADF_ROBUST:
                marginal_count += 1
        else:
            removed_cols.append(col)
    
    if verbose:
        if removed_cols:
            lista = removed_cols[:5]
            sufixo = '...' if len(removed_cols) > 5 else ''
            print(f"[Seleção] {len(removed_cols)} colunas não-estacionárias removidas: {lista}{sufixo}")
        if marginal_count > 0:
            print(f"[Seleção] {marginal_count} colunas passaram com critério relaxado (modo adaptativo).")
    
    return df[stationary_cols]


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9, verbose: bool = True) -> pd.DataFrame:
    """
    Remove features altamente correlacionadas para reduzir redundância.
    Mantém a primeira coluna em cada par correlacionado.
    
    Args:
        df: DataFrame com features.
        threshold: Limiar de correlação (padrão 0.9).
        verbose: Exibe colunas removidas.
    
    Returns:
        DataFrame com colinearidade reduzida.
    """
    corr_matrix = df.corr().abs()
    
    # Triângulo superior da matriz de correlação
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Colunas com correlação acima do limiar
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    
    if verbose and to_drop:
        print(f"[Seleção] {len(to_drop)} colunas altamente correlacionadas removidas: {to_drop[:5]}...")
    
    return df.drop(columns=to_drop)


def select_features(df: pd.DataFrame, 
                    target_col: str = 'IBC-Br',
                    adf_alpha: float = 0.05, 
                    corr_threshold: float = 0.9,
                    remove_correlated: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de seleção de features.
    
    1. Separa target das features.
    2. Remove features não-estacionárias.
    3. (Opcional) Remove features altamente correlacionadas.
    4. Recombina com target.
    
    Args:
        df: DataFrame com target e features.
        target_col: Nome da coluna target.
        adf_alpha: Nível de significância para teste de estacionariedade.
        corr_threshold: Limiar para remoção de correlação.
        remove_correlated: Se False, pula remoção de correlação (melhor para modelos de árvore).
        verbose: Exibe progresso.
    
    Returns:
        DataFrame com features selecionadas + target.
    """
    if verbose:
        print(f"\n[Seleção] Iniciando com {df.shape[1]} colunas...")
    
    target = df[[target_col]]
    features = df.drop(columns=[target_col])
    
    # Etapa 1: Verificação de estacionariedade
    features = remove_nonstationary(features, alpha=adf_alpha, verbose=verbose)
    
    # Etapa 2: Remoção de correlação (desativado para modelos de árvore)
    if remove_correlated:
        features = remove_highly_correlated(features, threshold=corr_threshold, verbose=verbose)
    elif verbose:
        print(f"[Seleção] Correlação ignorada (modelos de árvore lidam bem com multicolinearidade)")
    
    result = pd.concat([target, features], axis=1)
    
    if verbose:
        print(f"[Seleção] Final: {result.shape[1]} colunas restantes.\n")
    
    return result
