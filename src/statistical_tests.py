import numpy as np
from scipy import stats
import warnings as _warnings

# Mínimo de amostras para teste DM confiável
MIN_SAMPLES_DM_ROBUST = 50
MIN_SAMPLES_DM_MARGINAL = 30


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2, small_sample_correction: bool = True) -> tuple:
    """
    Teste Diebold-Mariano para comparação de precisão preditiva.
    Inclui correção de pequenas amostras Harvey-Leybourne-Newbold.
    
    Testa se a previsão 1 é significativamente diferente da previsão 2.
    
    Args:
        e1: Erros do modelo 1 (real - previsto).
        e2: Erros do modelo 2 (real - previsto).
        h: Horizonte de previsão (padrão 1 para um passo à frente).
        power: Potência da função de perda (1 para MAE, 2 para MSE).
        small_sample_correction: Aplica correção HLN para amostras pequenas.
    
    Returns:
        Tupla com (estatística DM, p-valor, interpretação).
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    
    # Diferencial de perda
    d = np.abs(e1)**power - np.abs(e2)**power
    
    n = len(d)
    mean_d = np.mean(d)
    
    # Ajuste de autocovariância para previsões h passos à frente
    gamma = []
    for lag in range(h):
        gamma.append(np.cov(d[lag:], d[:n-lag])[0, 1] if n > lag else 0)
    
    # Variância da diferença média (com correção Newey-West para h > 1)
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    
    if var_d <= 0:
        return 0.0, 1.0
    
    # Estatística DM
    dm_stat = mean_d / np.sqrt(var_d)
    
    # Correção Harvey-Leybourne-Newbold para amostras pequenas
    if small_sample_correction and n < MIN_SAMPLES_DM_ROBUST:
        hln_correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        dm_stat_corrected = dm_stat * hln_correction
        
        # Usa distribuição t com n-1 graus de liberdade
        df = n - 1
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df))
        dm_stat = dm_stat_corrected
    else:
        # Normal assintótica para amostras grandes
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def compare_models(results: dict, actual_col: str = 'Actual', models: list = None) -> None:
    """
    Executa testes Diebold-Mariano comparando modelos com o Ensemble.
    
    Args:
        results: DataFrame com 'Actual' e colunas de previsão dos modelos.
        actual_col: Nome da coluna de valores reais.
        models: Lista de modelos a comparar. Se None, usa ['RF', 'GB'].
    """
    import pandas as pd
    
    if models is None:
        models = ['RF', 'GB']
    
    actual = results[actual_col].values
    ensemble = results['Ensemble'].values
    
    e_ensemble = actual - ensemble
    n = len(actual)
    
    print(f"\n{'='*60}")
    print(f"TESTE DIEBOLD-MARIANO (Ensemble vs Modelos Base) | n={n}")
    print("="*60)
    
    for model in models:
        if model not in results.columns:
            continue
        
        e_model = actual - results[model].values
        
        dm_stat, p_value = diebold_mariano_test(e_model, e_ensemble, h=1, power=2)
        
        print(f"\n{model} vs Ensemble:")
        print(f"  Estatística DM: {dm_stat:.4f}")
        print(f"  P-valor:        {p_value:.4f}")
