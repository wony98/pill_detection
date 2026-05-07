"""
eda_common.py — 프로젝트 간 재사용하는 EDA 공통 함수

사용법:
    from utils.eda_common import quick_audit, describe_plus, null_report
    
    # DataFrame 빠른 감사
    report = quick_audit(df, target='label')
    print(report)
"""
from typing import Optional, Union, Dict, List, Any
import warnings

import numpy as np
import pandas as pd


# ===========================================================================
# L0. Sanity Check 함수들
# ===========================================================================

def quick_audit(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """
    DataFrame의 L0 Sanity + L1 Target 빠른 감사.
    
    Returns:
        컬럼별 요약 정보
    """
    audit = pd.DataFrame({
        'dtype': df.dtypes,
        'n_missing': df.isnull().sum(),
        'pct_missing': (df.isnull().mean() * 100).round(2),
        'n_unique': df.nunique(),
        'pct_unique': (df.nunique() / len(df) * 100).round(2),
        'sample_values': [df[col].dropna().unique()[:3].tolist() for col in df.columns],
    })
    
    # 의심 플래그
    audit['flag'] = ''
    audit.loc[audit['n_unique'] == 1, 'flag'] += 'constant '
    audit.loc[audit['pct_missing'] > 50, 'flag'] += 'high_missing '
    audit.loc[audit['pct_unique'] > 95, 'flag'] += 'high_cardinality '
    audit.loc[(audit['pct_unique'] < 1) & (audit['dtype'] != 'object'), 'flag'] += 'low_variance '
    
    if target and target in df.columns:
        print(f"\n=== Target: {target} ===")
        print(f"  dtype: {df[target].dtype}")
        print(f"  n_unique: {df[target].nunique()}")
        if df[target].dtype in ['int64', 'float64']:
            print(f"  mean/std: {df[target].mean():.4f} / {df[target].std():.4f}")
        else:
            print(f"  distribution:\n{df[target].value_counts(normalize=True).head(10)}")
    
    return audit


def null_report(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """결측치 리포트. threshold 이상 결측인 컬럼만."""
    null_pct = df.isnull().mean()
    result = (
        null_pct[null_pct > threshold]
        .sort_values(ascending=False)
        .to_frame('pct_missing')
    )
    result['n_missing'] = (result['pct_missing'] * len(df)).astype(int)
    return result


def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict:
    """중복 탐지. 완전 중복 + 서브셋 기반 중복."""
    result = {
        'full_duplicates': df.duplicated().sum(),
        'total_rows': len(df),
    }
    if subset:
        result['subset_duplicates'] = df.duplicated(subset=subset).sum()
        result['subset'] = subset
    
    result['pct_duplicates'] = result['full_duplicates'] / len(df) * 100
    return result


def check_column_types(df: pd.DataFrame, expected: Dict[str, str]) -> List[str]:
    """기대하는 dtype과 실제 dtype 비교. 불일치 리포트."""
    issues = []
    for col, expected_dtype in expected.items():
        if col not in df.columns:
            issues.append(f"MISSING: {col} (expected {expected_dtype})")
            continue
        actual = str(df[col].dtype)
        if expected_dtype not in actual:
            issues.append(f"TYPE_MISMATCH: {col} = {actual} (expected {expected_dtype})")
    return issues


# ===========================================================================
# L1. Target 분석 함수들
# ===========================================================================

def target_summary(y: pd.Series, problem_type: str = 'auto') -> Dict:
    """타겟 변수 종합 요약."""
    summary = {
        'n_samples': len(y),
        'n_missing': y.isnull().sum(),
    }
    
    if problem_type == 'auto':
        if y.dtype in ['object', 'category'] or y.nunique() < 20:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
    
    summary['problem_type'] = problem_type
    
    if problem_type == 'classification':
        counts = y.value_counts()
        summary['n_classes'] = len(counts)
        summary['class_distribution'] = counts.to_dict()
        summary['majority_class'] = counts.index[0]
        summary['majority_pct'] = counts.iloc[0] / len(y) * 100
        summary['minority_class'] = counts.index[-1]
        summary['minority_pct'] = counts.iloc[-1] / len(y) * 100
        summary['imbalance_ratio'] = counts.iloc[0] / counts.iloc[-1]
        summary['baseline_accuracy'] = counts.iloc[0] / len(y)
    else:
        summary['mean'] = y.mean()
        summary['std'] = y.std()
        summary['median'] = y.median()
        summary['skew'] = y.skew()
        summary['kurtosis'] = y.kurt()
        summary['min'] = y.min()
        summary['max'] = y.max()
        summary['baseline_mae'] = (y - y.median()).abs().mean()
        summary['baseline_rmse'] = np.sqrt(((y - y.mean()) ** 2).mean())
    
    return summary


def target_drift_over_time(df: pd.DataFrame, target: str, time_col: str, 
                            freq: str = 'M') -> pd.DataFrame:
    """시간에 따른 타겟 분포 변화. 시계열/날짜 데이터에서 drift 감지."""
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy['period'] = df_copy[time_col].dt.to_period(freq)
    
    if df[target].dtype in ['object', 'category'] or df[target].nunique() < 20:
        # 분류: 시간별 클래스 비율
        result = df_copy.groupby('period')[target].value_counts(normalize=True).unstack(fill_value=0)
    else:
        # 회귀: 시간별 평균/표준편차
        result = df_copy.groupby('period')[target].agg(['mean', 'std', 'count'])
    
    return result


# ===========================================================================
# L2. Input 분석 함수들
# ===========================================================================

def describe_plus(df: pd.DataFrame, percentiles: List[float] = None) -> pd.DataFrame:
    """pandas describe() 확장판. skew, kurtosis, outlier ratio 포함."""
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    numeric = df.select_dtypes(include=[np.number])
    base = numeric.describe(percentiles=percentiles).T
    
    # 추가 통계
    base['skew'] = numeric.skew()
    base['kurtosis'] = numeric.kurt()
    
    # IQR 이상치 비율
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    base['pct_outlier_iqr'] = (
        ((numeric < lower) | (numeric > upper)).mean() * 100
    ).round(2)
    
    return base


def categorical_cardinality(df: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    """범주형 컬럼의 cardinality 분석."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    result = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
        top_freq = (df[col] == top_value).mean() if top_value else 0
        
        result.append({
            'column': col,
            'n_unique': n_unique,
            'is_high_cardinality': n_unique > threshold,
            'top_value': top_value,
            'top_value_pct': top_freq * 100,
            'rare_values_pct': (df[col].value_counts(normalize=True) < 0.01).sum(),
        })
    
    return pd.DataFrame(result).sort_values('n_unique', ascending=False)


def correlation_with_target(df: pd.DataFrame, target: str, 
                             method: str = 'pearson') -> pd.Series:
    """피처별 타겟과의 상관관계."""
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        warnings.warn(f"{target} not numeric. Skipping correlation.")
        return pd.Series(dtype=float)
    
    corrs = numeric.corr(method=method)[target].drop(target)
    return corrs.sort_values(key=abs, ascending=False)


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """상관계수 threshold 이상인 피처 쌍 탐지 (다중공선성)."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    
    # 상삼각만 (중복 제거)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    pairs = []
    for col in upper.columns:
        high = upper[col][upper[col] > threshold]
        for other_col, val in high.items():
            pairs.append({'feature_1': other_col, 'feature_2': col, 'correlation': val})
    
    return pd.DataFrame(pairs).sort_values('correlation', ascending=False)


# ===========================================================================
# L3. Input × Target 분석 함수들
# ===========================================================================

def target_mean_by_quantile(df: pd.DataFrame, feature: str, target: str, 
                             n_bins: int = 10) -> pd.DataFrame:
    """수치형 피처의 분위수별 타겟 평균. 단조성/비선형성 포착."""
    df_copy = df[[feature, target]].dropna().copy()
    df_copy['bin'] = pd.qcut(df_copy[feature], n_bins, duplicates='drop', labels=False)
    
    result = df_copy.groupby('bin').agg(
        feature_min=(feature, 'min'),
        feature_max=(feature, 'max'),
        feature_mean=(feature, 'mean'),
        target_mean=(target, 'mean'),
        target_std=(target, 'std'),
        count=(target, 'count'),
    ).round(4)
    
    return result


def target_by_category(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """범주형 피처별 타겟 분포."""
    result = df.groupby(feature)[target].agg(
        count='count',
        target_mean='mean',
        target_std='std',
    ).sort_values('count', ascending=False)
    
    result['pct'] = (result['count'] / len(df) * 100).round(2)
    return result


# ===========================================================================
# L4. Split & Leakage 체크
# ===========================================================================

def check_split_distribution(train: pd.DataFrame, val: pd.DataFrame, 
                              target: str) -> pd.DataFrame:
    """Train/Val 타겟 분포 비교. 크게 다르면 split 이상."""
    train_dist = train[target].value_counts(normalize=True).sort_index()
    val_dist = val[target].value_counts(normalize=True).sort_index()
    
    comparison = pd.DataFrame({
        'train_pct': train_dist,
        'val_pct': val_dist,
    }).fillna(0)
    comparison['diff'] = (comparison['train_pct'] - comparison['val_pct']).abs()
    comparison['ratio'] = (comparison['val_pct'] / (comparison['train_pct'] + 1e-10)).round(3)
    
    return comparison.sort_values('diff', ascending=False)


def check_group_leakage(train: pd.DataFrame, val: pd.DataFrame, 
                        group_col: str) -> Dict:
    """Group split이 제대로 됐는지 검증. 겹침 0이어야 함."""
    train_groups = set(train[group_col])
    val_groups = set(val[group_col])
    
    return {
        'n_train_groups': len(train_groups),
        'n_val_groups': len(val_groups),
        'n_overlap': len(train_groups & val_groups),
        'leakage_detected': len(train_groups & val_groups) > 0,
    }


def check_feature_drift(train: pd.DataFrame, test: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> pd.DataFrame:
    """Train/Test 피처 분포 차이 (KS test 기반)."""
    from scipy.stats import ks_2samp
    
    if features is None:
        features = train.select_dtypes(include=[np.number]).columns.tolist()
    
    result = []
    for col in features:
        if col not in test.columns:
            continue
        train_vals = train[col].dropna().values
        test_vals = test[col].dropna().values
        
        if len(train_vals) == 0 or len(test_vals) == 0:
            continue
        
        ks_stat, p_value = ks_2samp(train_vals, test_vals)
        result.append({
            'feature': col,
            'ks_statistic': round(ks_stat, 4),
            'p_value': round(p_value, 4),
            'drift_detected': p_value < 0.05,
            'train_mean': round(train_vals.mean(), 4),
            'test_mean': round(test_vals.mean(), 4),
        })
    
    return pd.DataFrame(result).sort_values('ks_statistic', ascending=False)


# ===========================================================================
# 유틸리티
# ===========================================================================

def memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 메모리 사용량 컬럼별 분석."""
    mem = df.memory_usage(deep=True) / 1024 / 1024
    return pd.DataFrame({
        'dtype': df.dtypes,
        'memory_mb': mem[1:].round(2),
        'pct_of_total': (mem[1:] / mem[1:].sum() * 100).round(2),
    }).sort_values('memory_mb', ascending=False)


def suggest_dtype_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """메모리 최적화 위한 dtype 변경 제안."""
    suggestions = []
    for col in df.columns:
        current = df[col].dtype
        if str(current).startswith('int64'):
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 255:
                    suggestions.append((col, str(current), 'uint8'))
                elif col_max < 65535:
                    suggestions.append((col, str(current), 'uint16'))
                elif col_max < 4294967295:
                    suggestions.append((col, str(current), 'uint32'))
            else:
                if col_min > -128 and col_max < 127:
                    suggestions.append((col, str(current), 'int8'))
                elif col_min > -32768 and col_max < 32767:
                    suggestions.append((col, str(current), 'int16'))
        elif str(current) == 'float64':
            suggestions.append((col, str(current), 'float32'))
        elif str(current) == 'object':
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.5:
                suggestions.append((col, str(current), 'category'))
    
    return pd.DataFrame(suggestions, columns=['column', 'current', 'suggested'])
