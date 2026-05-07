"""
viz_helpers.py — 시각화 헬퍼. 일관된 스타일로 EDA 플롯 생성.

사용법:
    from utils.viz_helpers import setup_korean_font, plot_distribution, plot_target_by_feature
    
    setup_korean_font()  # 한글 깨짐 방지
    plot_distribution(df, 'age')
"""
from typing import Optional, Union, List
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ===========================================================================
# 스타일 세팅
# ===========================================================================

def setup_korean_font():
    """한글 폰트 자동 설정 (환경별)."""
    import platform
    system = platform.system()
    
    candidates = []
    if system == 'Darwin':  # macOS
        candidates = ['AppleGothic', 'Apple SD Gothic Neo']
    elif system == 'Windows':
        candidates = ['Malgun Gothic', 'NanumGothic']
    else:  # Linux (Colab 포함)
        candidates = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 한글 폰트 설정: {font}")
            return
    
    warnings.warn("한글 폰트를 못 찾았습니다. Colab이면: !apt install -y fonts-nanum")


def setup_plot_style():
    """일관된 플롯 스타일 적용."""
    try:
        import seaborn as sns
        sns.set_theme(style='whitegrid', context='notebook')
    except ImportError:
        pass
    
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 120,
        'savefig.bbox': 'tight',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


# ===========================================================================
# 분포 플롯
# ===========================================================================

def plot_distribution(df: pd.DataFrame, col: str, 
                       log_scale: bool = False,
                       bins: int = 50,
                       save_path: Optional[str] = None):
    """단일 컬럼 분포 플롯 (히스토그램 + KDE)."""
    data = df[col].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 히스토그램
    axes[0].hist(data, bins=bins, alpha=0.7, edgecolor='black')
    axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'mean={data.mean():.2f}')
    axes[0].axvline(data.median(), color='green', linestyle='--', label=f'median={data.median():.2f}')
    axes[0].set_title(f'{col} distribution')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Count')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(data, vert=False)
    axes[1].set_title(f'{col} boxplot')
    axes[1].set_xlabel(col)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_target_distribution(y: pd.Series, save_path: Optional[str] = None):
    """타겟 변수 분포 (분류/회귀 자동 판별)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if y.dtype in ['object', 'category'] or y.nunique() < 20:
        # 분류
        counts = y.value_counts()
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_title(f'Target distribution (n_classes={len(counts)})')
        ax.set_ylabel('Count')
        
        # 불균형 비율
        ratio = counts.max() / counts.min()
        ax.text(0.02, 0.98, f'imbalance: {ratio:.1f}x',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # 회귀
        ax.hist(y.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_title('Target distribution')
        ax.set_xlabel('Target value')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_class_longtail(counts: pd.Series, top_n: int = 40,
                         save_path: Optional[str] = None):
    """Long-tail 분포 시각화 (클래스 불균형용)."""
    sorted_counts = counts.sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 전체 랭크 (log scale)
    axes[0].bar(range(len(sorted_counts)), sorted_counts.values)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Class rank')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title(f'Class distribution (all {len(sorted_counts)} classes)')
    
    # Top N 상세
    top = sorted_counts.head(top_n)
    axes[1].barh(range(len(top)), top.values)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Count')
    axes[1].set_title(f'Top {top_n} classes')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ===========================================================================
# Input × Target 플롯
# ===========================================================================

def plot_target_by_feature(df: pd.DataFrame, feature: str, target: str,
                             n_bins: int = 10, save_path: Optional[str] = None):
    """수치형 피처의 분위수별 타겟 평균 플롯 (단조성 확인)."""
    data = df[[feature, target]].dropna().copy()
    data['bin'] = pd.qcut(data[feature], n_bins, duplicates='drop', labels=False)
    
    grouped = data.groupby('bin').agg(
        feature_mean=(feature, 'mean'),
        target_mean=(target, 'mean'),
        target_std=(target, 'std'),
        count=(target, 'count'),
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(
        grouped['feature_mean'], grouped['target_mean'],
        yerr=grouped['target_std'] / np.sqrt(grouped['count']),
        marker='o', capsize=5, linewidth=2
    )
    
    # 전체 평균 기준선
    overall_mean = data[target].mean()
    ax.axhline(overall_mean, color='red', linestyle='--', alpha=0.5,
               label=f'Overall mean={overall_mean:.3f}')
    
    ax.set_xlabel(feature)
    ax.set_ylabel(f'{target} (mean)')
    ax.set_title(f'{target} vs {feature} (n_bins={n_bins})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
    """상관관계 히트맵."""
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn 필요")
    
    if features is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        numeric = df[features]
    
    corr = numeric.corr()
    
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.5), max(6, len(corr) * 0.5)))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_train_test_distribution(train: pd.DataFrame, test: pd.DataFrame,
                                   col: str, save_path: Optional[str] = None):
    """Train/Test 분포 비교 (covariate shift 감지)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    train_data = train[col].dropna()
    test_data = test[col].dropna()
    
    if train[col].dtype in [np.float64, np.int64] and train[col].nunique() > 20:
        # 연속형
        bins = np.linspace(
            min(train_data.min(), test_data.min()),
            max(train_data.max(), test_data.max()),
            50
        )
        ax.hist(train_data, bins=bins, alpha=0.5, label=f'train (n={len(train_data)})', density=True)
        ax.hist(test_data, bins=bins, alpha=0.5, label=f'test (n={len(test_data)})', density=True)
    else:
        # 범주형
        train_pct = train[col].value_counts(normalize=True)
        test_pct = test[col].value_counts(normalize=True)
        categories = sorted(set(train_pct.index) | set(test_pct.index))
        
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w/2, [train_pct.get(c, 0) for c in categories], w, label='train')
        ax.bar(x + w/2, [test_pct.get(c, 0) for c in categories], w, label='test')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
    
    ax.set_title(f'Train vs Test: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Density' if train[col].nunique() > 20 else 'Proportion')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ===========================================================================
# Missing 시각화
# ===========================================================================

def plot_missing_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """결측 패턴 매트릭스 (행 × 컬럼)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    missing_matrix = df.isnull().astype(int)
    ax.imshow(missing_matrix.T, aspect='auto', cmap='gray_r', interpolation='nearest')
    
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel('Row index')
    ax.set_title('Missing pattern (black = missing)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_missing_bar(df: pd.DataFrame, save_path: Optional[str] = None):
    """결측률 막대그래프."""
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) == 0:
        print("결측치 없음")
        return None
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.3)))
    ax.barh(range(len(missing_pct)), missing_pct.values)
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index)
    ax.set_xlabel('% Missing')
    ax.set_title('Missing value proportion by column')
    
    # 위험 구간 표시
    ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
