"""
Kenya Food Price Inflation Tracker - Data Manager
Centralized data loading and caching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent / "data"
CLEAN_DIR = DATA_DIR / "clean"
MODELS_DIR = Path(__file__).parent.parent / "models" / "trained"


@dataclass
class DataLoadResult:
    staples: Optional[pd.DataFrame] = None
    monthly: Optional[pd.DataFrame] = None
    maize_features: Optional[pd.DataFrame] = None
    prophet_fc: Optional[pd.DataFrame] = None
    sarima_fc: Optional[pd.DataFrame] = None
    model_comparison: Optional[pd.DataFrame] = None
    error: Optional[str] = None


@lru_cache(maxsize=1)
def load_all_data() -> DataLoadResult:
    """Load all datasets with intelligent error handling"""
    result = DataLoadResult()
    
    try:
        staples_path = CLEAN_DIR / "wfp_core_staples.csv"
        if staples_path.exists():
            result.staples = pd.read_csv(staples_path, parse_dates=['date'])
            result.staples['cm_name'] = result.staples['cm_name'].str.strip()
    except Exception as e:
        result.error = f"Failed to load staples data: {e}"
    
    try:
        monthly_path = CLEAN_DIR / "wfp_monthly_avg.csv"
        if monthly_path.exists():
            result.monthly = pd.read_csv(monthly_path, parse_dates=['date'])
    except Exception:
        pass
    
    try:
        maize_path = CLEAN_DIR / "maize_features.csv"
        if maize_path.exists():
            result.maize_features = pd.read_csv(maize_path, parse_dates=['date'])
    except Exception:
        pass
    
    try:
        prophet_path = MODELS_DIR / "prophet_forecast.csv"
        if prophet_path.exists():
            result.prophet_fc = pd.read_csv(prophet_path, parse_dates=['ds'])
    except Exception:
        pass
    
    try:
        sarima_path = MODELS_DIR / "sarima_forecast.csv"
        if sarima_path.exists():
            result.sarima_fc = pd.read_csv(sarima_path, parse_dates=['ds'])
    except Exception:
        pass
    
    try:
        comparison_path = MODELS_DIR / "model_comparison.csv"
        if comparison_path.exists():
            result.model_comparison = pd.read_csv(comparison_path, index_col=0)
    except Exception:
        pass
    
    return result


def get_commodities(df: pd.DataFrame) -> list:
    """Get sorted list of unique commodities"""
    return sorted(df['cm_name'].unique().tolist()) if df is not None else []


def get_regions(df: pd.DataFrame) -> list:
    """Get sorted list of unique regions with 'All' option"""
    return ['All Regions'] + sorted(df['adm1_name'].unique().tolist()) if df is not None else []


def get_markets(df: pd.DataFrame) -> list:
    """Get sorted list of unique markets"""
    return sorted(df['mkt_name'].unique().tolist()) if df is not None else []


def filter_data(
    df: pd.DataFrame,
    commodity: Optional[str] = None,
    region: Optional[str] = None,
    year_range: Optional[tuple] = None
) -> pd.DataFrame:
    """Filter dataframe by various criteria"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    if commodity:
        filtered = filtered[filtered['cm_name'] == commodity]
    
    if region and region != 'All Regions':
        filtered = filtered[filtered['adm1_name'] == region]
    
    if year_range:
        filtered = filtered[
            (filtered['date'].dt.year >= year_range[0]) &
            (filtered['date'].dt.year <= year_range[1])
        ]
    
    return filtered


def validate_data(df: pd.DataFrame, min_records: int = 10) -> bool:
    """Validate dataframe has minimum required records"""
    if df is None:
        return False
    return len(df) >= min_records


def get_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate data quality metrics"""
    if df is None or len(df) == 0:
        return {'completeness': 0, 'records': 0}
    
    completeness = 1 - (df['mp_price'].isna().sum() / len(df))
    
    return {
        'completeness': completeness * 100,
        'records': len(df),
        'unique_commodities': df['cm_name'].nunique(),
        'unique_regions': df['adm1_name'].nunique(),
        'date_range': (df['date'].min(), df['date'].max())
    }


if __name__ == "__main__":
    data = load_all_data()
    if data.staples is not None:
        print(f"Loaded {len(data.staples)} records")
        print(f"Commodities: {len(get_commodities(data.staples))}")
        print(f"Regions: {len(get_regions(data.staples))}")
    else:
        print("Error loading data") if data.error else print("No data available")