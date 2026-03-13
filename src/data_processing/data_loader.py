"""
Kenya Food Price Inflation Tracker
Data Loading Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and prepare food price data for analysis"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.clean_dir = self.data_dir / "clean"
        
    def load_wfp_raw(self, filename: str = "wfp_food_prices_kenya_full.csv") -> pd.DataFrame:
        """Load raw WFP Kenya food price data"""
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} records from {filename}")
        return df
    
    def load_wfp_clean(self, filename: str = "wfp_kenya_clean.csv") -> pd.DataFrame:
        """Load cleaned WFP data"""
        filepath = self.clean_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Clean data not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['date'])
        print(f"✓ Loaded {len(df):,} clean records from {filename}")
        return df
    
    def load_staples(self) -> pd.DataFrame:
        """Load core staple commodities dataset"""
        return self.load_wfp_clean("wfp_core_staples.csv")
    
    def load_monthly_avg(self) -> pd.DataFrame:
        """Load monthly aggregated prices"""
        filepath = self.clean_dir / "wfp_monthly_avg.csv"
        df = pd.read_csv(filepath, parse_dates=['date'])
        print(f"✓ Loaded {len(df):,} monthly records")
        return df
    
    def load_maize_features(self, ml_ready: bool = True) -> pd.DataFrame:
        """Load feature-engineered maize dataset"""
        filename = "maize_features_ml.csv" if ml_ready else "maize_features.csv"
        filepath = self.clean_dir / filename
        df = pd.read_csv(filepath, parse_dates=['date'])
        print(f"✓ Loaded {len(df):,} maize records with features")
        return df
    
    def filter_by_commodity(self, df: pd.DataFrame, commodity: str) -> pd.DataFrame:
        """Filter dataset by commodity name"""
        # Use cm_name column (actual column in WFP data)
        filtered = df[df['cm_name'] == commodity].copy()
        print(f"✓ Filtered to {len(filtered):,} records for {commodity}")
        return filtered
    
    def filter_by_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Filter dataset by region"""
        filtered = df[df['adm1_name'] == region].copy()
        print(f"✓ Filtered to {len(filtered):,} records for {region}")
        return filtered
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter dataset by date range"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        print(f"✓ Filtered to {len(filtered):,} records from {start_date} to {end_date}")
        return filtered
    
    def get_national_average(self, df: pd.DataFrame, commodity: str, 
                            date_col: str = 'date', price_col: str = 'mp_price') -> pd.DataFrame:
        """Calculate national average price for a commodity"""
        commodity_data = self.filter_by_commodity(df, commodity)
        national_avg = commodity_data.groupby(date_col)[price_col].mean().reset_index()
        national_avg.columns = ['date', 'price']
        print(f"✓ Calculated national average: {len(national_avg)} time periods")
        return national_avg
    
    def get_regional_averages(self, df: pd.DataFrame, commodity: str) -> pd.DataFrame:
        """Calculate regional average prices for a commodity"""
        commodity_data = self.filter_by_commodity(df, commodity)
        regional = commodity_data.groupby('adm1_name')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
        regional.columns = ['region', 'avg_price', 'std_price', 'observations']
        regional = regional.sort_values('avg_price', ascending=False)
        print(f"✓ Calculated averages for {len(regional)} regions")
        return regional
    
    def get_commodity_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique commodities"""
        commodities = sorted(df['cm_name'].unique())
        return commodities
    
    def get_region_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique regions"""
        regions = sorted(df['adm1_name'].unique())
        return regions
    
    def get_market_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique markets"""
        markets = sorted(df['mkt_name'].unique())
        return markets
    
    def get_date_range(self, df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get min and max dates in dataset"""
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].min(), df['date'].max()
    
    def summarize_dataset(self, df: pd.DataFrame) -> dict:
        """Get comprehensive dataset summary"""
        df['date'] = pd.to_datetime(df['date'])
        
        summary = {
            'total_records': len(df),
            'unique_commodities': df['cm_name'].nunique(),
            'unique_markets': df['mkt_name'].nunique(),
            'unique_regions': df['adm1_name'].nunique(),
            'date_range': (df['date'].min().strftime('%Y-%m-%d'), 
                          df['date'].max().strftime('%Y-%m-%d')),
            'years_covered': df['date'].dt.year.nunique(),
            'avg_price': df['mp_price'].mean(),
            'median_price': df['mp_price'].median(),
            'price_std': df['mp_price'].std()
        }
        
        return summary


# Convenience function
def load_data(data_type: str = "staples") -> pd.DataFrame:
    """
    Quick data loading function
    
    Args:
        data_type: One of 'raw', 'clean', 'staples', 'monthly', 'maize'
    
    Returns:
        DataFrame with requested data
    """
    loader = DataLoader()
    
    if data_type == "raw":
        return loader.load_wfp_raw()
    elif data_type == "clean":
        return loader.load_wfp_clean()
    elif data_type == "staples":
        return loader.load_staples()
    elif data_type == "monthly":
        return loader.load_monthly_avg()
    elif data_type == "maize":
        return loader.load_maize_features()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    
    print("\n=== Testing Data Loader ===\n")
    
    # Load staples
    staples = loader.load_staples()
    print(f"\nStaples dataset shape: {staples.shape}")
    
    # Get summary
    summary = loader.summarize_dataset(staples)
    print("\n=== Dataset Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Get commodity list
    commodities = loader.get_commodity_list(staples)
    print(f"\n=== Commodities ({len(commodities)}) ===")
    for commodity in commodities:
        print(f"  - {commodity}")
    
    print("\n✓ Data loader tests completed successfully")
