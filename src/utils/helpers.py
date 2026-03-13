"""
Kenya Food Price Inflation Tracker
Utility Helper Functions
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_inflation_rate(current_price: float, previous_price: float) -> float:
    """
    Calculate inflation rate between two prices
    
    Args:
        current_price: Current price
        previous_price: Previous period price
    
    Returns:
        Inflation rate as percentage
    """
    if previous_price == 0:
        return np.nan
    return ((current_price - previous_price) / previous_price) * 100


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Args:
        start_value: Starting value
        end_value: Ending value
        periods: Number of periods (years)
    
    Returns:
        CAGR as percentage
    """
    if start_value == 0 or periods == 0:
        return np.nan
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


def calculate_coefficient_of_variation(data: pd.Series) -> float:
    """
    Calculate coefficient of variation (CV)
    
    Args:
        data: Series of values
    
    Returns:
        CV as percentage
    """
    mean_val = data.mean()
    if mean_val == 0:
        return np.nan
    return (data.std() / mean_val) * 100


def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (1.5 = moderate, 3.0 = extreme)
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series
    
    Args:
        df: DataFrame (must be sorted by time)
        column: Column to create lags for
        lags: List of lag periods (e.g., [1, 3, 6, 12])
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Args:
        df: DataFrame (must be sorted by time)
        column: Column to calculate rolling stats for
        windows: List of window sizes (e.g., [3, 6, 12])
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
    return df


def get_seasonal_month(month: int) -> str:
    """
    Classify month into Kenyan agricultural season
    
    Args:
        month: Month number (1-12)
    
    Returns:
        Season label
    """
    if month in [4, 5]:
        return 'harvest_long'
    elif month in [10, 11]:
        return 'harvest_short'
    elif month in [1, 2, 3]:
        return 'lean'
    else:
        return 'normal'


def encode_cyclical_feature(values: pd.Series, max_value: int) -> Tuple[pd.Series, pd.Series]:
    """
    Encode cyclical features (like month) using sin/cos
    
    Args:
        values: Series of values (e.g., month numbers)
        max_value: Maximum value in cycle (e.g., 12 for months)
    
    Returns:
        Tuple of (sin_encoded, cos_encoded)
    """
    sin_encoded = np.sin(2 * np.pi * values / max_value)
    cos_encoded = np.cos(2 * np.pi * values / max_value)
    return sin_encoded, cos_encoded


def calculate_price_volatility(prices: pd.Series, window: int = 12) -> pd.Series:
    """
    Calculate rolling price volatility (standard deviation)
    
    Args:
        prices: Series of prices
        window: Rolling window size
    
    Returns:
        Series of volatility values
    """
    return prices.rolling(window=window).std()


def calculate_price_momentum(prices: pd.Series, window: int = 3) -> pd.Series:
    """
    Calculate price momentum (rate of change)
    
    Args:
        prices: Series of prices
        window: Window for momentum calculation
    
    Returns:
        Series of momentum values
    """
    return prices.pct_change(periods=window) * 100


def format_currency(amount: float, currency: str = "KES", decimals: int = 2) -> str:
    """
    Format number as currency
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimals: Decimal places
    
    Returns:
        Formatted string
    """
    return f"{amount:,.{decimals}f} {currency}"


def format_percentage(value: float, decimals: int = 1, with_sign: bool = True) -> str:
    """
    Format number as percentage
    
    Args:
        value: Value to format
        decimals: Decimal places
        with_sign: Include +/- sign
    
    Returns:
        Formatted string
    """
    if with_sign:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def split_time_series(df: pd.DataFrame, test_size: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets
    
    Args:
        df: DataFrame (must be sorted by time)
        test_size: Number of periods for test set
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate common forecasting metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }


def print_metrics(metrics: dict, title: str = "Model Metrics"):
    """
    Pretty print model metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the printout
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:10.2f}")
        else:
            print(f"{key:25s}: {value}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test helper functions
    print("\n=== Testing Helper Functions ===\n")
    
    # Test inflation calculation
    inflation = calculate_inflation_rate(50, 45)
    print(f"Inflation (45→50): {format_percentage(inflation)}")
    
    # Test CAGR
    cagr = calculate_cagr(35, 50, 15)
    print(f"CAGR (35→50 over 15 years): {format_percentage(cagr)}")
    
    # Test currency formatting
    print(f"Formatted price: {format_currency(1234.567)}")
    
    # Test cyclical encoding
    months = pd.Series([1, 2, 3, 4, 5, 6])
    sin_months, cos_months = encode_cyclical_feature(months, 12)
    print(f"Cyclical encoding test: sin[1]={sin_months.iloc[0]:.3f}, cos[1]={cos_months.iloc[0]:.3f}")
    
    # Test metrics
    y_true = np.array([45, 47, 48, 50, 52])
    y_pred = np.array([46, 46, 49, 51, 51])
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Forecast Metrics")
    
    print("✓ Helper function tests completed successfully")
