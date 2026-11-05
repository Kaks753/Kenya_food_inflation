"""
Kenya Food Price Inflation Tracker - UI Components Library
Reusable UI components with consistent styling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Callable
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

THEME = {
    'primary': '#1e3c72',
    'secondary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#16a085',
    'light': '#f8f9fa',
    'dark': '#2c3e50',
    'gray': '#6c757d'
}


@st.cache_resource
def get_theme() -> dict:
    return THEME


def inject_global_css():
    """Inject global CSS styles"""
    st.markdown("""
    <style>
        :root {
            --primary: #1e3c72;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #f8f9fa;
            --dark: #2c3e50;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            border-radius: 8px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }
        .metric-label { font-size: 0.85rem; color: var(--dark); font-weight: 500; margin-bottom: 0.25rem; }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: var(--primary); }
        .metric-unit { font-size: 0.8rem; color: var(--gray); }
        .page-header { font-size: 2rem; font-weight: 700; color: var(--primary); margin-bottom: 1.5rem; }
        .section-header { font-size: 1.4rem; font-weight: 600; color: var(--dark); margin: 1.5rem 0 1rem 0; }
        .info-box {
            background: var(--light);
            padding: 1.25rem;
            border-radius: 10px;
            border-left: 4px solid var(--primary);
            margin: 1rem 0;
        }
        .success-box { background: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid var(--success); }
        .warning-box { background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid var(--warning); }
        .danger-box { background: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid var(--danger); }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background: var(--light); border-radius: 8px; padding: 8px 16px; font-weight: 500; }
        .stTabs [aria-selected="true"] { background: var(--primary); color: white; }
    </style>
    """, unsafe_allow_html=True)


def create_metric_card(label: str, value: str, delta: Optional[str] = None, help_text: str = ""):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{help_text}</div>
        {f'<div style="color: {THEME["success"]}; font-size: 0.9rem; margin-top: 0.25rem;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def create_price_trend_chart(
    df: pd.DataFrame,
    commodity: str,
    title: str = None
) -> go.Figure:
    """Create a price trend chart with consistent styling"""
    if title is None:
        title = f"{commodity} Price Trends"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['mp_price'],
        mode='lines',
        name='Price',
        line=dict(color=THEME['primary'], width=2.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (KES/kg)",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    return fig


def create_regional_heatmap(
    df: pd.DataFrame,
    commodity: str
) -> go.Figure:
    """Create regional price comparison heatmap"""
    regional_data = df[df['cm_name'] == commodity].groupby('adm1_name')['mp_price'].mean()
    
    fig = px.bar(
        x=regional_data.values,
        y=regional_data.index,
        orientation='h',
        title=f"{commodity} - Regional Price Comparison",
        labels={'x': 'Price (KES/kg)', 'y': 'Region'},
        color=regional_data.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig


def create_monthly_heatmap(
    df: pd.DataFrame,
    commodity: str
) -> go.Figure:
    """Create seasonal pattern heatmap"""
    commodity_df = df[df['cm_name'] == commodity].copy()
    commodity_df['month'] = commodity_df['date'].dt.month
    commodity_df['year'] = commodity_df['date'].dt.year
    
    pivot = commodity_df.pivot_table(
        values='mp_price',
        index='year',
        columns='month',
        aggfunc='mean'
    )
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = month_names
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Price (KES)"),
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    return fig


def calculate_derived_metrics(
    df: pd.DataFrame,
    commodity: str,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate derived metrics for a commodity"""
    filtered = df[df['cm_name'] == commodity].copy()
    
    if region and region != 'All Regions':
        filtered = filtered[filtered['adm1_name'] == region]
    
    if len(filtered) == 0:
        return {}
    
    avg_price = filtered['mp_price'].mean()
    max_price = filtered['mp_price'].max()
    min_price = filtered['mp_price'].min()
    latest_price = filtered.sort_values('date').iloc[-1]['mp_price'] if len(filtered) > 0 else 0
    volatility = (filtered['mp_price'].std() / avg_price * 100) if avg_price > 0 else 0
    
    year_2006 = filtered[filtered['date'].dt.year == 2006]['mp_price'].mean()
    year_2021 = filtered[filtered['date'].dt.year == 2021]['mp_price'].mean()
    growth = ((year_2021 - year_2006) / year_2006 * 100) if year_2006 > 0 and year_2021 > 0 else 0
    
    return {
        'current_price': latest_price,
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'volatility': volatility,
        'growth_rate': growth,
        'price_range': max_price - min_price,
        'records': len(filtered)
    }


def create_alert_message(metric_type: str, value: float, threshold: float) -> Optional[str]:
    """Create alert message based on metric thresholds"""
    if metric_type == 'volatility':
        if value > 30:
            return st.error(f"⚠️ **HIGH VOLATILITY**: Prices fluctuated {value:.1f}% - consider hedging strategies")
        elif value > 15:
            return st.warning(f"📊 **MODERATE VOLATILITY**: Normal fluctuations of {value:.1f}%")
        else:
            return st.success(f"✅ **LOW VOLATILITY**: Stable market ({value:.1f}% variation)")
    elif metric_type == 'growth':
        if value > 50:
            return st.warning(f"📈 **STRONG GROWTH**: {value:.1f}% increase over 15 years")
        else:
            return st.info(f"📊 Growth rate: {value:.1f}%")
    return None


def generate_seasonal_forecast(
    df: pd.DataFrame,
    commodity: str,
    periods: int = 12
) -> tuple:
    """Generate seasonal forecast using ensemble method"""
    commodity_data = df[df['cm_name'] == commodity].copy()
    if len(commodity_data) == 0:
        return None, None
    
    monthly_avg = commodity_data.groupby('date')['mp_price'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('date')
    
    if len(monthly_avg) < 6:
        return None, None
    
    prices = monthly_avg['mp_price'].values
    
    alpha = 0.3
    exp_forecast = []
    last_value = prices[-1]
    for i in range(periods):
        prev_mean = np.mean(prices[-12:]) if len(prices) >= 12 else np.mean(prices)
        next_val = alpha * last_value + (1 - alpha) * prev_mean
        exp_forecast.append(next_val)
        last_value = next_val
    
    if len(prices) > 1:
        x = np.arange(len(prices))
        z = np.polyfit(x, prices, 1)
        trend = np.poly1d(z)
        linear_forecast = [trend(len(prices) + i) for i in range(periods)]
    else:
        linear_forecast = [prices[-1]] * periods
    
    if len(prices) >= 12:
        seasonal_pattern = prices[-12:]
        seasonal_forecast = [seasonal_pattern[i % 12] for i in range(periods)]
    else:
        seasonal_forecast = [prices[-1]] * periods
    
    ensemble = (0.4 * np.array(exp_forecast) + 
                0.3 * np.array(linear_forecast) + 
                0.3 * np.array(seasonal_forecast))
    
    last_date = monthly_avg['date'].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )
    
    std_dev = np.std(prices) if len(prices) > 1 else ensemble.mean() * 0.1
    confidence_width = std_dev * 1.645
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'price': ensemble,
        'lower': np.maximum(ensemble - confidence_width, 0),
        'upper': ensemble + confidence_width,
        'exp_smoothing': exp_forecast,
        'linear_trend': linear_forecast,
        'seasonal': seasonal_forecast
    })
    
    return monthly_avg, forecast_df


if __name__ == "__main__":
    print("Components library loaded successfully")