"""
Kenya Food Price Inflation Tracker - Streamlit Dashboard
Modernized version with modular architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_manager import load_all_data, get_commodities, get_regions, filter_data, get_data_quality_score

st.set_page_config(
    page_title="Kenya Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
            border-radius: 8px;
            transition: all 0.3s ease;
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
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background: var(--light); border-radius: 8px; padding: 8px 16px; font-weight: 500; }
        .stTabs [aria-selected="true"] { background: var(--primary); color: white; }
    </style>
    """, unsafe_allow_html=True)


def render_header(page_title: str, emoji: str = "🌾"):
    """Render page header with greeting"""
    nairobi_tz = __import__('pytz').timezone('Africa/Nairobi')
    nairobi_time = datetime.now(nairobi_tz)
    hour = nairobi_time.hour
    
    if hour < 12:
        greeting = "Good Morning"
    elif hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    
    st.markdown(f"""
    <h1 class="page-header">{greeting} {emoji}</h1>
    <p style="color: {THEME['gray']}; font-size: 1.1rem;">{page_title}</p>
    """, unsafe_allow_html=True)


def render_sidebar(data):
    """Render sidebar navigation and stats"""
    with st.sidebar:
        st.markdown("<h2 style='color: var(--primary);'>🌾 Navigation</h2>", unsafe_allow_html=True)
        
        page = st.radio(
            "",
            ["🏠 Home", "📊 Price Explorer", "💰 Inflation Calculator", 
             "🔮 Price Forecasts", "📈 Market Intelligence", "ℹ️ About"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if data.staples is not None:
            st.markdown("### 📊 Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{len(data.staples):,}")
            with col2:
                st.metric("Commodities", data.staples['cm_name'].nunique())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Markets", data.staples['mkt_name'].nunique())
            with col2:
                years = data.staples['date'].dt.year.nunique()
                st.metric("Years", f"{data.staples['date'].min().year}-{data.staples['date'].max().year}")
            
            completeness = 1 - (data.staples['mp_price'].isna().sum() / len(data.staples))
            st.markdown("### 📈 Data Quality")
            st.progress(completeness)
            st.caption(f"Completeness: {completeness*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown("""
        <a href="https://github.com/Kaks753/food-inflation" target="_blank">
            <span style="display: block; padding: 8px 16px; background: var(--primary); color: white; 
            border-radius: 8px; text-align: center; margin: 4px 0; text-decoration: none;">GitHub</span>
        </a>
        """, unsafe_allow_html=True)
        
        return page


@st.cache_data
def load_data_cached():
    """Cache data loading"""
    return load_all_data()


def render_home_page(data):
    """Render Home Dashboard page"""
    render_header("Market Pulse Dashboard", "🌾")
    
    if data.staples is None:
        st.error(data.error or "Failed to load data")
        return
    
    commodities = get_commodities(data.staples)
    maize_data = data.staples[data.staples['cm_name'].str.contains('maize', case=False, na=False, regex=False)]
    
    if len(maize_data) == 0:
        maize_data = data.staples.iloc[:500] if len(data.staples) > 0 else pd.DataFrame()
    
    if len(maize_data) > 0:
        avg_price = maize_data['mp_price'].mean()
        latest_price = maize_data.sort_values('date').iloc[-1]['mp_price']
        volatility = (maize_data['mp_price'].std() / avg_price * 100) if avg_price > 0 else 0
        
        year_2006 = data.staples[data.staples['date'].dt.year == 2006]['mp_price'].mean()
        year_2021 = data.staples[data.staples['date'].dt.year == 2021]['mp_price'].mean()
        growth = ((year_2021 - year_2006) / year_2006 * 100) if year_2006 > 0 and year_2021 > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"{latest_price:.2f} KES/kg", help="Latest maize price")
        with col2:
            st.metric("15-Year Growth", f"+{growth:.1f}%", help="Total price increase since 2006")
        with col3:
            st.metric("Volatility", f"{volatility:.1f}%", help="Price fluctuation level")
        with col4:
            st.metric("Data Records", f"{len(data.staples):,}")
        
        if volatility > 30:
            st.error(f"⚠️ **HIGH VOLATILITY**: Prices fluctuated {volatility:.1f}% - consider hedging")
        elif volatility > 15:
            st.warning(f"📊 **MODERATE VOLATILITY**: Normal fluctuations of {volatility:.1f}%")
        else:
            st.success(f"✅ **LOW VOLATILITY**: Stable market ({volatility:.1f}% variation)")
    
    st.markdown("---")
    st.markdown("<h3 class='section-header'>📊 Market Coverage</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_markets = data.staples.groupby('mkt_name').size().sort_values(ascending=False).head(8)
        if len(top_markets) > 0:
            fig = px.bar(
                x=top_markets.values,
                y=top_markets.index,
                orientation='h',
                title='Top Markets by Data Coverage',
                labels={'x': 'Price Records', 'y': 'Market'},
                color=top_markets.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        commodity_counts = data.staples['cm_name'].value_counts().head(8)
        if len(commodity_counts) > 0:
            fig = px.pie(
                values=commodity_counts.values,
                names=commodity_counts.index,
                title='Top 8 Tracked Commodities',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_price_explorer(data):
    """Render Price Explorer page"""
    render_header("🔍 Price Explorer", "📊")
    
    if data.staples is None:
        st.error(data.error or "Failed to load data")
        return
    
    st.markdown("""
    <div class="info-box">
        <p><strong>Explore historical price trends</strong> for different commodities across regions and time periods.</p>
    </div>
    """, unsafe_allow_html=True)
    
    commodities = get_commodities(data.staples)
    regions = get_regions(data.staples)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_commodity = st.selectbox("Select Commodity", commodities, key="explorer_comm")
    with col2:
        selected_region = st.selectbox("Select Region", regions, key="explorer_region")
    with col3:
        year_range = st.slider("Year Range", 2006, 2021, (2006, 2021), key="explorer_years")
    
    filtered = filter_data(data.staples, selected_commodity, selected_region, year_range)
    
    if len(filtered) == 0:
        st.warning("No data available for selected filters")
        return
    
    avg_price = filtered['mp_price'].mean()
    max_price = filtered['mp_price'].max()
    min_price = filtered['mp_price'].min()
    cv = (filtered['mp_price'].std() / avg_price * 100) if avg_price > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Price", f"{avg_price:.2f} KES/kg")
    with col2:
        st.metric("Max Price", f"{max_price:.2f} KES/kg")
    with col3:
        st.metric("Min Price", f"{min_price:.2f} KES/kg")
    with col4:
        st.metric("Volatility", f"{cv:.1f}%")
    
    st.markdown("---")
    
    price_trend = filtered.groupby('date')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
    price_trend.columns = ['date', 'price', 'std', 'count']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_trend['date'],
        y=price_trend['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color=THEME['primary'], width=2.5)
    ))
    
    if 'std' in price_trend.columns and price_trend['std'].notna().any():
        fig.add_trace(go.Scatter(
            x=price_trend['date'],
            y=price_trend['price'] + price_trend['std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=price_trend['date'],
            y=price_trend['price'] - price_trend['std'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(width=0),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{selected_commodity} - Historical Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price (KES/kg)",
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_inflation_calculator(data):
    """Render Inflation Calculator page"""
    render_header("💰 Inflation Calculator", "💰")
    
    if data.staples is None:
        st.error(data.error or "Failed to load data")
        return
    
    st.markdown("""
    <div class="info-box">
        <p><strong>Calculate the real cost impact</strong> of food price inflation on your household budget.</p>
    </div>
    """, unsafe_allow_html=True)
    
    commodities = get_commodities(data.staples)
    
    col1, col2 = st.columns(2)
    with col1:
        calc_commodity = st.selectbox("Select Food Item", commodities, key="calc_comm")
        start_year = st.selectbox("Start Year", range(2006, 2021), index=0, key="calc_start")
    with col2:
        quantity = st.number_input("Monthly Quantity (kg)", 1.0, 100.0, 10.0, step=1.0, key="calc_qty")
        end_year = st.selectbox("End Year", range(2007, 2022), index=len(range(2007, 2022))-1, key="calc_end")
    
    if st.button("Calculate Impact", type="primary", use_container_width=True, key="calc_btn"):
        commodity_data = data.staples[data.staples['cm_name'] == calc_commodity]
        
        start_vals = commodity_data[commodity_data['date'].dt.year == start_year]['mp_price']
        end_vals = commodity_data[commodity_data['date'].dt.year == end_year]['mp_price']
        
        if len(start_vals) == 0 or len(end_vals) == 0:
            st.warning(f"No data for {calc_commodity} in selected years")
            return
        
        start_avg = start_vals.mean()
        end_avg = end_vals.mean()
        price_change = end_avg - start_avg
        percent_change = (price_change / start_avg) * 100 if start_avg > 0 else 0
        annual_rate = percent_change / (end_year - start_year) if end_year > start_year else 0
        
        st.markdown("---")
        st.markdown("## 📊 Impact Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Price in {start_year}", f"{start_avg:.2f} KES/kg")
        with col2:
            delta = f"+{percent_change:.1f}%" if percent_change > 0 else f"{percent_change:.1f}%"
            st.metric(f"Price in {end_year}", f"{end_avg:.2f} KES/kg", delta=delta)
        with col3:
            st.metric("Monthly Impact", f"{price_change * quantity:+.2f} KES")
        with col4:
            st.metric("Annual Impact", f"{annual_rate:+.1f}%/yr")


def render_price_forecasts(data):
    """Render Price Forecasts page"""
    render_header("🔮 Price Forecasts", "🔮")
    
    if data.staples is None:
        st.error(data.error or "Failed to load data")
        return
    
    commodities = get_commodities(data.staples)
    
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        forecast_commodity = st.selectbox("Select Commodity", commodities, key="forecast_comm")
        periods = st.slider("Forecast Horizon (months)", 3, 24, 12, key="forecast_periods")
    with col2:
        comm_data = data.staples[data.staples['cm_name'] == forecast_commodity]
        if len(comm_data) > 0:
            st.success(f"✅ {len(comm_data)} records available")
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True, key="gen_forecast"):
        if len(comm_data) < 12:
            st.error("Need at least 12 months of data for forecasting")
            return
        
        with st.spinner("Generating forecast..."):
            commodity_data = data.staples[data.staples['cm_name'] == forecast_commodity].copy()
            commodity_data['month'] = commodity_data['date'].dt.month
            
            historical = commodity_data.groupby('date')['mp_price'].mean().reset_index()
            historical = historical.sort_values('date')
            
            if len(historical) < 12:
                st.error("Insufficient data for forecasting")
                return
            
            prices = historical['mp_price'].values
            alpha = 0.3
            exp_forecast, linear_forecast, seasonal_forecast = [], [], []
            
            last_val = prices[-1]
            for i in range(periods):
                prev_mean = np.mean(prices[-12:]) if len(prices) >= 12 else np.mean(prices)
                next_val = alpha * last_val + (1 - alpha) * prev_mean
                exp_forecast.append(next_val)
                last_val = next_val
            
            x = np.arange(len(prices))
            z = np.polyfit(x, prices, 1)
            trend = np.poly1d(z)
            linear_forecast = [trend(len(prices) + i) for i in range(periods)]
            
            if len(prices) >= 12:
                seasonal_pattern = prices[-12:]
                seasonal_forecast = [seasonal_pattern[i % 12] for i in range(periods)]
            else:
                seasonal_forecast = [prices[-1]] * periods
            
            ensemble = (0.4 * np.array(exp_forecast) + 
                       0.3 * np.array(linear_forecast) + 
                       0.3 * np.array(seasonal_forecast))
            
            last_date = historical['date'].max()
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='MS'
            )
            
            std_dev = np.std(prices)
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'price': ensemble,
                'lower': np.maximum(ensemble - std_dev * 1.645, 0),
                'upper': ensemble + std_dev * 1.645
            })
            
            st.markdown("## 📈 Price Forecast")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical['date'], y=historical['mp_price'],
                mode='lines', name='Historical',
                line=dict(color=THEME['primary'], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['price'],
                mode='lines', name='Forecast',
                line=dict(color=THEME['danger'], width=3, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
                fill='toself', name='Confidence Interval',
                fillcolor='rgba(231, 76, 60, 0.15)', line=dict(color='rgba(0,0,0,0)')
            ))
            
            fig.update_layout(
                title=f"{forecast_commodity} - {periods} Month Forecast",
                xaxis_title="Date", yaxis_title="Price (KES/kg)",
                height=500, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


def render_market_intelligence(data):
    """Render Market Intelligence page"""
    render_header("📊 Market Intelligence", "📈")
    
    if data.staples is None:
        st.error(data.error or "Failed to load data")
        return
    
    tab1, tab2, tab3 = st.tabs(["Regional Comparison", "Seasonal Patterns", "Correlations"])
    
    with tab1:
        st.markdown("### Regional Price Dynamics")
        commodity = st.selectbox("Select Commodity", get_commodities(data.staples), key="reg_tab_comm")
        regional = data.staples[data.staples['cm_name'] == commodity]
        
        if len(regional) > 0:
            region_avg = regional.groupby('adm1_name')['mp_price'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=region_avg.values, y=region_avg.index,
                orientation='h', title=f"{commodity} - Regional Averages",
                color=region_avg.values, color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Seasonal Price Patterns")
        seasonal_comm = st.selectbox("Select Commodity", get_commodities(data.staples), key="season_tab_comm")
        seasonal_data = data.staples[data.staples['cm_name'] == seasonal_comm].copy()
        
        if len(seasonal_data) > 0:
            seasonal_data['month'] = seasonal_data['date'].dt.month
            monthly = seasonal_data.groupby('month')['mp_price'].agg(['mean', 'std']).reset_index()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['month_name'], y=monthly['mean'],
                mode='lines+markers', name='Avg Price',
                line=dict(color=THEME['secondary'], width=3),
                error_y=dict(type='data', array=monthly['std'], visible=True)
            ))
            fig.update_layout(title=f"Seasonal Pattern - {seasonal_comm}", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Commodity Correlations")
        top_comms = data.staples['cm_name'].value_counts().head(6).index.tolist()
        selected = st.multiselect(
            "Select commodities (2-8)", get_commodities(data.staples),
            default=top_comms, max_selections=8, key="corr_sel"
        )
        
        if len(selected) >= 2:
            corr_data = []
            valid_comms = []
            for comm in selected:
                m = data.staples[data.staples['cm_name'] == comm].groupby('date')['mp_price'].mean()
                if len(m) > 0:
                    corr_data.append(m)
                    valid_comms.append(comm)
            
            if len(valid_comms) >= 2:
                cdf = pd.DataFrame(corr_data, index=valid_comms).T
                cm = cdf.corr()
                
                fig = px.imshow(cm, labels=dict(color='Correlation'), 
                               color_continuous_scale='RdBu', aspect='auto', zmin=-1, zmax=1,
                               text_auto='.2f')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)


def render_about_page(data):
    """Render About page"""
    render_header("ℹ️ About This Platform", "🌾")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        Provide data-driven insights for better food price decisions.
        
        ### 🛠️ Features
        - **Price Explorer**: Track prices across markets
        - **Inflation Calculator**: Understand personal impact
        - **Price Forecasts**: Plan future purchases
        - **Market Intelligence**: Deep market analysis
        
        ### 📊 Data Source
        World Food Programme (WFP) - 2006 to 2021
        """)
    
    with col2:
        st.markdown("### 📈 Platform Stats")
        if data.staples is not None:
            st.metric("Total Records", f"{len(data.staples):,}")
            st.metric("Commodities", data.staples['cm_name'].nunique())
            st.metric("Markets", data.staples['mkt_name'].nunique())
            st.metric("Years", f"{data.staples['date'].min().year}-{data.staples['date'].max().year}")


def main():
    inject_global_css()
    
    data = load_data_cached()
    page = render_sidebar(data)
    
    if page == "🏠 Home":
        render_home_page(data)
    elif page == "📊 Price Explorer":
        render_price_explorer(data)
    elif page == "💰 Inflation Calculator":
        render_inflation_calculator(data)
    elif page == "🔮 Price Forecasts":
        render_price_forecasts(data)
    elif page == "📈 Market Intelligence":
        render_market_intelligence(data)
    elif page == "ℹ️ About":
        render_about_page(data)


if __name__ == "__main__":
    main()