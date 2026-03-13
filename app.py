"""
Kenya Food Price Inflation Tracker - Advanced Interactive Dashboard
=====================================================================

Author: Stephen Muema
Portfolio: https://muemastephenportfolio.netlify.app/
Repository: https://github.com/Kaks753/food-inflation
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

# Page configuration
st.set_page_config(
    page_title="Kenya Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #e74c3c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #e74c3c;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_data():
    """Load all datasets with intelligent error handling"""
    try:
        # Core datasets
        staples = pd.read_csv('data/clean/wfp_core_staples.csv', parse_dates=['date'])
        monthly = pd.read_csv('data/clean/wfp_monthly_avg.csv', parse_dates=['date'])
        
        # Clean commodity names for consistency
        staples['cm_name'] = staples['cm_name'].str.strip()
        
        # Try to load features
        try:
            maize_features = pd.read_csv('data/clean/maize_features.csv', parse_dates=['date'])
        except:
            maize_features = None
        
        # Model outputs
        try:
            prophet_fc = pd.read_csv('models/trained/prophet_forecast.csv', parse_dates=['ds'])
        except:
            prophet_fc = None
        
        try:
            model_comparison = pd.read_csv('models/trained/model_comparison.csv', index_col=0)
        except:
            model_comparison = None
        
        return staples, monthly, maize_features, prophet_fc, model_comparison
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None, None, None, None, None

# Enhanced forecasting function
@st.cache_data
def generate_enhanced_forecast(df, commodity, periods=12):
    """Generate intelligent forecasts using ensemble of methods"""
    try:
        commodity_data = df[df['cm_name'] == commodity].copy()
        if len(commodity_data) == 0:
            return None, None
        
        # Calculate monthly average
        monthly_avg = commodity_data.groupby('date')['mp_price'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('date')
        
        if len(monthly_avg) < 6:
            return None, None
        
        prices = monthly_avg['mp_price'].values
        
        # Method 1: Exponential Smoothing
        alpha = 0.3
        exp_forecast = []
        last_value = prices[-1]
        for i in range(periods):
            next_val = alpha * last_value + (1 - alpha) * np.mean(prices[-12:]) if len(prices) >= 12 else last_value
            exp_forecast.append(next_val)
            last_value = next_val
        
        # Method 2: Linear Trend
        if len(prices) > 1:
            x = np.arange(len(prices))
            z = np.polyfit(x, prices, 1)
            trend = np.poly1d(z)
            linear_forecast = [trend(len(prices) + i) for i in range(periods)]
        else:
            linear_forecast = [prices[-1]] * periods
        
        # Method 3: Seasonal Naive
        if len(prices) >= 12:
            seasonal_pattern = prices[-12:]
            seasonal_forecast = []
            for i in range(periods):
                seasonal_forecast.append(seasonal_pattern[i % 12])
        else:
            seasonal_forecast = [prices[-1]] * periods
        
        # Ensemble forecast (weighted average)
        ensemble = (0.4 * np.array(exp_forecast) + 
                    0.3 * np.array(linear_forecast) + 
                    0.3 * np.array(seasonal_forecast))
        
        last_date = monthly_avg['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=30),
            periods=periods,
            freq='MS'
        )
        
        # Calculate confidence intervals
        std_dev = np.std(prices) if len(prices) > 1 else ensemble.mean() * 0.1
        confidence_width = std_dev * 1.645  # 90% confidence default
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'price': ensemble,
            'lower': ensemble - confidence_width,
            'upper': ensemble + confidence_width,
            'exp_smoothing': exp_forecast,
            'linear_trend': linear_forecast,
            'seasonal': seasonal_forecast
        })
        
        # Ensure no negative prices
        forecast_df['lower'] = forecast_df['lower'].clip(lower=0)
        forecast_df['price'] = forecast_df['price'].clip(lower=0)
        
        return monthly_avg, forecast_df
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None, None

# Load data
staples, monthly, maize_features, prophet_fc, model_comparison = load_data()

# Check if data loaded successfully
if staples is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3892/3892646.png", width=100)
    st.title("🌾 Navigation")
    
    page = st.radio(
        "Select Page",
        [
            "🏠 Home Dashboard",
            "📊 Price Explorer", 
            "💰 Inflation Calculator",
            "🔮 Price Forecasts",
            "📈 Market Intelligence",
            "ℹ️ About",
            "👨‍💻 Developer"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Data Overview")
    
    total_records = len(staples)
    commodities = staples['cm_name'].nunique()
    markets = staples['mkt_name'].nunique()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    with col2:
        st.metric("Commodities", commodities)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Markets", markets)
    with col2:
        st.metric("Data Period", "2006-2021")
    
    # Data quality indicator
    completeness = (1 - staples['mp_price'].isna().sum() / len(staples)) * 100
    st.markdown("### 📈 Data Quality")
    st.progress(completeness / 100)
    st.caption(f"Completeness: {completeness:.1f}%")
    
    st.markdown("---")
    
    # Links
    st.markdown("### 🔗 Connect")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Kaks753/food-inflation)
    [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue)](https://muemastephenportfolio.netlify.app/)
    """)

# ============================================================================
# PAGE: HOME DASHBOARD
# ============================================================================
if page == "🏠 Home Dashboard":
    
    # Personalized Welcome
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good Morning"
    elif current_hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "Visitor"
    
    st.markdown(f"""
    <h1 class='main-header'>
        {greeting}, {st.session_state.user_name}! 🌾
    </h1>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class='info-box'>
    <h4>🌾 Welcome to Your Complete Food Price Intelligence System</h4>
    <p>Your one-stop platform for tracking, analyzing, and forecasting food prices across Kenya. 
    Everything you need is just one click away!</p>
    </div>
    """, unsafe_allow_html=True)
    
 # Smart Key Metrics
    st.markdown("## Market Pulse")
    
    # Find maize data safely
    maize_keywords = ['Maize', 'maize']
    maize_data = None
    
    for keyword in maize_keywords:
        mask = staples['cm_name'].str.contains(keyword, case=False, na=False)
        if mask.any():
            maize_data = staples[mask].copy()
            break
    
    if maize_data is None or len(maize_data) == 0:
        maize_data = staples.iloc[:100]  # fallback to first 100 rows
    
    # Calculate metrics with safety checks
    maize_avg = maize_data['mp_price'].mean() if len(maize_data) > 0 else 0
    maize_max = maize_data['mp_price'].max() if len(maize_data) > 0 else 0
    maize_min = maize_data['mp_price'].min() if len(maize_data) > 0 else 0
    
    # Get latest price
    latest_maize = maize_data.sort_values('date').iloc[-1]['mp_price'] if len(maize_data) > 0 else 0
    price_range = maize_max - maize_min
    
    # Calculate volatility safely
    volatility = (maize_data['mp_price'].std() / maize_avg * 100) if maize_avg > 0 else 0
    
    # Historical growth
    first_year_avg = maize_data[maize_data['date'].dt.year == 2006]['mp_price'].mean() if 2006 in maize_data['date'].dt.year.values else 0
    last_year_avg = maize_data[maize_data['date'].dt.year == 2021]['mp_price'].mean() if 2021 in maize_data['date'].dt.year.values else 0
    
    if first_year_avg > 0 and last_year_avg > 0:
        growth_rate = ((last_year_avg - first_year_avg) / first_year_avg) * 100
    else:
        growth_rate = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Maize Price (2021)",
            value=f"{latest_maize:.2f} KES/kg",
            delta=f"{((latest_maize - maize_avg) / maize_avg * 100):.1f}% vs avg" if maize_avg > 0 else None,
            help="Latest recorded price compared to historical average"
        )
    
    with col2:
        st.metric(
            label="15-Year Growth",
            value=f"+{growth_rate:.1f}%" if growth_rate > 0 else f"{growth_rate:.1f}%",
            delta=f"CAGR: {(growth_rate/15):.1f}%/year" if growth_rate != 0 else None,
            help="Total price increase from 2006 to 2021"
        )
    
    with col3:
        st.metric(
            label="Price Volatility",
            value=f"{volatility:.1f}%",
            delta=f"Range: {price_range:.0f} KES",
            help="Coefficient of variation showing price stability"
        )
    
    with col4:
        st.metric(
            label="Best Buying Season",
            value="Oct-Dec",
            delta="↓ 20% cheaper",
            help="Harvest season typically offers 15-20% lower prices"
        )
    
    # Data coverage visualization
    st.markdown("## Market Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top markets
        top_markets = staples.groupby('mkt_name').size().sort_values(ascending=False).head(10)
        if len(top_markets) > 0:
            fig = px.bar(
                x=top_markets.values,
                y=top_markets.index,
                orientation='h',
                title='Top 10 Markets by Data Coverage',
                labels={'x': 'Number of Price Records', 'y': 'Market'},
                color=top_markets.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Commodities by category
        commodity_counts = staples['cm_name'].value_counts().head(8)
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
    
    # Regional insights
    if 'adm1_name' in staples.columns:
        st.markdown("## Regional Market Dynamics")
        
        region_avg = staples.groupby('adm1_name')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
        region_avg.columns = ['Region', 'Avg Price (KES)', 'Std Dev', 'Records']
        region_avg = region_avg.sort_values('Avg Price (KES)', ascending=False)
        
        if len(region_avg) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    region_avg,
                    x='Region',
                    y='Avg Price (KES)',
                    error_y='Std Dev',
                    title='Average Food Prices by Region',
                    labels={'Avg Price (KES)': 'Average Price (KES/kg)'},
                    color='Avg Price (KES)',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Key Insights")
                st.markdown(f"""
                - **Highest prices**: {region_avg.iloc[0]['Region']} ({region_avg.iloc[0]['Avg Price (KES)']:.1f} KES)
                - **Lowest prices**: {region_avg.iloc[-1]['Region']} ({region_avg.iloc[-1]['Avg Price (KES)']:.1f} KES)
                - **Price gap**: {(region_avg.iloc[0]['Avg Price (KES)'] - region_avg.iloc[-1]['Avg Price (KES)']):.1f} KES
                - **Most data**: {region_avg.loc[region_avg['Records'].idxmax(), 'Region']}
                """)


# ============================================================================
# PAGE: PRICE EXPLORER
# ============================================================================
elif page == "📊 Price Explorer":
    st.markdown("<h1 class='main-header'>📊 Price Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <p><strong>Explore historical price trends</strong> for different commodities across regions and time periods.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    commodity_options = sorted(staples['cm_name'].unique())
    region_options = ['All Regions'] + sorted(staples['adm1_name'].unique().tolist())
    
    with col1:
        selected_commodity = st.selectbox(
            "Select Commodity",
            commodity_options
        )
    
    with col2:
        selected_region = st.selectbox(
            "Select Region",
            region_options
        )
    
    with col3:
        year_range = st.slider(
            "Year Range",
            min_value=2006,
            max_value=2021,
            value=(2006, 2021)
        )
    
    # Filter data
    filtered_data = staples[staples['cm_name'] == selected_commodity].copy()
    filtered_data = filtered_data[
        (filtered_data['date'].dt.year >= year_range[0]) &
        (filtered_data['date'].dt.year <= year_range[1])
    ]
    
    if selected_region != 'All Regions':
        filtered_data = filtered_data[filtered_data['adm1_name'] == selected_region]
    
    if len(filtered_data) == 0:
        st.warning("No data available for selected filters. Try different options.")
        st.stop()
    
    # Aggregate
    price_trend = filtered_data.groupby('date')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
    price_trend.columns = ['date', 'price', 'std', 'count']
    
    # Main price chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_trend['date'],
        y=price_trend['price'],
        mode='lines',
        name='Price',
        line=dict(color='#3498db', width=3),
        fill='tonexty'
    ))
    
    # Add confidence band
    if 'std' in price_trend.columns:
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
            line=dict(width=0),
            fillcolor='rgba(52, 152, 219, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{selected_commodity} Price Trend",
        xaxis_title='Date',
        yaxis_title='Price (KES/kg)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"{price_trend['price'].mean():.2f} KES")
    with col2:
        st.metric("Max Price", f"{price_trend['price'].max():.2f} KES")
    with col3:
        st.metric("Min Price", f"{price_trend['price'].min():.2f} KES")
    with col4:
        cv = (price_trend['price'].std() / price_trend['price'].mean() * 100) if price_trend['price'].mean() > 0 else 0
        st.metric("Volatility", f"{cv:.1f}%")




# ============================================================================
# PAGE: INFLATION CALCULATOR - COMPLETE POLISHED VERSION
# ============================================================================
elif page == "💰 Inflation Calculator":
    st.markdown("<h1 class='main-header'>💰 Inflation Calculator</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>How This Works</h4>
    <p>Calculate the <strong>real cost impact</strong> of food price inflation on your household budget. 
    This tool uses actual historical data to show how prices have changed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        calc_commodity = st.selectbox(
            "Select Food Item",
            sorted(staples['cm_name'].unique())
        )
        
        start_year = st.selectbox(
            "Start Year",
            range(2006, 2021),
            index=0
        )
    
    with col2:
        quantity = st.number_input(
            "Monthly Quantity (kg)",
            min_value=1.0,
            max_value=1000.0,
            value=10.0,
            step=1.0
        )
        
        end_year = st.selectbox(
            "End Year",
            range(2007, 2022),
            index=len(range(2007, 2022))-1
        )
    
    if st.button("Calculate Impact", type="primary", use_container_width=True):
        
        commodity_data = staples[staples['cm_name'] == calc_commodity].copy()
        
        start_data = commodity_data[commodity_data['date'].dt.year == start_year]['mp_price']
        end_data = commodity_data[commodity_data['date'].dt.year == end_year]['mp_price']
        
        if len(start_data) == 0 or len(end_data) == 0:
            st.warning(f"No data available for {calc_commodity} in selected years.")
            available_years = sorted(commodity_data['date'].dt.year.unique())
            if len(available_years) > 0:
                st.info(f"Available years: {', '.join(map(str, available_years))}")
        else:
            start_avg = start_data.mean()
            end_avg = end_data.mean()
            
            price_change = end_avg - start_avg
            percent_change = (price_change / start_avg) * 100 if start_avg > 0 else 0
            year_diff = end_year - start_year
            annual_rate = percent_change / year_diff if year_diff > 0 else 0
            
            start_cost = start_avg * quantity
            end_cost = end_avg * quantity
            monthly_increase = end_cost - start_cost
            annual_increase = monthly_increase * 12
            
            st.markdown("---")
            st.markdown("## Impact Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Price in {start_year}", f"{start_avg:.2f} KES/kg")
            with col2:
                delta_text = f"+{percent_change:.1f}%" if percent_change > 0 else f"{percent_change:.1f}%"
                st.metric(f"Price in {end_year}", f"{end_avg:.2f} KES/kg", delta=delta_text)
            with col3:
                st.metric("Monthly Impact", f"{monthly_increase:+.2f} KES")
            with col4:
                st.metric("Annual Impact", f"{annual_increase:+.2f} KES", 
                         delta=f"{annual_rate:.1f}% per year" if annual_rate != 0 else None)
            
            # Visualization
            yearly_data = commodity_data.groupby(commodity_data['date'].dt.year)['mp_price'].mean().reset_index()
            yearly_data.columns = ['year', 'price']
            yearly_data = yearly_data[(yearly_data['year'] >= start_year) & (yearly_data['year'] <= end_year)]
            
            if len(yearly_data) > 0:
                yearly_data['monthly_cost'] = yearly_data['price'] * quantity
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Price per Kg', 'Your Monthly Cost'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data['price'],
                        mode='lines+markers',
                        name='Price/kg',
                        line=dict(color='#e74c3c', width=3)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_data['year'],
                        y=yearly_data['monthly_cost'],
                        name='Monthly Cost',
                        marker_color='#3498db'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # ===== SMART INSIGHTS SECTION =====
                st.markdown("## 📊 Impact Analysis & Recommendations")

                # Calculate purchasing power erosion
                purchasing_power_erosion = (1 - (start_avg / end_avg)) * 100 if end_avg > 0 else 0

                # Create three columns for different insights
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    # Impact severity card - NO FIXED HEIGHT
                    if percent_change > 50:
                        impact_color = "#e74c3c"
                        impact_icon = "🔴"
                        impact_level = "SEVERE"
                        recommendation = "Re-evaluate your food budget and consider bulk purchasing during harvest seasons (Oct-Dec). Look for alternative protein sources like beans when maize prices spike."
                    elif percent_change > 30:
                        impact_color = "#f39c12"
                        impact_icon = "🟠"
                        impact_level = "HIGH"
                        recommendation = "Consider adjusting your consumption patterns. Buy in bulk during low-price months and consider joining a community buying group for better rates."
                    elif percent_change > 15:
                        impact_color = "#f1c40f"
                        impact_icon = "🟡"
                        impact_level = "MODERATE"
                        recommendation = "Your food budget is feeling the pressure. Track seasonal patterns and plan purchases around harvest times to minimize impact."
                    elif percent_change > 0:
                        impact_color = "#3498db"
                        impact_icon = "🔵"
                        impact_level = "MILD"
                        recommendation = "Inflation is manageable. Focus on smart shopping habits and keep monitoring price trends to stay ahead."
                    else:
                        impact_color = "#27ae60"
                        impact_icon = "🟢"
                        impact_level = "DECREASE"
                        recommendation = f"Prices have actually decreased! This is a great time to stock up. Consider buying extra to hedge against future increases."

                    st.markdown(f"""
                    <div style='background: {impact_color}20; padding: 1rem; border-radius: 10px; border-left: 5px solid {impact_color}; margin-bottom: 1rem;'>
                        <h4 style='margin:0; color: {impact_color};'>{impact_icon} Impact Level: {impact_level}</h4>
                        <p style='margin-top: 0.5rem; margin-bottom: 0.3rem;'><strong>Price Change:</strong> {percent_change:+.1f}% over {year_diff} years</p>
                        <p style='margin: 0; font-size: 0.95rem;'>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    # Purchasing power card - NO FIXED HEIGHT
                    if percent_change > 0:
                        power_text = f"Your KES now buys {purchasing_power_erosion:.1f}% less {calc_commodity} than in {start_year}"
                        power_detail = f"What cost 100 KES in {start_year} now costs {100 * (1 + percent_change/100):.0f} KES"
                    else:
                        power_text = f"Your KES now buys {abs(purchasing_power_erosion):.1f}% more {calc_commodity} than in {start_year}"
                        power_detail = f"What cost 100 KES in {start_year} now costs {100 * (1 + percent_change/100):.0f} KES"

                    st.markdown(f"""
                    <div style='background: #3498db20; padding: 1rem; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 1rem;'>
                        <h4 style='margin:0; color: #3498db;'>💰 Purchasing Power</h4>
                        <p style='margin-top: 0.5rem; margin-bottom: 0.3rem;'><strong>{power_text}</strong></p>
                        <p style='margin: 0.3rem 0;'>{power_detail}</p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #7f8c8d;'>Annual inflation rate: {annual_rate:+.1f}% per year</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_c:
                    # Budget impact card - NO FIXED HEIGHT
                    monthly_percent = (monthly_increase / start_cost) * 100 if start_cost > 0 else 0
                    yearly_total = end_cost * 12
                    yearly_total_start = start_cost * 12
                    
                    st.markdown(f"""
                    <div style='background: #27ae6020; padding: 1rem; border-radius: 10px; border-left: 5px solid #27ae60; margin-bottom: 1rem;'>
                        <h4 style='margin:0; color: #27ae60;'>📊 Budget Impact</h4>
                        <p style='margin-top: 0.5rem; margin-bottom: 0.3rem;'><strong>Monthly:</strong> {start_cost:,.0f} KES → {end_cost:,.0f} KES</p>
                        <p style='margin: 0.3rem 0;'><strong>Yearly:</strong> {yearly_total_start:,.0f} KES → {yearly_total:,.0f} KES</p>
                        <p style='margin: 0.3rem 0;'><strong>Extra yearly cost:</strong> {annual_increase:+,.0f} KES ({monthly_percent:+.1f}% of original)</p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #7f8c8d;'>That's {annual_increase/12:,.0f} KES more per month</p>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== BUYER & SELLER STRATEGIC RECOMMENDATIONS =====
                
                st.markdown("### 🎯 Strategic Recommendations")

                # Determine commodity category for targeted advice
                commodity_lower = calc_commodity.lower()
                if 'maize' in commodity_lower or 'wheat' in commodity_lower or 'rice' in commodity_lower:
                    category = "Staple Grain"
                    buyer_advice = [
                        "📦 Buy in bulk immediately after harvest (October-December) when prices are lowest",
                        "🤝 Join a cereal bank or cooperative for better collective bargaining",
                        "🏪 Compare wholesale vs retail - wholesale markets offer 15-25% savings",
                        "📊 Track price trends using our Price Explorer to time your purchases"
                    ]
                    seller_advice = [
                        "📈 Best selling period: June-August when prices peak (20-30% premium)",
                        "🏙️ Target urban markets (Nairobi, Mombasa) for 15-25% higher prices",
                        "📦 Invest in storage to sell during lean season for maximum profit",
                        "⭐ Grade your produce - clean, graded grain fetches premium prices"
                    ]
                elif 'bean' in commodity_lower or 'lentil' in commodity_lower or 'pea' in commodity_lower:
                    category = "Legume/Pulse"
                    buyer_advice = [
                        "📅 Buy before March-May when prices peak in lean season",
                        "📦 Stock up during harvest (August-September) - dried beans store 6-12 months",
                        "🔄 Consider substitutions - try cheaper legumes when prices spike",
                        "🌱 Choose local varieties - they're often cheaper than imported"
                    ]
                    seller_advice = [
                        "📈 Peak selling window: March-May when supplies are low",
                        "🏪 Sell in retail packs (1kg/2kg bags) for higher margins",
                        "⭐ Premium varieties can fetch 20-30% more than common types",
                        "🌍 Explore urban markets where prices are 20-40% higher"
                    ]
                elif 'milk' in commodity_lower or 'beef' in commodity_lower or 'meat' in commodity_lower:
                    category = "Animal Product"
                    buyer_advice = [
                        "🤝 Consider fixed-price contracts with suppliers to manage volatility",
                        "🔄 Explore alternatives like eggs or legumes when meat prices spike",
                        "🎉 Buy during festive discounts (Dec-Jan, Apr-May) for better rates",
                        "🚜 Direct-from-farm purchases can save 15-25%"
                    ]
                    seller_advice = [
                        "📈 Premium pricing during festivals: Dec-Jan and Apr-May",
                        "🤝 Secure contracts with processors for stable offtake",
                        "🏭 Add value through simple processing (yogurt, cheese) for retail margins",
                        "⭐ Quality certification can command 15-25% premium"
                    ]
                elif 'vegetable' in commodity_lower or 'tomato' in commodity_lower or 'onion' in commodity_lower:
                    category = "Fresh Produce"
                    buyer_advice = [
                        "📅 Buy in season and preserve/dry excess for off-season",
                        "🏪 Local markets beat supermarkets by 20-40%",
                        "🤝 Build vendor relationships for consistent quality and rates",
                        "📊 Monitor daily prices - fresh produce volatility is high"
                    ]
                    seller_advice = [
                        "📈 Price spikes during off-season - plan production cycles",
                        "🏪 Direct-to-consumer at local markets yields highest margins",
                        "📦 Reduce post-harvest loss through proper handling and storage",
                        "🤝 Supply hotels/restaurants for consistent, premium sales"
                    ]
                else:
                    category = "General Food Item"
                    buyer_advice = [
                        "📊 Track monthly patterns to identify best buying times",
                        "🏪 Compare 3+ markets before major purchases",
                        "📦 Buy in bulk when prices are 10% below average",
                        "🔮 Use Price Forecasts page to plan major purchases"
                    ]
                    seller_advice = [
                        "📊 Identify premium markets using our Price Explorer",
                        "📅 Time sales with seasonal peaks in Seasonal Patterns tab",
                        "🌍 Explore regional arbitrage - transport to high-price areas",
                        "🤝 Build buyer networks to reduce marketing costs"
                    ]

                # Create two columns for buyer and seller advice
                col_buyer, col_seller = st.columns(2)

                with col_buyer:
                    # Format buyer advice as HTML with proper bold using <strong> tags
                    buyer_html = ""
                    for advice in buyer_advice:
                        # Split at the first colon or dash to identify the bold part
                        if ":" in advice:
                            parts = advice.split(":", 1)
                            buyer_html += f"<li><strong>{parts[0]}:</strong>{parts[1]}</li>"
                        elif "-" in advice[:20]:  # Check if dash is within first 20 chars
                            parts = advice.split("-", 1)
                            buyer_html += f"<li><strong>{parts[0]}-</strong>{parts[1]}</li>"
                        else:
                            buyer_html += f"<li>{advice}</li>"
                    
                    st.markdown(f"""
                    <div style='background: #e8f4fd; padding: 1rem; border-radius: 10px; height: 300px; overflow-y: auto;'>
                        <h4 style='margin:0; color: #3498db;'>🛒 For Buyers</h4>
                        <p style='margin-top: 0.5rem; font-size: 0.9rem; font-style: italic;'>How to save money on {calc_commodity}</p>
                        <ul style='margin-top: 0.5rem; padding-left: 1.2rem;'>
                            {buyer_html}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with col_seller:
                    # Format seller advice as HTML with proper bold using <strong> tags
                    seller_html = ""
                    for advice in seller_advice:
                        if ":" in advice:
                            parts = advice.split(":", 1)
                            seller_html += f"<li><strong>{parts[0]}:</strong>{parts[1]}</li>"
                        elif "-" in advice[:20]:
                            parts = advice.split("-", 1)
                            seller_html += f"<li><strong>{parts[0]}-</strong>{parts[1]}</li>"
                        else:
                            seller_html += f"<li>{advice}</li>"
                    
                    st.markdown(f"""
                    <div style='background: #fef5e7; padding: 1rem; border-radius: 10px; height: 300px; overflow-y: auto;'>
                        <h4 style='margin:0; color: #f39c12;'>🌾 For Sellers</h4>
                        <p style='margin-top: 0.5rem; font-size: 0.9rem; font-style: italic;'>How to maximize profits on {calc_commodity}</p>
                        <ul style='margin-top: 0.5rem; padding-left: 1.2rem;'>
                            {seller_html}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== FINANCIAL PROJECTION =====
                st.markdown("### 🔮 Future Cost Projection")

                # Calculate future projection
                future_projection = []
                projection_years = 5
                for i in range(projection_years + 1):
                    future_year = end_year + i
                    if i == 0:
                        future_price = end_avg
                    else:
                        # Project using annual rate
                        future_price = end_avg * (1 + annual_rate/100) ** i
                    future_projection.append({'year': future_year, 'price': future_price})
                
                proj_df = pd.DataFrame(future_projection)
                proj_df['monthly_cost'] = proj_df['price'] * quantity
                proj_df['annual_cost'] = proj_df['monthly_cost'] * 12
                
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Scatter(
                    x=proj_df['year'],
                    y=proj_df['annual_cost'],
                    mode='lines+markers',
                    name='Projected Annual Cost',
                    line=dict(color='#e74c3c', width=3)
                ))
                
                # Add confidence band
                fig_proj.add_trace(go.Scatter(
                    x=proj_df['year'].tolist() + proj_df['year'].tolist()[::-1],
                    y=(proj_df['annual_cost'] * 1.1).tolist() + (proj_df['annual_cost'] * 0.9).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Projection Range (±10%)',
                    hoverinfo='skip'
                ))
                
                fig_proj.update_layout(
                    title=f"What You'll Spend on {calc_commodity} in the Next {projection_years} Years",
                    xaxis_title="Year",
                    yaxis_title="Annual Cost (KES)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_proj, use_container_width=True)

                # ===== REGIONAL PRICE CONTEXT =====
                national_avg_end = staples[staples['date'].dt.year == end_year]['mp_price'].mean()
                if not pd.isna(national_avg_end) and national_avg_end > 0:
                    vs_national = ((end_avg - national_avg_end) / national_avg_end) * 100
                    
                    if abs(vs_national) > 20:
                        comparison_icon = "⚠️"
                        comparison_text = "Significantly different"
                        comparison_color = "#e74c3c"
                    elif abs(vs_national) > 10:
                        comparison_icon = "📊"
                        comparison_text = "Moderately different"
                        comparison_color = "#f39c12"
                    else:
                        comparison_icon = "✅"
                        comparison_text = "Fair price"
                        comparison_color = "#27ae60"

                    st.markdown(f"""
                    <div style='background: {comparison_color}15; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid {comparison_color};'>
                        <h4 style='margin:0; color: {comparison_color};'>{comparison_icon} Regional Price Check</h4>
                        <p style='margin-top: 0.5rem; margin-bottom: 0;'>
                            <strong>You're paying {end_avg:.1f} KES/kg</strong> for {calc_commodity}. 
                            The national average is {national_avg_end:.1f} KES/kg.
                        </p>
                        <p style='margin: 0.3rem 0 0 0;'>
                            That's <strong style='color: {comparison_color};'>{vs_national:+.1f}% {('above' if vs_national > 0 else 'below')} average</strong>.
                        </p>
                        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                            {comparison_icon} <strong>{comparison_text}:</strong> {
                                "You might want to shop around for better deals." if vs_national > 10 else
                                "You're getting a good deal compared to national average!" if vs_national < -10 else
                                "You're paying about the same as most Kenyans."
                            }
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== ACTION TIMELINE =====
                st.markdown("### 📅 What This Means for Your Wallet")

                timeline_col1, timeline_col2, timeline_col3 = st.columns(3)

                with timeline_col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: #e8f4fd; border-radius: 10px; height: 160px;'>
                        <h4 style='margin:0; color: #3498db;'>🔵 Immediate</h4>
                        <p style='font-size: 0.9rem; color: #7f8c8d; margin:0;'>Next 1-3 months</p>
                        <hr style='margin: 0.5rem 0;'>
                        <p style='font-size: 1.2rem; font-weight: bold; margin:0;'>{monthly_increase:+,.0f} KES</p>
                        <p style='font-size: 0.85rem; margin:0;'>Adjust monthly budget</p>
                    </div>
                    """, unsafe_allow_html=True)

                with timeline_col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: #fef5e7; border-radius: 10px; height: 160px;'>
                        <h4 style='margin:0; color: #f39c12;'>🟡 Short-term</h4>
                        <p style='font-size: 0.9rem; color: #7f8c8d; margin:0;'>Next 3-12 months</p>
                        <hr style='margin: 0.5rem 0;'>
                        <p style='font-size: 1.2rem; font-weight: bold; margin:0;'>{annual_increase:+,.0f} KES</p>
                        <p style='font-size: 0.85rem; margin:0;'>Extra yearly cost</p>
                    </div>
                    """, unsafe_allow_html=True)

                with timeline_col3:
                    future_year = proj_df['year'].iloc[-1]
                    future_cost = proj_df['annual_cost'].iloc[-1]
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: #e8f5e8; border-radius: 10px; height: 160px;'>
                        <h4 style='margin:0; color: #27ae60;'>🟢 Long-term</h4>
                        <p style='font-size: 0.9rem; color: #7f8c8d; margin:0;'>By {future_year}</p>
                        <hr style='margin: 0.5rem 0;'>
                        <p style='font-size: 1.2rem; font-weight: bold; margin:0;'>{future_cost:,.0f} KES</p>
                        <p style='font-size: 0.85rem; margin:0;'>Projected yearly cost</p>
                    </div>
                    """, unsafe_allow_html=True)

                # ===== DISCLAIMER =====
                st.markdown("""
                ---
                <div style='font-size: 0.8rem; color: #95a5a6; text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 5px;'>
                    <p style='margin:0;'><i>⚠️ Disclaimer: These projections are based on historical trends and actual inflation rates. 
                    Actual future prices may vary due to market conditions, weather, climate change, economic policies, 
                    and global factors. Use this as a planning tool, not financial advice.</i></p>
                </div>
                """, unsafe_allow_html=True)



# ============================================================================
# PAGE: PRICE FORECASTS - INNOVATIVE SMART VERSION
# ============================================================================
elif page == "🔮 Price Forecasts":
    st.markdown("<h1 class='main-header'>🔮 Smart Price Forecasting</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>🎯 AI-Powered Predictive Intelligence</h4>
    <p>Our advanced forecasting engine analyzes <strong>15+ years of historical data, seasonal patterns, and market trends</strong> 
    to generate intelligent price predictions. We're transparent about uncertainty - longer forecasts show wider confidence intervals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all commodities
    all_commodities = sorted(staples['cm_name'].unique())
    
    # ===== MAIN FORECAST CONTROLS =====
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        forecast_commodity = st.selectbox(
            "🌾 Select Commodity to Forecast",
            all_commodities,
            help="Choose the food item you want to predict",
            key="forecast_commodity_main"
        )
        
        # Visual forecast horizon selector
        st.markdown("#### 📅 Forecast Horizon")
        
        horizon_options = {
            "🔵 Short-term (3-12 months)": {"max": 12, "reliability": "High", "desc": "Most accurate for planning purchases"},
            "🟡 Medium-term (1-3 years)": {"max": 36, "reliability": "Moderate", "desc": "Good for budget planning"},
            "🟠 Long-term (3-5 years)": {"max": 60, "reliability": "Low", "desc": "Shows general trends only"},
            "🎯 Custom Date": {"max": None, "reliability": "Variable", "desc": "Choose any future date"}
        }
        
        horizon_choice = st.radio(
            "Select forecast horizon",
            options=list(horizon_options.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="horizon_radio"
        )
        
        horizon_info = horizon_options[horizon_choice]
        
        # Dynamic input based on selection
        if horizon_choice == "🎯 Custom Date":
            col_a, col_b = st.columns(2)
            with col_a:
                target_year = st.selectbox(
                    "Year",
                    options=range(2024, 2030),
                    index=0,
                    key="target_year"
                )
            with col_b:
                target_month = st.selectbox(
                    "Month",
                    options=range(1, 13),
                    format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
                    index=0,
                    key="target_month"
                )
            
            last_date = staples['date'].max()
            target_date = datetime(target_year, target_month, 1)
            months_diff = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
            
            if months_diff < 1:
                st.error("❌ Target date must be in the future")
                st.stop()
            elif months_diff > 60:
                st.error("❌ Cannot forecast beyond 5 years (statistically unreliable)")
                st.stop()
            
            forecast_periods = months_diff
            target_display = target_date.strftime('%B %Y')
            
            # Set reliability based on months
            if months_diff <= 12:
                reliability = "High"
                reliability_color = "#27ae60"
            elif months_diff <= 36:
                reliability = "Moderate"
                reliability_color = "#f39c12"
            else:
                reliability = "Low"
                reliability_color = "#e74c3c"
                
        else:
            # Extract max months from the option
            max_months = horizon_info["max"]
            reliability = horizon_info["reliability"]
            
            if "Short-term" in horizon_choice:
                default_val = 6
                reliability_color = "#27ae60"
            elif "Medium-term" in horizon_choice:
                default_val = 24
                reliability_color = "#f39c12"
            else:  # Long-term
                default_val = 48
                reliability_color = "#e74c3c"
            
            forecast_periods = st.slider(
                f"Select number of months to forecast",
                min_value=3,
                max_value=max_months,
                value=default_val,
                step=3 if max_months > 24 else 1,
                help=horizon_info["desc"],
                key="periods_slider"
            )
            
            last_date = staples['date'].max()
            target_date = last_date + pd.DateOffset(months=forecast_periods)
            target_display = target_date.strftime('%B %Y')
    
    with col2:
        # ===== DATA QUALITY METER =====
        st.markdown("### 📊 Data Quality")
        
        commodity_data = staples[staples['cm_name'] == forecast_commodity]
        data_years = commodity_data['date'].dt.year.nunique()
        data_months = len(commodity_data)
        last_data_date = commodity_data['date'].max()
        
        # Quality score calculation
        expected_months = data_years * 12
        completeness = min(100, (data_months / expected_months) * 100) if expected_months > 0 else 0
        
        if data_months >= 48:
            quality_score = 95
            quality_text = "Excellent"
            quality_color = "#27ae60"
        elif data_months >= 24:
            quality_score = 80
            quality_text = "Good"
            quality_color = "#3498db"
        elif data_months >= 12:
            quality_score = 60
            quality_text = "Fair"
            quality_color = "#f39c12"
        else:
            quality_score = 40
            quality_text = "Limited"
            quality_color = "#e74c3c"
        
        # Display quality gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 24}},
            title={'text': f"Quality: {quality_text}", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': quality_color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 40], 'color': "#f8d7da"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#d4edda"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': quality_score
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Key stats
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Years of Data", data_years)
            st.metric("Last Update", last_data_date.strftime('%b %Y'))
        with col_y:
            st.metric("Total Records", data_months)
            st.metric("Completeness", f"{completeness:.0f}%")
    
    # ===== RELIABILITY INDICATOR =====
    st.markdown("---")
    reliability_col1, reliability_col2, reliability_col3 = st.columns([1, 2, 1])
    
    with reliability_col2:
        if reliability == "High":
            st.success(f"""
            ✅ **High Reliability Forecast**  
            Forecast until {target_display}  
            Based on strong historical data and short-term patterns
            """)
        elif reliability == "Moderate":
            st.warning(f"""
            📊 **Moderate Reliability Forecast**  
            Forecast until {target_display}  
            Good for trend analysis but expect some uncertainty
            """)
        else:
            st.error(f"""
            ⚠️ **Low Reliability Forecast**  
            Forecast until {target_display}  
            Shows general direction only - update regularly
            """)
    
    # ===== ADVANCED SETTINGS =====
    with st.expander("⚙️ Advanced Forecast Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.select_slider(
                "🎯 Confidence Level",
                options=[80, 85, 90, 95, 99],
                value=90,
                help="Higher confidence = wider prediction intervals",
                key="confidence_advanced"
            )
            
            # Confidence explanation
            conf_text = {
                80: "Narrower range, less certainty",
                85: "Balanced for trading decisions",
                90: "Standard statistical confidence",
                95: "Wider range, high certainty",
                99: "Very wide range, near certainty"
            }
            st.caption(conf_text[confidence_level])
        
        with col2:
            model_type = st.radio(
                "🤖 Forecast Model",
                ["Ensemble (Recommended)", "Conservative", "Aggressive"],
                help="Ensemble combines multiple models for best accuracy",
                key="model_advanced"
            )
            
            model_desc = {
                "Ensemble (Recommended)": "Balanced - best for most users",
                "Conservative": "Wider intervals - risk-averse",
                "Aggressive": "Narrower intervals - confident"
            }
            st.caption(model_desc[model_type])
        
        with col3:
            st.markdown("#### 📊 Display Options")
            show_seasonality = st.checkbox("Show seasonal patterns", value=True, key="show_seasonal")
            show_components = st.checkbox("Show trend breakdown", value=False, key="show_components")
            show_historical = st.checkbox("Show all historical data", value=True, key="show_historical")
    
    # ===== GENERATE FORECAST BUTTON =====
    if st.button("🚀 Generate Smart Forecast", type="primary", use_container_width=True, key="generate_btn"):
        
        # Validate data sufficiency
        if len(commodity_data) < 12:
            st.error(f"❌ Insufficient data for {forecast_commodity}. Need at least 12 months of historical data.")
            st.stop()
        
        with st.spinner("🤖 Analyzing patterns and generating intelligent forecast..."):
            
            # Generate forecast
            historical, forecast = generate_enhanced_forecast(staples, forecast_commodity, forecast_periods)
            
            if historical is None or forecast is None:
                st.error(f"❌ Cannot generate forecast for {forecast_commodity}")
                st.stop()
            
            # Calculate confidence intervals based on settings
            conf_multipliers = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}
            base_multiplier = conf_multipliers[confidence_level]
            
            # Adjust multiplier based on model type
            if model_type == "Conservative":
                multiplier = base_multiplier * 1.2
            elif model_type == "Aggressive":
                multiplier = base_multiplier * 0.8
            else:
                multiplier = base_multiplier
            
            # Calculate confidence intervals
            std_dev = historical['mp_price'].std()
            forecast['lower'] = forecast['price'] - (std_dev * multiplier)
            forecast['upper'] = forecast['price'] + (std_dev * multiplier)
            forecast['lower'] = forecast['lower'].clip(lower=0)
            
            # ===== MAIN FORECAST CHART =====
            st.markdown("## 📈 Price Forecast Visualization")
            
            fig = go.Figure()
            
            # Historical data
            if show_historical:
                fig.add_trace(go.Scatter(
                    x=historical['date'],
                    y=historical['mp_price'],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='#2c3e50', width=2),
                    hovertemplate='Date: %{x}<br>Price: %{y:.2f} KES<extra>Historical</extra>'
                ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['price'],
                mode='lines',
                name='Forecast',
                line=dict(color='#e74c3c', width=3, dash='dash'),
                hovertemplate='Date: %{x}<br>Forecast: %{y:.2f} KES<extra>Forecast</extra>'
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence Interval',
                hoverinfo='skip'
            ))
            
            # Forecast start line
            last_hist_date = historical['date'].max()
            fig.add_shape(
                type="line",
                x0=last_hist_date,
                y0=0,
                x1=last_hist_date,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=last_hist_date,
                y=0.95,
                yref="paper",
                text="Forecast Start",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-30,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
            
            fig.update_layout(
                title=f"{forecast_commodity} - Price Forecast to {target_display}",
                xaxis_title="Date",
                yaxis_title="Price (KES/kg)",
                hovermode='x unified',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== FORECAST METRICS =====
            st.markdown("## 📊 Key Forecast Metrics")
            
            last_price = historical['mp_price'].iloc[-1]
            avg_forecast = forecast['price'].mean()
            min_forecast = forecast['lower'].min()
            max_forecast = forecast['upper'].max()
            first_f = forecast['price'].iloc[0]
            last_f = forecast['price'].iloc[-1]
            trend_pct = ((last_f - first_f) / first_f) * 100 if first_f > 0 else 0
            
            # Create metric cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Current Price",
                    f"{last_price:.2f} KES",
                    help="Latest recorded price"
                )
            
            with col_m2:
                st.metric(
                    "Forecast Average",
                    f"{avg_forecast:.2f} KES",
                    delta=f"{((avg_forecast - last_price) / last_price * 100):+.1f}%",
                    help="Average predicted price over forecast period"
                )
            
            with col_m3:
                st.metric(
                    "Expected Range",
                    f"{min_forecast:.0f} - {max_forecast:.0f} KES",
                    help=f"{confidence_level}% confidence interval"
                )
            
            with col_m4:
                direction = "📈" if trend_pct > 0 else "📉" if trend_pct < 0 else "➡️"
                st.metric(
                    "Overall Trend",
                    f"{direction} {trend_pct:+.1f}%",
                    delta=f"Over {forecast_periods} months",
                    delta_color="normal" if trend_pct > 0 else "inverse",
                    help="Price change direction over forecast period"
                )
            
            # ===== SEASONAL PATTERN ANALYSIS =====
            if show_seasonality and len(historical) >= 24:
                st.markdown("## 📅 Seasonal Pattern Analysis")
                
                # Calculate seasonal indices
                hist_copy = historical.copy()
                hist_copy['month'] = hist_copy['date'].dt.month
                monthly_avg = hist_copy.groupby('month')['mp_price'].mean()
                overall_avg = hist_copy['mp_price'].mean()
                seasonal_indices = (monthly_avg / overall_avg - 1) * 100 if overall_avg > 0 else pd.Series([0]*12)
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Create two columns for chart and insights
                col_s1, col_s2 = st.columns([1.5, 1])
                
                with col_s1:
                    fig_seasonal = go.Figure()
                    
                    # Prepare data
                    valid_months = seasonal_indices.index.tolist()
                    valid_names = [month_names[i-1] for i in valid_months]
                    valid_values = seasonal_indices.values
                    
                    # Color based on positive/negative
                    colors = ['#e74c3c' if x > 0 else '#27ae60' for x in valid_values]
                    
                    fig_seasonal.add_trace(go.Bar(
                        x=valid_names,
                        y=valid_values,
                        marker_color=colors,
                        text=[f"{x:+.1f}%" for x in valid_values],
                        textposition='outside',
                        textfont=dict(size=11, color='black'),
                        hovertemplate='Month: %{x}<br>Variation: %{y:+.1f}%<extra></extra>'
                    ))
                    
                    # Add zero line
                    fig_seasonal.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    
                    # Calculate y-axis range with padding
                    y_max = max(abs(min(valid_values)), abs(max(valid_values))) * 1.3
                    
                    fig_seasonal.update_layout(
                        title="Monthly Price Variations vs Annual Average",
                        xaxis_title="Month",
                        yaxis_title="Variation from Average (%)",
                        height=350,
                        yaxis=dict(
                            range=[-y_max, y_max],
                            zeroline=True,
                            zerolinecolor='gray',
                            zerolinewidth=1
                        ),
                        margin=dict(t=40, b=40, l=50, r=50),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_seasonal, use_container_width=True)
                
                with col_s2:
                    # Best and worst months
                    best_idx = valid_values.argmin()
                    worst_idx = valid_values.argmax()
                    
                    best_price = monthly_avg.iloc[best_idx]
                    worst_price = monthly_avg.iloc[worst_idx]
                    savings = worst_price - best_price
                    
                    st.markdown("#### 💡 Seasonal Insights")
                    
                    st.markdown(f"""
                    <div style='background: #27ae6020; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                        <h4 style='margin:0; color: #27ae60;'>✅ Best Time to Buy</h4>
                        <p style='font-size: 1.2rem; margin:0;'><strong>{valid_names[best_idx]}</strong></p>
                        <p>Avg Price: {best_price:.1f} KES/kg</p>
                        <p>{valid_values[best_idx]:+.1f}% below average</p>
                    </div>
                    
                    <div style='background: #e74c3c20; padding: 1rem; border-radius: 10px;'>
                        <h4 style='margin:0; color: #e74c3c;'>⚠️ Highest Prices</h4>
                        <p style='font-size: 1.2rem; margin:0;'><strong>{valid_names[worst_idx]}</strong></p>
                        <p>Avg Price: {worst_price:.1f} KES/kg</p>
                        <p>{valid_values[worst_idx]:+.1f}% above average</p>
                        <p><strong>Save {savings:.1f} KES/kg</strong> by buying in {valid_names[best_idx]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ===== FORECAST COMPONENTS =====
            if show_components:
                st.markdown("## 🔍 Forecast Components Breakdown")
                
                # Decompose forecast
                x = np.arange(len(forecast))
                z = np.polyfit(x, forecast['price'], 1)
                trend_line = np.poly1d(z)
                trend_component = trend_line(x)
                seasonal_component = forecast['price'] - trend_component
                
                fig_components = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Combined Forecast', 'Trend Component (Long-term direction)', 
                                  'Seasonal Component (Monthly patterns)'),
                    vertical_spacing=0.1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=forecast['price'], 
                              mode='lines', name='Forecast', line=dict(color='#e74c3c', width=2)),
                    row=1, col=1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=trend_component, 
                              mode='lines', name='Trend', line=dict(color='#27ae60', width=2)),
                    row=2, col=1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=seasonal_component, 
                              mode='lines', name='Seasonal', line=dict(color='#3498db', width=2)),
                    row=3, col=1
                )
                
                fig_components.update_layout(height=600, showlegend=False)
                fig_components.update_xaxes(title_text="Date", row=3, col=1)
                fig_components.update_yaxes(title_text="KES/kg", row=1, col=1)
                fig_components.update_yaxes(title_text="KES/kg", row=2, col=1)
                fig_components.update_yaxes(title_text="KES/kg", row=3, col=1)
                
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Explanation
                st.info("""
                **📚 Understanding the components:**
                - **Trend**: The long-term price direction (up/down/stable) - use for strategic planning
                - **Seasonal**: Regular monthly patterns driven by harvest cycles and holidays
                - **Combined**: The final forecast = Trend + Seasonal
                """)
            
            # ===== SMART RECOMMENDATIONS =====
            st.markdown("## 💡 Smart Recommendations")
            
            # Calculate volatility
            volatility = (historical['mp_price'].std() / historical['mp_price'].mean()) * 100 if historical['mp_price'].mean() > 0 else 0
            
            # Determine market condition
            if trend_pct > 15:
                market_outlook = "strongly rising"
                buyer_action = "Buy sooner rather than later"
                seller_action = "Hold if possible, prices expected to increase"
                urgency = "High"
            elif trend_pct > 5:
                market_outlook = "moderately rising"
                buyer_action = "Gradual purchasing recommended"
                seller_action = "Good time to sell"
                urgency = "Medium"
            elif trend_pct < -15:
                market_outlook = "strongly falling"
                buyer_action = "Delay purchases, wait for lower prices"
                seller_action = "Sell now before prices drop further"
                urgency = "High"
            elif trend_pct < -5:
                market_outlook = "moderately falling"
                buyer_action = "Buy in stages as prices drop"
                seller_action = "Consider holding if you can"
                urgency = "Medium"
            else:
                market_outlook = "stable"
                buyer_action = "Normal purchasing patterns"
                seller_action = "Normal selling patterns"
                urgency = "Low"
            
            # Create recommendation columns
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown(f"""
                <div style='background: #3498db20; padding: 1.2rem; border-radius: 10px; border-left: 5px solid #3498db; height: 180px;'>
                    <h4 style='margin:0; color: #3498db;'>🛒 For Buyers</h4>
                    <p style='margin-top: 0.5rem; font-size: 1.1rem;'><strong>Market: {market_outlook}</strong></p>
                    <p style='margin: 0.3rem 0;'>✅ {buyer_action}</p>
                    <p style='margin: 0.3rem 0; color: #7f8c8d;'>Urgency: {urgency}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                st.markdown(f"""
                <div style='background: #f39c1220; padding: 1.2rem; border-radius: 10px; border-left: 5px solid #f39c12; height: 180px;'>
                    <h4 style='margin:0; color: #f39c12;'>🌾 For Sellers</h4>
                    <p style='margin-top: 0.5rem; font-size: 1.1rem;'><strong>Opportunity: {market_outlook}</strong></p>
                    <p style='margin: 0.3rem 0;'>💰 {seller_action}</p>
                    <p style='margin: 0.3rem 0; color: #7f8c8d;'>Best selling window: {
                        "Now" if trend_pct > 0 else "Wait" if trend_pct < -10 else "Monitor market"
                    }</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Volatility alert
            st.markdown("---")
            if volatility > 30:
                st.error(f"⚠️ **HIGH VOLATILITY ALERT**: Price swings of ±{volatility:.0f}% are common. Consider hedging strategies and avoid large positions.")
            elif volatility > 15:
                st.warning(f"📊 **MODERATE VOLATILITY**: Normal market fluctuations of ±{volatility:.0f}% expected. Plan accordingly.")
            else:
                st.success(f"✅ **LOW VOLATILITY**: Stable market with only ±{volatility:.0f}% variation. Good for planning.")
            
            # ===== DOWNLOAD FORECAST DATA =====
            st.markdown("## 📥 Export Forecast Data")
            
            # Create download dataframe
            download_df = pd.DataFrame({
                'Year': forecast['date'].dt.year,
                'Month': forecast['date'].dt.strftime('%b'),
                'Month_Num': forecast['date'].dt.month,
                'Forecast_Price_KES': forecast['price'].round(2),
                'Lower_Bound_KES': forecast['lower'].round(2),
                'Upper_Bound_KES': forecast['upper'].round(2),
                'Confidence_Level': f'{confidence_level}%'
            })
            
            # Add seasonal variation if available
            if show_seasonality and len(historical) >= 24:
                forecast_months = forecast['date'].dt.month
                seasonal_map = {m: v for m, v in zip(valid_months, valid_values)}
                download_df['Seasonal_Variation_%'] = [f"{seasonal_map.get(m, 0):+.1f}%" for m in forecast_months]
            
            # Format for display
            display_df = download_df[['Year', 'Month', 'Forecast_Price_KES', 'Lower_Bound_KES', 'Upper_Bound_KES']]
            if 'Seasonal_Variation_%' in download_df.columns:
                display_df['Seasonal'] = download_df['Seasonal_Variation_%']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download buttons
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Forecast (CSV)",
                    data=csv,
                    file_name=f"{forecast_commodity.replace(' ', '_')}_forecast_{target_date.strftime('%Y_%m')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv"
                )
            
            with col_d2:
                # Create summary JSON
                summary = {
                    'commodity': forecast_commodity,
                    'forecast_date': target_date.strftime('%Y-%m-%d'),
                    'current_price': float(last_price),
                    'avg_forecast': float(avg_forecast),
                    'trend_percent': float(trend_pct),
                    'confidence_level': confidence_level,
                    'reliability': reliability
                }
                import json
                json_str = json.dumps(summary, indent=2)
                st.download_button(
                    label="📋 Download Summary (JSON)",
                    data=json_str,
                    file_name=f"{forecast_commodity.replace(' ', '_')}_summary.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_json"
                )
            
            # ===== LONG-TERM DISCLAIMER =====
            if reliability == "Low":
                st.markdown("---")
                st.warning("""
                <div style='background: #fff3cd; padding: 1rem; border-radius: 5px;'>
                    <h4 style='margin:0; color: #856404;'>⚠️ Important Note on Long-term Forecasts</h4>
                    <p style='margin-top: 0.5rem;'>Forecasts beyond 3 years show general trends only. Many factors can affect actual prices:</p>
                    <ul>
                        <li>Weather patterns and climate change</li>
                        <li>Economic policies and inflation</li>
                        <li>Global market conditions</li>
                        <li>Technology and farming practices</li>
                    </ul>
                    <p><strong>Use this forecast for strategic planning, not precise budgeting.</strong> Update regularly as new data becomes available.</p>
                </div>
                """, unsafe_allow_html=True)
# ============================================================================
# PAGE: MARKET INTELLIGENCE
# ============================================================================
elif page == "📈 Market Intelligence":
    st.markdown("<h1 class='main-header'>📈 Market Intelligence</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>Deep Market Analysis</h4>
    <p>Comprehensive insights combining regional variations, seasonal patterns, and market dynamics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Regional Comparison", "Seasonal Patterns", "Price Correlations"])
    
    # TAB 1: Regional Comparison
    with tab1:
        st.markdown("### Regional Price Dynamics")
        
        region_commodity = st.selectbox(
            "Select Commodity",
            sorted(staples['cm_name'].unique()),
            key='region_comm'
        )
        
        regional_data = staples[staples['cm_name'] == region_commodity].copy()
        
        if len(regional_data) > 0:
            # Regional averages
            region_avg = regional_data.groupby('adm1_name')['mp_price'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=region_avg.values,
                y=region_avg.index,
                orientation='h',
                title=f"Average Prices by Region - {region_commodity}",
                labels={'x': 'Price (KES/kg)', 'y': 'Region'},
                color=region_avg.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional statistics
            st.markdown("### Regional Statistics")
            stats = regional_data.groupby('adm1_name')['mp_price'].agg(['mean', 'std', 'min', 'max']).round(2)
            stats.columns = ['Average', 'Std Dev', 'Min', 'Max']
            st.dataframe(stats, use_container_width=True)
    
    # TAB 2: Seasonal Patterns
    with tab2:
        st.markdown("### Seasonal Price Patterns")
        
        seasonal_commodity = st.selectbox(
            "Select Commodity",
            sorted(staples['cm_name'].unique()),
            key='seasonal_comm'
        )
        
        seasonal_data = staples[staples['cm_name'] == seasonal_commodity].copy()
        
        if len(seasonal_data) > 0:
            seasonal_data['month'] = seasonal_data['date'].dt.month
            monthly_pattern = seasonal_data.groupby('month')['mp_price'].agg(['mean', 'std']).reset_index()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pattern['month_name'] = monthly_pattern['month'].map(lambda x: month_names[x-1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_pattern['month_name'],
                y=monthly_pattern['mean'],
                mode='lines+markers',
                name='Average Price',
                line=dict(color='#3498db', width=3),
                error_y=dict(
                    type='data',
                    array=monthly_pattern['std'],
                    visible=True
                )
            ))
            fig.update_layout(
                title=f"Seasonal Pattern - {seasonal_commodity}",
                xaxis_title="Month",
                yaxis_title="Price (KES/kg)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best and worst months
            best_idx = monthly_pattern['mean'].idxmin()
            worst_idx = monthly_pattern['mean'].idxmax()
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Best Month: {monthly_pattern.loc[best_idx, 'month_name']} "
                          f"({monthly_pattern.loc[best_idx, 'mean']:.2f} KES)")
            with col2:
                st.warning(f"⚠️ Highest Month: {monthly_pattern.loc[worst_idx, 'month_name']} "
                          f"({monthly_pattern.loc[worst_idx, 'mean']:.2f} KES)")
    
    # TAB 3: Price Correlations
    with tab3:
        st.markdown("### Commodity Price Correlations")
        st.info("Select multiple commodities to see how their prices move together.")
        
        top_commodities = staples['cm_name'].value_counts().head(8).index.tolist()
        selected = st.multiselect(
            "Select commodities (2-8)",
            options=sorted(staples['cm_name'].unique()),
            default=top_commodities[:4],
            max_selections=8
        )
        
        if len(selected) >= 2:
            # Create correlation matrix
            corr_data = []
            valid_comms = []
            
            for comm in selected:
                monthly = staples[staples['cm_name'] == comm].groupby('date')['mp_price'].mean()
                if len(monthly) > 0:
                    corr_data.append(monthly)
                    valid_comms.append(comm)
            
            if len(valid_comms) >= 2:
                corr_df = pd.DataFrame(corr_data, index=valid_comms).T
                corr_matrix = corr_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Price Correlation Matrix",
                    labels=dict(color='Correlation'),
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    zmin=-1, zmax=1,
                    text_auto='.2f'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **How to read:**
                - **Blue (close to 1)**: Prices move together
                - **Red (close to -1)**: Prices move opposite
                - **White (close to 0)**: No relationship
                """)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-header'>ℹ️ About This Platform</h1>", unsafe_allow_html=True)
    
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
        st.markdown("""
        ### 📈 Platform Stats
        """)
        st.metric("Total Records", f"{len(staples):,}")
        st.metric("Commodities", staples['cm_name'].nunique())
        st.metric("Markets", staples['mkt_name'].nunique())
        st.metric("Years", f"{staples['date'].min().year} - {staples['date'].max().year}")

# ============================================================================
# PAGE: DEVELOPER
# ============================================================================
elif page == "👨‍💻 Developer":
    st.markdown("<h1 class='main-header'>👨‍💻 Developer</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    
    with col2:
        st.markdown("""
        # Stephen Muema
        ## Data Scientist & Machine Learning Engineer
        
        Transforming complex datasets into actionable insights.
        
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Kaks753)
        [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://muemastephenportfolio.netlify.app/)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Skills
    - Python, Pandas, NumPy
    - Machine Learning
    - Data Visualization
    - Streamlit
    
    ### About This Project
    The Kenya Food Price Intelligence System demonstrates end-to-end data science 
    capabilities from data cleaning to interactive deployment.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem 0;'>
    <p><strong>Kenya Food Price Intelligence System</strong> | Developed by Stephen Muema | 2024</p>
    <p>Data Source: World Food Programme (WFP)</p>
</div>
""", unsafe_allow_html=True)