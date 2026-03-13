"""
Download food price datasets from multiple sources
"""
import pandas as pd
import requests
import os
from pathlib import Path

def download_wfp_data():
    """
    Download WFP Global Food Prices Database
    Source: https://data.humdata.org/dataset/global-wfp-food-prices
    """
    print("Downloading WFP Global Food Prices data...")
    
    # Alternative: Use the global dataset and filter for Kenya
    # HDX API endpoint for WFP global food prices
    url = "https://data.humdata.org/api/3/action/package_show?id=global-wfp-food-prices"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Get the CSV resource
            resources = data['result']['resources']
            csv_resource = [r for r in resources if r['format'].upper() == 'CSV'][0]
            csv_url = csv_resource['url']
            
            print(f"Downloading from: {csv_url}")
            df = pd.read_csv(csv_url)
            
            # Filter for Kenya
            kenya_df = df[df['adm0_name'] == 'Kenya'].copy()
            
            output_path = Path('data/raw/wfp_food_prices_kenya.csv')
            kenya_df.to_csv(output_path, index=False)
            print(f"✓ Downloaded WFP Kenya data: {len(kenya_df)} rows, {len(kenya_df.columns)} columns")
            print(f"  Saved to: {output_path}")
            return True
        else:
            print(f"✗ Failed to fetch WFP data: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error downloading WFP data: {str(e)}")
        return False

def download_fao_data():
    """
    Download FAO Food Price data for Kenya
    Using FAO API
    """
    print("\nDownloading FAO Food Price data for Kenya...")
    
    try:
        # FAO Food Price Index data (global context)
        fao_url = "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data_deflated.csv"
        
        df = pd.read_csv(fao_url, encoding='latin1')
        output_path = Path('data/raw/fao_food_price_index.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ Downloaded FAO Food Price Index: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading FAO data: {str(e)}")
        return False

def download_sample_kenya_data():
    """
    Create sample Kenya food price data if downloads fail
    This uses representative data structure
    """
    print("\nCreating sample Kenya food price dataset...")
    
    # Sample data based on WFP structure
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='MS')
    
    foods = [
        ('Maize (white)', 'KG'),
        ('Rice', 'KG'),
        ('Wheat flour', 'KG'),
        ('Beans (dry)', 'KG'),
        ('Sugar', 'KG'),
        ('Cooking oil (vegetable)', 'L'),
        ('Milk (fresh)', 'L'),
        ('Eggs', 'Piece'),
    ]
    
    markets = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
    
    data = []
    import numpy as np
    np.random.seed(42)
    
    for date in dates:
        for food, unit in foods:
            base_price = {
                'Maize (white)': 45,
                'Rice': 120,
                'Wheat flour': 55,
                'Beans (dry)': 110,
                'Sugar': 130,
                'Cooking oil (vegetable)': 280,
                'Milk (fresh)': 60,
                'Eggs': 15,
            }[food]
            
            # Add trend and seasonality
            months_since_start = (date.year - 2020) * 12 + date.month
            trend = base_price * (1 + 0.08 * months_since_start / 12)  # 8% annual inflation
            seasonal = base_price * 0.1 * np.sin(2 * np.pi * date.month / 12)
            
            for market in markets:
                market_factor = np.random.uniform(0.95, 1.05)
                noise = np.random.uniform(-0.05, 0.05) * trend
                price = trend + seasonal + noise * market_factor
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'admin1': 'Kenya',
                    'market': market,
                    'commodity': food,
                    'unit': unit,
                    'price': round(price, 2),
                    'currency': 'KES',
                    'source': 'Sample Data'
                })
    
    df = pd.DataFrame(data)
    output_path = Path('data/raw/kenya_food_prices_sample.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Created sample dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Saved to: {output_path}")
    return True

if __name__ == "__main__":
    os.chdir('/home/user/webapp')
    
    print("=" * 60)
    print("Food Price Data Download Script")
    print("=" * 60)
    
    wfp_success = download_wfp_data()
    fao_success = download_fao_data()
    sample_success = download_sample_kenya_data()
    
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  WFP Data: {'✓ Success' if wfp_success else '✗ Failed'}")
    print(f"  FAO Data: {'✓ Success' if fao_success else '✗ Failed'}")
    print(f"  Sample Data: {'✓ Success' if sample_success else '✗ Failed'}")
    print("=" * 60)
