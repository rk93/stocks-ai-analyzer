import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from config import PERIODS, INITIAL_INVESTMENT

def calculate_returns(symbol, company_name, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.Ticker(f"{symbol}.NS").history(start=start_date, end=end_date)
        if len(data) < 2:
            return None
        
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        
        returns_percentage = ((final_price - initial_price) / initial_price) * 100
        current_value = (INITIAL_INVESTMENT / initial_price) * final_price
        
        return returns_percentage, current_value
    except Exception as e:
        logging.error(f"Error processing {symbol} ({company_name}): {e}")
        return None

def run_stock_analysis(test_mode, num_test_stocks):
    df = pd.read_csv('nifty500_companies.csv')
    
    if test_mode:
        df = df.sample(n=num_test_stocks, random_state=42)

    results = {period: [] for period in PERIODS}

    for index, row in df.iterrows():
        symbol = row['Symbol']
        company_name = row['Company Name']
        logging.info(f"Processing stock {index + 1}/{len(df)}: {symbol} ({company_name})")
        for period, days in PERIODS.items():
            result = calculate_returns(symbol, company_name, days)
            if result:
                returns_percentage, current_value = result
                results[period].append({
                    'Symbol': symbol,
                    'Company Name': company_name,
                    'Returns %': returns_percentage,
                    'Current Value': current_value
                })

    top_performers = pd.DataFrame()
    for period, data in results.items():
        df_period = pd.DataFrame(data)
        top_period = df_period.nlargest(25, 'Returns %')
        top_period['Period'] = period
        top_performers = pd.concat([top_performers, top_period])

    top_performers.to_csv('all_top_performers.csv', index=False)
    logging.info("Saved all top performers to CSV")

    return top_performers