import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

def get_stock_data(symbol, company_name, days=365*2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.Ticker(f"{symbol}.NS").history(start=start_date, end=end_date)
        if len(data) < 50:
            logging.warning(f"Insufficient data for {symbol} ({company_name})")
            return None
        
        data['Returns'] = data['Close'].pct_change()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data.dropna()
    except Exception as e:
        logging.error(f"Error processing {symbol} ({company_name}): {e}")
        return None

def run_ml_predictions(top_performers):
    top_stocks = top_performers[['Symbol', 'Company Name']].drop_duplicates()
    all_stock_data = []
    
    for _, row in top_stocks.iterrows():
        symbol = row['Symbol']
        company_name = row['Company Name']
        logging.info(f"Collecting data for ML model - Stock: {symbol} ({company_name})")
        stock_data = get_stock_data(symbol, company_name)
        if stock_data is not None and len(stock_data) >= 50:
            stock_data['Symbol'] = symbol
            stock_data['Company Name'] = company_name
            all_stock_data.append(stock_data)

    if not all_stock_data:
        logging.error("No valid stock data available for ML model")
        return None

    combined_data = pd.concat(all_stock_data, ignore_index=True)

    features = ['Returns', 'SMA_20', 'SMA_50', 'Volatility']
    X = combined_data[features]
    y = combined_data['Close'].pct_change().shift(-1).fillna(0)  # Predict next day's return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train > 0) 
    logging.info("ML model training completed")

    stock_predictions = []
    for _, row in top_stocks.iterrows():
        symbol = row['Symbol']
        company_name = row['Company Name']
        logging.info(f"Predicting for stock: {symbol} ({company_name})")
        stock_data = get_stock_data(symbol, company_name, days=300)
        if stock_data is not None and len(stock_data) >= 50:
            recent_data = stock_data.iloc[-1][features]
            recent_data_scaled = scaler.transform([recent_data])
            probability = rf_model.predict_proba(recent_data_scaled)[0][1]
            stock_predictions.append((symbol, company_name, probability))

    if not stock_predictions:
        logging.warning("No valid predictions were made")
        return None

    top_5_stocks = sorted(stock_predictions, key=lambda x: x[2], reverse=True)[:5]
    ml_recommendations_df = pd.DataFrame(top_5_stocks, columns=['Symbol', 'Company Name', 'Probability'])
    ml_recommendations_df.to_csv('ml_recommendations.csv', index=False)
    logging.info("Saved ML recommendations to CSV")

    return ml_recommendations_df