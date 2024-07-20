import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import telegram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Telegram setup
TELEGRAM_BOT_TOKEN = '7431234538:AAGIVdVSVymRj2nsOkpkTICxYl5EnSVSR3k' #os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = '5184466403' #os.environ.get('TELEGRAM_CHAT_ID')
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info("Telegram message sent successfully")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# Read the CSV file
df = pd.read_csv('nifty500_companies.csv')

# Define the periods
periods = {
    '7d': 7, '15d': 15, '1m': 30, '2m': 60, '3m': 90,
    '6m': 180, '1y': 365, '3y': 1095, '5y': 1825
}

# Initial investment
initial_investment = 100000

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
        current_value = (initial_investment / initial_price) * final_price
        
        return returns_percentage, current_value
    except Exception as e:
        logging.error(f"Error processing {symbol} ({company_name}): {e}")
        return None

# Dictionary to store results
results = {period: [] for period in periods}

# Calculate returns for each stock and period
for index, row in df.iterrows():
    symbol = row['Symbol']
    company_name = row['Company Name']
    logging.info(f"Processing stock {index + 1}/{len(df)}: {symbol} ({company_name})")
    for period, days in periods.items():
        result = calculate_returns(symbol, company_name, days)
        if result:
            returns_percentage, current_value = result
            results[period].append({
                'Symbol': symbol,
                'Company Name': company_name,
                'Returns %': returns_percentage,
                'Current Value': current_value
            })

# Save top 25 results for each period to separate CSV files
for period, data in results.items():
    df_period = pd.DataFrame(data)
    df_period_sorted = df_period.sort_values('Returns %', ascending=False)
    top_25 = df_period_sorted.head(25)
    top_25.to_csv(f'top_25_stock_performance_{period}.csv', index=False)
    logging.info(f"Saved top 25 stock performance for {period} to CSV")

# Get top performers for each period
top_performers = pd.DataFrame()
for period, data in results.items():
    df_period = pd.DataFrame(data)
    top_period = df_period.nlargest(25, 'Returns %')
    top_period['Period'] = period
    top_performers = pd.concat([top_performers, top_period])

# Save top performers to CSV
top_performers.to_csv('all_top_performers.csv', index=False)
logging.info("Saved all top performers to CSV")

# Prepare message for Telegram
telegram_message = "Top 5 Performers for Each Period:\n\n"
for period in periods:
    telegram_message += f"{period}:\n"
    top_5 = top_performers[top_performers['Period'] == period].head()
    for _, row in top_5.iterrows():
        telegram_message += f"{row['Symbol']} ({row['Company Name']}): {row['Returns %']:.2f}%\n"
    telegram_message += "\n"

# Send Telegram message
send_telegram_message(telegram_message)

# Plot top performers for each period
plt.figure(figsize=(20, 15))
for i, period in enumerate(periods, 1):
    plt.subplot(3, 3, i)
    data = top_performers[top_performers['Period'] == period].head(10)
    sns.barplot(x='Symbol', y='Returns %', data=data)
    plt.title(f'Top 10 Performers - {period}')
    plt.xticks(rotation=90)
    plt.tight_layout()
plt.savefig('top_performers_plot.png')
plt.close()

# ML-based stock recommendation
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

# Prepare data for top performing stocks
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
    send_telegram_message("Unable to make ML-based recommendations due to insufficient data.")
else:
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

    def predict_stock(symbol, company_name):
        stock_data = get_stock_data(symbol, company_name, days=60)
        if stock_data is None or len(stock_data) < 50:
            return None
        
        recent_data = stock_data.iloc[-1][features]
        recent_data_scaled = scaler.transform([recent_data])
        probability = rf_model.predict_proba(recent_data_scaled)[0][1]
        return probability

    stock_predictions = []
    for _, row in top_stocks.iterrows():
        symbol = row['Symbol']
        company_name = row['Company Name']
        logging.info(f"Predicting for stock: {symbol} ({company_name})")
        prediction = predict_stock(symbol, company_name)
        if prediction is not None:
            stock_predictions.append((symbol, company_name, prediction))

    if not stock_predictions:
        logging.warning("No valid predictions were made")
        send_telegram_message("Unable to make ML-based recommendations due to insufficient prediction data.")
    else:
        top_5_stocks = sorted(stock_predictions, key=lambda x: x[2], reverse=True)[:5]

        # Save ML recommendations to CSV
        ml_recommendations_df = pd.DataFrame(top_5_stocks, columns=['Symbol', 'Company Name', 'Probability'])
        ml_recommendations_df.to_csv('ml_recommendations.csv', index=False)
        logging.info("Saved ML recommendations to CSV")

        # Send ML recommendations via Telegram
        ml_message = "Top 5 stocks recommended by ML model:\n\n"
        for _, row in ml_recommendations_df.iterrows():
            ml_message += f"{row['Symbol']} ({row['Company Name']}): {row['Probability']:.2f}\n"
        send_telegram_message(ml_message)

        # Plot ML recommendations
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Symbol', y='Probability', data=ml_recommendations_df)
        plt.title('Top 5 Stocks Recommended by ML Model')
        plt.xlabel('Symbol')
        plt.ylabel('Probability of Positive Return')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        for i, v in enumerate(ml_recommendations_df['Probability']):
            plt.text(i, v, f"{ml_recommendations_df['Company Name'].iloc[i]}", ha='center', va='bottom', rotation=90)
        plt.tight_layout()
        plt.savefig('ml_recommendations_plot.png')
        plt.close()

logging.info("Analysis completed")