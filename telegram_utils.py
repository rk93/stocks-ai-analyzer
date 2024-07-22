import requests
from datetime import datetime
import logging
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_GROUP_CHAT_ID

def send_telegram_message(data, message_type):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if message_type == 'performance':
            message = format_performance_message(data, current_date)
        elif message_type == 'ml_recommendations':
            message = format_ml_recommendations_message(data, current_date)
        elif message_type == 'error':
            message = f"⚠️ Error: {data}"
        else:
            raise ValueError(f"Unknown message type: {message_type}")

        params = {
            "chat_id": TELEGRAM_GROUP_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        logging.info("Telegram message sent successfully to the group")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def format_performance_message(top_performers, current_date):
    message = f"🚀 <b>Stock Market Insights for {current_date}</b> 🚀\n\n"
    message += "Hey there, stock market enthusiast! 📈 Here's your daily dose of market magic:\n\n"

    emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    periods = top_performers['Period'].unique()

    for period in periods:
        message += f"🔥 <b>Top Performers - {period.upper()}</b> 🔥\n"
        top_5 = top_performers[top_performers['Period'] == period].head()
        for i, row in top_5.iterrows():
            emoji = emojis[i] if i < len(emojis) else "🔹"
            message += f"{emoji} <b>{row['Symbol']}</b> ({row['Company Name']}): {row['Returns %']:.2f}% 💰\n"
        message += "\n"

    return message

def format_ml_recommendations_message(ml_recommendations, current_date):
    message = f"🤖 <b>AI-Powered Stock Picks for {current_date}</b> 🤖\n\n"
    message += "Our crystal ball (ahem, ML model) predicts these stocks might just skyrocket! 🚀\n\n"

    emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (_, row) in enumerate(ml_recommendations.iterrows()):
        emoji = emojis[i] if i < len(emojis) else "🔹"
        message += f"{emoji} <b>{row['Symbol']}</b> ({row['Company Name']}): {row['Probability']:.2f} confidence 🔮\n"
    
    message += "\nRemember, even AI can't predict the future perfectly. Always do your own research! 📚💼"
    return message