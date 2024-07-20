import logging
from stock_analysis import run_stock_analysis
from ml_model import run_ml_predictions
from telegram_utils import send_telegram_message
from config import TEST_MODE, NUM_TEST_STOCKS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        top_performers = run_stock_analysis(TEST_MODE, NUM_TEST_STOCKS)
        ml_recommendations = run_ml_predictions(top_performers)
        
        send_telegram_message(top_performers, 'performance')
        if ml_recommendations is not None:
            send_telegram_message(ml_recommendations, 'ml_recommendations')
        
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        send_telegram_message(f"An error occurred: {e}", 'error')

if __name__ == "__main__":
    main()