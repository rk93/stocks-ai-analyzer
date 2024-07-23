import os

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TELEGRAM_GROUP_CHAT_ID = os.environ.get('TELEGRAM_GROUP_CHAT_ID')

# Validate Telegram configuration
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment variables")
if not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID is not set in the environment variables")

# Test mode configuration
TEST_MODE = os.environ.get('TEST_MODE', 'True').lower() == 'true'
NUM_TEST_STOCKS = int(os.environ.get('NUM_TEST_STOCKS', '20'))

# Analysis configuration
INITIAL_INVESTMENT = int(os.environ.get('INITIAL_INVESTMENT', '100000'))
PERIODS = {
    '5d': 5, '7d': 7, '15d': 15, '1m': 30, '2m': 60, '3m': 90,
    '6m': 180, '1y': 365, '3y': 1095, '5y': 1825
}