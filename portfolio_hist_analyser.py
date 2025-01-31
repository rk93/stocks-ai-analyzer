import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    def __init__(self, total_investment_gbp: float = 10000.0):
        self.logger = self._setup_logging()
        self.total_investment_gbp = total_investment_gbp
        self.portfolio = [
            'AAPL', 'TSLA', 'RR.L', 'NVDA', 'META',
            'AVGO', 'PLTR', 'GOOGL', 'MSFT', 'MSTR'
        ]
        self.gbp_usd_rate = self._get_gbp_usd_rate()
        self.max_position_size = total_investment_gbp * 0.15  # Max 15% per position
        
        # Portfolio tracking
        self.positions = {}  # Track current positions
        self.cash_gbp = total_investment_gbp  # Available cash
        self.trade_history = []  # Track all trades
        
        # ML components
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.ml_ready = False

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _get_gbp_usd_rate(self) -> float:
        try:
            gbp_usd = yf.Ticker("GBPUSD=X")
            return gbp_usd.info.get('regularMarketPrice', 1.25)
        except:
            return 1.25

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> tuple:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions"""
        try:
            # Get SPY data as market proxy
            spy = yf.Ticker("SPY")
            hist = spy.history(period='1mo')
            
            market_trend = 'Bullish' if hist['Close'].iloc[-1] > hist['Close'].mean() else 'Bearish'
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # Get VIX data for market fear
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period='1d')
            vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else None
            
            market_conditions = {
                'trend': market_trend,
                'volatility': volatility,
                'vix': vix_level,
                'risk_level': 'High' if (vix_level and vix_level > 25) else 'Moderate' if (vix_level and vix_level > 15) else 'Low'
            }
            
            return market_conditions
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {'trend': 'Unknown', 'volatility': 0, 'vix': None, 'risk_level': 'Unknown'}

    def analyze_stock(self, symbol: str, date: str = None) -> Dict:
        """Analyze a single stock"""
        try:
            stock = yf.Ticker(symbol)
            if date:
                # Get data for specific date and a few days before
                end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
                start_date = end_date - timedelta(days=180)  # 6 months of history for analysis
                hist = stock.history(start=start_date, end=end_date)
            else:
                hist = stock.history(period='6mo')
            
            if len(hist) < 50:  # Need enough historical data
                return {}
            
            # Use the last day's data for analysis
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            
            # Calculate technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close']).iloc[-1]
            macd, signal = self._calculate_macd(hist['Close'])
            
            # Volume analysis
            volume_sma = hist['Volume'].rolling(window=20).mean().iloc[-1]
            volume_trend = hist['Volume'].iloc[-1] / volume_sma
            
            # Support and Resistance
            recent_lows = hist['Low'].tail(20)
            recent_highs = hist['High'].tail(20)
            support = recent_lows.min()
            resistance = recent_highs.max()
            
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'prev_close': prev_close,
                'change_percent': ((current_price - prev_close) / prev_close) * 100,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi': rsi,
                'macd': macd.iloc[-1],
                'macd_signal': signal.iloc[-1],
                'volume_trend': volume_trend,
                'support': support,
                'resistance': resistance,
                'volatility': hist['Close'].pct_change().std() * np.sqrt(252)
            }
            
            # Add ML prediction if model is ready
            if self.ml_ready:
                features = self._prepare_ml_features(hist)
                if not features.empty:
                    features_scaled = self.scaler.transform(features.iloc[[-1]])
                    analysis['ml_probability'] = self.model.predict_proba(features_scaled)[0][1]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {}

    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns'] = data['Close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['rsi'] = self._calculate_rsi(data['Close'])
            
            # Volume features
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            
            # Trend features
            for period in [20, 50, 200]:
                sma = data['Close'].rolling(period).mean()
                features[f'sma_{period}_ratio'] = data['Close'] / sma
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            return pd.DataFrame()

    def generate_signals(self, analysis: Dict, market_conditions: Dict) -> Dict:
        """Generate trading signals based on technical and ML analysis"""
        signals = {
            'action': 'HOLD',
            'confidence': 0,
            'reasons': [],
            'suggested_position_size_gbp': 0,
            'entry_price': analysis.get('current_price', 0),
            'stop_loss': 0,
            'target_price': 0
        }
        
        if not analysis:
            return signals
        
        # Initialize scoring
        buy_score = 0
        sell_score = 0
        
        # RSI Analysis
        rsi = analysis.get('rsi', 50)
        if rsi < 30:
            buy_score += 2
            signals['reasons'].append("Oversold (RSI)")
        elif rsi > 70:
            sell_score += 2
            signals['reasons'].append("Overbought (RSI)")
        
        # Moving Average Analysis
        current_price = analysis['current_price']
        sma_20 = analysis.get('sma_20', current_price)
        sma_50 = analysis.get('sma_50', current_price)
        sma_200 = analysis.get('sma_200', current_price)
        
        if current_price > sma_20 > sma_50:
            buy_score += 1
            signals['reasons'].append("Bullish MA Trend")
        elif current_price < sma_20 < sma_50:
            sell_score += 1
            signals['reasons'].append("Bearish MA Trend")
        
        # MACD Analysis
        if analysis['macd'] > analysis['macd_signal']:
            buy_score += 1
            signals['reasons'].append("MACD Bullish")
        else:
            sell_score += 1
            signals['reasons'].append("MACD Bearish")
        
        # Volume Analysis
        volume_trend = analysis.get('volume_trend', 1)
        if volume_trend > 1.5:
            if analysis.get('change_percent', 0) > 0:
                buy_score += 1
                signals['reasons'].append("High Volume Upward Move")
            else:
                sell_score += 1
                signals['reasons'].append("High Volume Downward Move")
        
        # Support/Resistance Analysis
        if current_price <= analysis['support'] * 1.02:
            buy_score += 2
            signals['reasons'].append("Near Support")
        elif current_price >= analysis['resistance'] * 0.98:
            sell_score += 2
            signals['reasons'].append("Near Resistance")
        
        # ML Prediction (if available)
        if 'ml_probability' in analysis:
            if analysis['ml_probability'] > 0.7:
                buy_score += 2
                signals['reasons'].append(f"ML Bullish ({analysis['ml_probability']:.1%})")
            elif analysis['ml_probability'] < 0.3:
                sell_score += 2
                signals['reasons'].append(f"ML Bearish ({analysis['ml_probability']:.1%})")
        
        # Market Conditions Adjustment
        if market_conditions['trend'] == 'Bearish':
            buy_score *= 0.8  # Reduce buy signals in bearish market
            signals['reasons'].append("Bearish Market")
        elif market_conditions['trend'] == 'Bullish':
            sell_score *= 0.8  # Reduce sell signals in bullish market
            signals['reasons'].append("Bullish Market")
        
        # Calculate final confidence
        total_score = max(buy_score, sell_score)
        max_possible_score = 8  # Update if scoring system changes
        signals['confidence'] = (total_score / max_possible_score) * 100
        
        # Generate action and position size
        if buy_score > sell_score and signals['confidence'] >= 40:
            signals['action'] = 'BUY'
            
            # Calculate position size
            volatility_adj = 1 - min(analysis.get('volatility', 0.3), 0.5)
            base_position = self.max_position_size * (signals['confidence'] / 100)
            suggested_size = base_position * volatility_adj
            
            # Risk adjustment based on market conditions
            if market_conditions['risk_level'] == 'High':
                suggested_size *= 0.7
            
            signals['suggested_position_size_gbp'] = suggested_size
            signals['stop_loss'] = current_price * 0.95  # 5% stop loss
            signals['target_price'] = current_price * 1.1  # 10% profit target
            
        elif sell_score > buy_score and signals['confidence'] >= 40:
            signals['action'] = 'SELL'
        
        return signals

    def execute_trades(self):
        """Execute trades based on signals"""
        try:
            market_conditions = self.analyze_market_conditions()
            print(f"\nMarket Conditions: {market_conditions['trend']}, Risk Level: {market_conditions['risk_level']}")
            
            for symbol in self.portfolio:
                analysis = self.analyze_stock(symbol)
                if not analysis:
                    continue
                
                signals = self.generate_signals(analysis, market_conditions)
                current_price = analysis['current_price']
                
                # Handle BUY signals
                if signals['action'] == 'BUY' and self.cash_gbp >= signals['suggested_position_size_gbp']:
                    if symbol not in self.positions:  # Only buy if we don't already own it
                        quantity = int(signals['suggested_position_size_gbp'] * self.gbp_usd_rate / current_price)
                        cost = quantity * current_price / self.gbp_usd_rate
                        
                        if quantity > 0 and cost <= self.cash_gbp:
                            print(f"\n🟢 BUY Signal for {symbol}")
                            print(f"Price: ${current_price:.2f}")
                            print(f"Quantity: {quantity}")
                            print(f"Investment: £{cost:.2f}")
                            print(f"Confidence: {signals['confidence']:.1f}%")
                            print(f"Reasons: {', '.join(signals['reasons'])}")
                            
                            proceed = input("\nProceed with buy? (y/n): ").lower()
                            if proceed == 'y':
                                self.positions[symbol] = {
                                    'quantity': quantity,
                                    'entry_price': current_price,
                                    'stop_loss': signals['stop_loss'],
                                    'target_price': signals['target_price'],
                                    'entry_date': datetime.now()
                                }
                                self.cash_gbp -= cost
                                
                                self.trade_history.append({
                                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'quantity': quantity,
                                    'price': current_price,
                                    'cost_gbp': cost,
                                    'confidence': signals['confidence'],
                                    'reasons': ', '.join(signals['reasons'])
                                })
                
                # Handle SELL signals
                elif signals['action'] == 'SELL':
                    if symbol in self.positions:  # Only sell if we own the stock
                        position = self.positions[symbol]
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n🔴 SELL Signal for {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Quantity: {quantity}")
                        print(f"Proceeds: £{proceeds:.2f}")
                        print(f"Profit/Loss: £{profit:.2f}")
                        print(f"Confidence: {signals['confidence']:.1f}%")
                        print(f"Reasons: {', '.join(signals['reasons'])}")
                        
                        proceed = input("\nProceed with sell? (y/n): ").lower()
                        if proceed == 'y':
                            self.cash_gbp += proceeds
                            del self.positions[symbol]
                            
                            self.trade_history.append({
                                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'action': 'SELL',
                                'quantity': quantity,
                                'price': current_price,
                                'proceeds_gbp': proceeds,
                                'profit_gbp': profit,
                                'confidence': signals['confidence'],
                                'reasons': ', '.join(signals['reasons'])
                            })
                
                # Rest of the code (stop loss and take profit checks) remains the same
                
                elif symbol in self.positions:
                    position = self.positions[symbol]
                    if current_price <= position['stop_loss']:
                        # Stop loss hit
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        loss = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n⛔ STOP LOSS triggered for {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Quantity: {quantity}")
                        print(f"Proceeds: £{proceeds:.2f}")
                        print(f"Loss: £{loss:.2f}")
                        
                        proceed = input("\nProceed with stop loss? (y/n): ").lower()
                        if proceed == 'y':
                            self.cash_gbp += proceeds
                            del self.positions[symbol]
                            
                            self.trade_history.append({
                                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'action': 'STOP_LOSS',
                                'quantity': quantity,
                                'price': current_price,
                                'proceeds_gbp': proceeds,
                                'profit_gbp': loss,
                                'reasons': 'Stop loss triggered'
                            })
                    
                    elif current_price >= position['target_price']:
                        # Take profit
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n🎯 TAKE PROFIT triggered for {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Quantity: {quantity}")
                        print(f"Proceeds: £{proceeds:.2f}")
                        print(f"Profit: £{profit:.2f}")
                        
                        proceed = input("\nProceed with take profit? (y/n): ").lower()
                        if proceed == 'y':
                            self.cash_gbp += proceeds
                            del self.positions[symbol]
                            
                            self.trade_history.append({
                                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'action': 'TAKE_PROFIT',
                                'quantity': quantity,
                                'price': current_price,
                                'proceeds_gbp': proceeds,
                                'profit_gbp': profit,
                                'reasons': 'Target price reached'
                            })
        
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")

    def simulate_live_trading(self, interval_seconds: int = 300):
        """Run live trading simulation"""
        try:
            print("\nInitializing trading simulation...")
            print(f"Starting balance: £{self.cash_gbp:,.2f}")
            print("Training ML model...")
            
            first_run = True
            
            while True:
                if first_run:
                    print("\nStarting initial analysis...")
                    first_run = False
                else:
                    print(f"\nUpdating analysis... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Print portfolio status
                self._print_portfolio_status()
                
                # Execute trades
                self.execute_trades()
                
                # Wait for next update
                if not first_run:
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            self._print_final_results()
        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            self._print_final_results()

    def _print_portfolio_status(self):
        """Print current portfolio status"""
        try:
            print("\n=== Portfolio Status ===")
            
            # Calculate current portfolio value
            portfolio_value = self.cash_gbp
            positions_data = []
            
            for symbol, position in self.positions.items():
                analysis = self.analyze_stock(symbol)
                if analysis:
                    current_price = analysis['current_price']
                    position_value = position['quantity'] * current_price / self.gbp_usd_rate
                    profit_loss = position_value - (position['quantity'] * position['entry_price'] / self.gbp_usd_rate)
                    portfolio_value += position_value
                    
                    positions_data.append({
                        'Symbol': symbol,
                        'Quantity': position['quantity'],
                        'Entry Price': f"${position['entry_price']:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Value (£)': f"£{position_value:.2f}",
                        'P/L (£)': f"£{profit_loss:.2f}",
                        'P/L %': f"{(profit_loss / position_value * 100):.2f}%"
                    })
            
            if positions_data:
                print("\nOpen Positions:")
                print(tabulate(positions_data, headers='keys', tablefmt='pretty', showindex=False))
            else:
                print("\nNo open positions")
            
            print(f"\nCash Balance: £{self.cash_gbp:,.2f}")
            print(f"Portfolio Value: £{portfolio_value:,.2f}")
            print(f"Total P/L: £{(portfolio_value - self.total_investment_gbp):,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error printing portfolio status: {e}")

    def _print_final_results(self):
        """Print final trading results"""
        try:
            print("\n=== Trading Session Summary ===")
            
            # Print final portfolio status
            self._print_portfolio_status()
            
            if self.trade_history:
                print("\nTrade History:")
                trades_df = pd.DataFrame(self.trade_history)
                trades_df = trades_df.sort_values('date', ascending=False)
                
                # Calculate statistics
                total_trades = len(trades_df)
                
                # For BUY trades, we don't have profit yet
                profitable_trades = len([trade for trade in self.trade_history 
                                    if trade['action'] in ['SELL', 'TAKE_PROFIT'] 
                                    and trade.get('profit_gbp', 0) > 0])
                
                # Calculate total profit from completed trades only
                total_profit = sum(trade.get('profit_gbp', 0) for trade in self.trade_history 
                                if trade['action'] in ['SELL', 'TAKE_PROFIT'])
                
                print(f"\nTotal Trades: {total_trades}")
                print(f"Profitable Trades: {profitable_trades}")
                if total_trades > 0:
                    print(f"Win Rate: {(profitable_trades/total_trades*100):.2f}%")
                print(f"Total Profit/Loss: £{total_profit:.2f}")
                
                print("\nDetailed Trade History:")
                # Only include columns that exist in the DataFrame
                display_cols = ['date', 'symbol', 'action', 'quantity', 'price', 'cost_gbp', 'confidence', 'reasons']
                display_cols = [col for col in display_cols if col in trades_df.columns]
                print(tabulate(trades_df[display_cols], headers='keys', tablefmt='pretty', showindex=False))
            
            else:
                print("\nNo trades executed during this session")
                
        except Exception as e:
            self.logger.error(f"Error printing final results: {e}")

    def backtest_single_day(self, test_date: str):
        """Run automated backtesting for a specific date"""
        try:
            print(f"\nRunning backtesting simulation for {test_date}")
            print(f"Starting balance: £{self.cash_gbp:,.2f}")
            
            # Reset portfolio state
            self.positions = {}
            self.cash_gbp = self.total_investment_gbp
            self.trade_history = []
            
            # Get market conditions for that date
            market_conditions = self.analyze_market_conditions(test_date)
            print(f"\nMarket Conditions: {market_conditions['trend']}, Risk Level: {market_conditions['risk_level']}")
            
            # Session times (US market hours)
            sessions = {
                "morning": "09:30",
                "afternoon": "13:00",
                "evening": "16:00"
            }
            
            for session, time in sessions.items():
                print(f"\n=== {session.capitalize()} Session ({time}) ===")
                self._execute_backtest_trades(market_conditions, session, test_date)
                
                # Print session summary
                print(f"\n{session.capitalize()} Session Summary:")
                self._print_portfolio_status()
            
            # Print final results
            self._print_final_results()
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")

    def _execute_backtest_trades(self, market_conditions: Dict, session: str, test_date: str):
        """Execute automated trades for backtesting"""
        try:
            for symbol in self.portfolio:
                analysis = self.analyze_stock(symbol, test_date)
                if not analysis:
                    continue
                
                signals = self.generate_signals(analysis, market_conditions)
                current_price = analysis['current_price']
                
                print(f"\nAnalyzing {symbol}:")
                print(f"Price: ${current_price:.2f}")
                print(f"RSI: {analysis['rsi']:.1f}")
                print(f"Action: {signals['action']}")
                print(f"Confidence: {signals['confidence']:.1f}%")
                
                # BUY conditions - lowered threshold to 45%
                if signals['action'] == 'BUY' and self.cash_gbp >= signals['suggested_position_size_gbp']:
                    if symbol not in self.positions and signals['confidence'] >= 45:
                        quantity = int(signals['suggested_position_size_gbp'] * self.gbp_usd_rate / current_price)
                        cost = quantity * current_price / self.gbp_usd_rate
                        
                        if quantity > 0 and cost <= self.cash_gbp:
                            # Special conditions for oversold stocks
                            if analysis['rsi'] < 30 or len(signals['reasons']) >= 3:  # RSI oversold or multiple reasons
                                print(f"\n🟢 AUTO BUY: {symbol}")
                                print(f"Quantity: {quantity} shares @ ${current_price:.2f}")
                                print(f"Investment: £{cost:.2f}")
                                print(f"Reasons: {', '.join(signals['reasons'])}")
                                
                                self.positions[symbol] = {
                                    'quantity': quantity,
                                    'entry_price': current_price,
                                    'stop_loss': current_price * 0.97,  # Tighter 3% stop loss
                                    'target_price': current_price * 1.05,  # Smaller 5% target
                                    'entry_date': f"{session} session"
                                }
                                self.cash_gbp -= cost
                                
                                self.trade_history.append({
                                    'date': f"{test_date} {session}",
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'quantity': quantity,
                                    'price': current_price,
                                    'cost_gbp': cost,
                                    'confidence': signals['confidence'],
                                    'reasons': ', '.join(signals['reasons'])
                                })
                
                # SELL conditions - for existing positions
                elif symbol in self.positions:
                    position = self.positions[symbol]
                    proceeds = position['quantity'] * current_price / self.gbp_usd_rate
                    profit = proceeds - (position['quantity'] * position['entry_price'] / self.gbp_usd_rate)
                    
                    # Sell if: strong sell signal OR good profit OR stop loss
                    should_sell = (
                        (signals['action'] == 'SELL' and signals['confidence'] >= 45) or
                        (profit / proceeds >= 0.03) or  # 3% profit
                        (current_price <= position['stop_loss'])
                    )
                    
                    if should_sell:
                        sell_reason = 'Signal' if signals['action'] == 'SELL' else 'Profit Target' if profit > 0 else 'Stop Loss'
                        print(f"\n🔴 AUTO SELL ({sell_reason}): {symbol}")
                        print(f"Quantity: {position['quantity']} shares @ ${current_price:.2f}")
                        print(f"Profit/Loss: £{profit:.2f} ({(profit/proceeds)*100:.1f}%)")
                        
                        self.cash_gbp += proceeds
                        del self.positions[symbol]
                        
                        self.trade_history.append({
                            'date': f"{test_date} {session}",
                            'symbol': symbol,
                            'action': f'SELL_{sell_reason}',
                            'quantity': position['quantity'],
                            'price': current_price,
                            'proceeds_gbp': proceeds,
                            'profit_gbp': profit,
                            'profit_pct': (profit/proceeds)*100,
                            'holding_period': session
                        })
                        
        except Exception as e:
            self.logger.error(f"Error executing backtest trades: {e}")

    def analyze_market_conditions(self, date: str = None) -> Dict:
        """Analyze overall market conditions with option for historical data"""
        try:
            # Get SPY data as market proxy
            spy = yf.Ticker("SPY")
            if date:
                end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
                start_date = end_date - timedelta(days=30)  # 1 month of data
                hist = spy.history(start=start_date, end=end_date)
            else:
                hist = spy.history(period='1mo')
                
            if hist.empty:
                return {'trend': 'Unknown', 'volatility': 0, 'vix': None, 'risk_level': 'Unknown'}
            
            market_trend = 'Bullish' if hist['Close'].iloc[-1] > hist['Close'].mean() else 'Bearish'
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # Get VIX data
            vix = yf.Ticker("^VIX")
            if date:
                vix_hist = vix.history(start=start_date, end=end_date)
            else:
                vix_hist = vix.history(period='1d')
                
            vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else None
            
            return {
                'trend': market_trend,
                'volatility': volatility,
                'vix': vix_level,
                'risk_level': 'High' if (vix_level and vix_level > 25) else 'Moderate' if (vix_level and vix_level > 15) else 'Low'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {'trend': 'Unknown', 'volatility': 0, 'vix': None, 'risk_level': 'Unknown'}

    def backtest_period(self, start_date: str, end_date: str):
        """Run backtesting without resetting the portfolio daily."""
        try:
            print(f"\nRunning backtesting simulation from {start_date} to {end_date}")
            print(f"Starting balance: £{self.total_investment_gbp:,.2f}")

            # Initialize portfolio state *once* at the start
            self.positions = {}
            self.cash_gbp = self.total_investment_gbp
            self.trade_history = []

            # Create date range
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            dates = [
                (start + timedelta(days=x)).strftime('%Y-%m-%d') 
                for x in range((end - start).days + 1)
            ]

            # Track daily results
            daily_results = []

            for test_date in dates:
                print(f"\n{'='*50}")
                print(f"Trading Day: {test_date}")

                # Analyze market conditions for this day
                market_conditions = self.analyze_market_conditions(test_date)
                print(f"Market Conditions: {market_conditions['trend']}, Risk Level: {market_conditions['risk_level']}")

                # Execute a single trading pass for this day
                trades_before = len(self.trade_history)
                self._execute_backtest_trades_single(market_conditions, test_date)
                trades_after = len(self.trade_history)

                # Calculate how many trades happened today
                day_trades = trades_after - trades_before

                # Value the entire portfolio at day-end
                total_value = self._calculate_portfolio_valuation()
                daily_pnl = total_value - self.total_investment_gbp

                # Save daily result
                daily_results.append({
                    'date': test_date,
                    'trades': day_trades,
                    'profit_loss': daily_pnl,
                    # For simplicity, set day-level win_rate=0 or compute from trade_history if desired
                    'win_rate': 0.0,
                    'market_trend': market_conditions['trend'],
                    'risk_level': market_conditions['risk_level']
                })

            # Finally, print summary
            self._print_backtest_summary(daily_results)

        except Exception as e:
            self.logger.error(f"Error in period backtesting: {e}")


    def _execute_backtest_trades_single(self, market_conditions: Dict, test_date: str):
        """Single-session daily trades without morning/afternoon/evening loops."""
        for symbol in self.portfolio:
            analysis = self.analyze_stock(symbol, test_date)
            if not analysis:
                continue

            signals = self.generate_signals(analysis, market_conditions)
            current_price = analysis['current_price']

            print(f"\nAnalyzing {symbol}:")
            print(f"  Price=${current_price:.2f}, RSI={analysis['rsi']:.1f}, "
                f"Action={signals['action']}, Confidence={signals['confidence']:.1f}%")

            # ----- BUY logic -----
            if signals['action'] == 'BUY' and self.cash_gbp >= signals['suggested_position_size_gbp']:
                # Relax confidence threshold to 30% 
                if symbol not in self.positions and signals['confidence'] >= 30:
                    quantity = int(signals['suggested_position_size_gbp'] * self.gbp_usd_rate / current_price)
                    cost = quantity * current_price / self.gbp_usd_rate

                    if quantity > 0 and cost <= self.cash_gbp:
                        print(f"  🟢 BUY {symbol} @ ${current_price:.2f}")
                        self.positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': current_price,
                            'stop_loss': current_price * 0.95,   # 5% stop
                            'target_price': current_price * 1.10, # 10% target
                            'entry_date': test_date
                        }
                        self.cash_gbp -= cost

                        self.trade_history.append({
                            'date': test_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price,
                            'cost_gbp': cost,
                            'confidence': signals['confidence'],
                            'reasons': ', '.join(signals['reasons'])
                        })

            # ----- SELL logic -----
            elif symbol in self.positions:
                position = self.positions[symbol]
                proceeds = position['quantity'] * current_price / self.gbp_usd_rate
                profit = proceeds - (position['quantity'] * position['entry_price'] / self.gbp_usd_rate)

                # Sell if the signal says SELL, or we hit stop_loss/target_price
                if (signals['action'] == 'SELL') \
                or (current_price <= position['stop_loss']) \
                or (current_price >= position['target_price']):
                    print(f"  🔴 SELL {symbol} @ ${current_price:.2f}, P/L=£{profit:.2f}")
                    self.cash_gbp += proceeds
                    del self.positions[symbol]

                    self.trade_history.append({
                        'date': test_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'price': current_price,
                        'proceeds_gbp': proceeds,
                        'profit_gbp': profit
                    })


    def _calculate_portfolio_valuation(self) -> float:
        """
        Helper to get the total value of all open positions + cash.
        """
        total_value = self.cash_gbp
        for symbol, pos in self.positions.items():
            analysis = self.analyze_stock(symbol)
            if analysis:
                current_price = analysis['current_price']
                position_value = pos['quantity'] * current_price / self.gbp_usd_rate
                total_value += position_value
        return total_value



    def _print_backtest_summary(self, daily_results: List[Dict]):
        """Print summary of backtest period"""
        try:
            print("\n=== Backtest Period Summary ===")
            
            if not daily_results:
                print("No trading results to analyze")
                return
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame(daily_results)
            
            # Calculate overall statistics
            total_days = len(df)
            trading_days = len(df[df['trades'] > 0])
            total_trades = df['trades'].sum()
            total_pnl = df['profit_loss'].sum()
            profitable_days = len(df[df['profit_loss'] > 0])
            
            # Print summary statistics
            print(f"\nOverall Performance:")
            print(f"Total Days: {total_days}")
            print(f"Active Trading Days: {trading_days}")
            print(f"Total Trades: {total_trades}")
            print(f"Total P/L: £{total_pnl:,.2f}")
            if total_days > 0:
                print(f"Profitable Days: {profitable_days} ({profitable_days/total_days*100:.1f}%)")
                print(f"Average Daily P/L: £{(total_pnl/total_days):,.2f}")
            
            # Show daily results
            print("\nDaily Results:")
            print(tabulate(df.sort_values('date', ascending=False), 
                        headers='keys', 
                        tablefmt='pretty', 
                        floatfmt=".2f",
                        showindex=False))
            
        except Exception as e:
            self.logger.error(f"Error printing backtest summary: {e}")
# if __name__ == "__main__":
#     analyzer = PortfolioAnalyzer(total_investment_gbp=10000.0)
#     yesterday = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')
#     analyzer.backtest_single_day(yesterday)

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer(total_investment_gbp=10000.0)
    
    # Calculate date range (last 60 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    
    # Run backtest
    analyzer.backtest_period(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    