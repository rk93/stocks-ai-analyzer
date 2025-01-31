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

    def analyze_stock(self, symbol: str) -> Dict:
        """Analyze a single stock"""
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='6mo')
            
            if len(hist) < 50:  # Need enough historical data
                return {}
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
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
            
            # Get market conditions
            market_conditions = self.analyze_market_conditions()
            print(f"\nMarket Conditions: {market_conditions['trend']}, Risk Level: {market_conditions['risk_level']}")
            
            # Morning session (market open)
            print("\n=== Morning Session ===")
            self._execute_backtest_trades(market_conditions, session="morning")
            
            # Afternoon session (mid-day)
            print("\n=== Afternoon Session ===")
            self._execute_backtest_trades(market_conditions, session="afternoon")
            
            # Evening session (market close)
            print("\n=== Evening Session ===")
            self._execute_backtest_trades(market_conditions, session="evening")
            
            # Print final results
            self._print_final_results()
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")

    def _execute_backtest_trades(self, market_conditions: Dict, session: str):
        """Execute automated trades for backtesting"""
        try:
            for symbol in self.portfolio:
                analysis = self.analyze_stock(symbol)
                if not analysis:
                    continue
                
                signals = self.generate_signals(analysis, market_conditions)
                current_price = analysis['current_price']
                
                # Automated decision making
                if signals['action'] == 'BUY' and self.cash_gbp >= signals['suggested_position_size_gbp']:
                    if symbol not in self.positions and signals['confidence'] >= 60:  # Higher confidence threshold for automated trading
                        quantity = int(signals['suggested_position_size_gbp'] * self.gbp_usd_rate / current_price)
                        cost = quantity * current_price / self.gbp_usd_rate
                        
                        if quantity > 0 and cost <= self.cash_gbp:
                            print(f"\n🟢 AUTO BUY: {symbol}")
                            print(f"Price: ${current_price:.2f}")
                            print(f"Quantity: {quantity}")
                            print(f"Investment: £{cost:.2f}")
                            print(f"Confidence: {signals['confidence']:.1f}%")
                            
                            self.positions[symbol] = {
                                'quantity': quantity,
                                'entry_price': current_price,
                                'stop_loss': signals['stop_loss'],
                                'target_price': signals['target_price'],
                                'entry_date': f"{session} session"
                            }
                            self.cash_gbp -= cost
                            
                            self.trade_history.append({
                                'date': f"{session} session",
                                'symbol': symbol,
                                'action': 'BUY',
                                'quantity': quantity,
                                'price': current_price,
                                'cost_gbp': cost,
                                'confidence': signals['confidence'],
                                'reasons': ', '.join(signals['reasons'])
                            })
                
                elif signals['action'] == 'SELL' and symbol in self.positions:
                    position = self.positions[symbol]
                    if signals['confidence'] >= 45:  # Higher confidence threshold for automated trading
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n🔴 AUTO SELL: {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Quantity: {quantity}")
                        print(f"Proceeds: £{proceeds:.2f}")
                        print(f"Profit/Loss: £{profit:.2f}")
                        
                        self.cash_gbp += proceeds
                        del self.positions[symbol]
                        
                        self.trade_history.append({
                            'date': f"{session} session",
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': quantity,
                            'price': current_price,
                            'proceeds_gbp': proceeds,
                            'profit_gbp': profit,
                            'confidence': signals['confidence'],
                            'reasons': ', '.join(signals['reasons'])
                        })
                
                # Check stop losses and take profits
                elif symbol in self.positions:
                    position = self.positions[symbol]
                    if current_price <= position['stop_loss']:
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        loss = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n⛔ AUTO STOP LOSS: {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Loss: £{loss:.2f}")
                        
                        self.cash_gbp += proceeds
                        del self.positions[symbol]
                        
                        self.trade_history.append({
                            'date': f"{session} session",
                            'symbol': symbol,
                            'action': 'STOP_LOSS',
                            'quantity': quantity,
                            'price': current_price,
                            'proceeds_gbp': proceeds,
                            'profit_gbp': loss,
                            'reasons': 'Stop loss triggered'
                        })
                    
                    elif current_price >= position['target_price']:
                        quantity = position['quantity']
                        proceeds = quantity * current_price / self.gbp_usd_rate
                        profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
                        
                        print(f"\n🎯 AUTO TAKE PROFIT: {symbol}")
                        print(f"Price: ${current_price:.2f}")
                        print(f"Profit: £{profit:.2f}")
                        
                        self.cash_gbp += proceeds
                        del self.positions[symbol]
                        
                        self.trade_history.append({
                            'date': f"{session} session",
                            'symbol': symbol,
                            'action': 'TAKE_PROFIT',
                            'quantity': quantity,
                            'price': current_price,
                            'proceeds_gbp': proceeds,
                            'profit_gbp': profit,
                            'reasons': 'Target price reached'
                        })
                        
        except Exception as e:
            self.logger.error(f"Error executing backtest trades: {e}")


# if __name__ == "__main__":
#     analyzer = PortfolioAnalyzer(total_investment_gbp=10000.0)
#     analyzer.simulate_live_trading(interval_seconds=300)  # Update every 5 minutes

# if __name__ == "__main__":
#     analyzer = PortfolioAnalyzer(total_investment_gbp=10000.0)
#     yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#     analyzer.backtest_single_day(yesterday)

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer(total_investment_gbp=10000.0)
    analyzer.simulate_live_trading(interval_seconds=300)  # Update every 5 minutes

    # proceed = input("\nProceed with trade? (y/n): ").lower()
    # if proceed == 'y':
    #     self.positions[symbol] = {
    #         'quantity': quantity,
    #         'entry_price': current_price,
    #         'stop_loss': signals['stop_loss'],
    #         'target_price': signals['target_price'],
    #         'entry_date': datetime.now()
    #     }
    #     self.cash_gbp -= cost
        
    #     self.trade_history.append({
    #         'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #         'symbol': symbol,
    #         'action': 'BUY',
    #         'quantity': quantity,
    #         'price': current_price,
    #         'cost_gbp': cost,
    #         'confidence': signals['confidence'],
    #         'reasons': ', '.join(signals['reasons'])
    #     })

    # elif signals['action'] == 'SELL' and symbol in self.positions:
    #     position = self.positions[symbol]
    #     quantity = position['quantity']
    #     proceeds = quantity * current_price / self.gbp_usd_rate
    #     profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
    #     print('profit ->', profit)

    # # New code (add this):
    # elif signals['action'] == 'SELL':
    #     if symbol in self.positions:  # Only process sell signals for stocks we own
    #         position = self.positions[symbol]
    #         quantity = position['quantity']
    #         proceeds = quantity * current_price / self.gbp_usd_rate
    #         profit = proceeds - (quantity * position['entry_price'] / self.gbp_usd_rate)
            
    #         print(f"\n🔴 SELL Signal for {symbol}")
    #         print(f"Price: ${current_price:.2f}")
    #         print(f"Quantity: {quantity}")
    #         print(f"Proceeds: £{proceeds:.2f}")
    #         print(f"Profit/Loss: £{profit:.2f}")
    #         print(f"Confidence: {signals['confidence']:.1f}%")
    #         print(f"Reasons: {', '.join(signals['reasons'])}")
            
    #         proceed = input("\nProceed with sell? (y/n): ").lower()
    #         if proceed == 'y':
    #             self.cash_gbp += proceeds
    #             del self.positions[symbol]
                
    #             self.trade_history.append({
    #                 'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #                 'symbol': symbol,
    #                 'action': 'SELL',
    #                 'quantity': quantity,
    #                 'price': current_price,
    #                 'proceeds_gbp': proceeds,
    #                 'profit_gbp': profit,
    #                 'confidence': signals['confidence'],
    #                 'reasons': ', '.join(signals['reasons'])
    #             })