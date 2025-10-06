from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

class AdvancedCandleAnalyzer:
    def __init__(self):
        self.available_indicators = ['RSI', 'MACD', 'BBANDS', 'STOCH', 'EMA', 'SMA']
    
    def fetch_data(self, symbol, period="6mo"):
        try:
            logger.info(f"Загружаем данные для {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"Нет данных для {symbol}")
                return None
                
            return data
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {str(e)}")
            return None
    
    def calculate_simple_indicators(self, df):
        try:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            return df
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {str(e)}")
            return df
    
    def detect_patterns(self, df):
        patterns = []
        
        try:
            for i in range(2, len(df)):
                if i >= len(df): continue
                
                current = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                
                # Doji
                body_size = abs(current['Open'] - current['Close'])
                total_range = current['High'] - current['Low']
                
                if total_range > 0 and body_size / total_range < 0.1:
                    patterns.append({
                        'date': current.name.strftime('%Y-%m-%d'),
                        'pattern': 'Doji',
                        'signal': 'Neutral'
                    })
                
                # Bullish Engulfing
                if (prev['Close'] < prev['Open'] and 
                    current['Open'] < prev['Close'] and 
                    current['Close'] > prev['Open']):
                    patterns.append({
                        'date': current.name.strftime('%Y-%m-%d'),
                        'pattern': 'Bullish Engulfing', 
                        'signal': 'Bullish'
                    })
                    
                # Bearish Engulfing
                if (prev['Close'] > prev['Open'] and 
                    current['Open'] > prev['Close'] and 
                    current['Close'] < prev['Open']):
                    patterns.append({
                        'date': current.name.strftime('%Y-%m-%d'),
                        'pattern': 'Bearish Engulfing',
                        'signal': 'Bearish'
                    })
                
                # Hammer
                if (current['Close'] > current['Open'] and
                    (current['Close'] - current['Low']) > 2 * (current['High'] - current['Close']) and
                    (current['Open'] - current['Low']) > 2 * (current['High'] - current['Open'])):
                    patterns.append({
                        'date': current.name.strftime('%Y-%m-%d'),
                        'pattern': 'Hammer',
                        'signal': 'Bullish'
                    })
            
            return patterns[-10:]
        except Exception as e:
            logger.error(f"Ошибка обнаружения паттернов: {str(e)}")
            return []

analyzer = AdvancedCandleAnalyzer()

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Candle Analyzer API работает",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['GET'])
def analyze_stock():
    try:
        symbol = request.args.get('symbol', 'AAPL')
        period = request.args.get('period', '6mo')
        
        logger.info(f"Анализируем {symbol} за период {period}")
        
        stock_data = analyzer.fetch_data(symbol, period)
        if stock_data is None or stock_data.empty:
            return jsonify({
                'error': f'Не удалось загрузить данные для {symbol}. Проверьте правильность тикера.',
                'demo_available': True
            }), 400
        
        stock_data = analyzer.calculate_simple_indicators(stock_data)
        patterns = analyzer.detect_patterns(stock_data)
        
        # Подготовка данных для графика
        chart_data = []
        for index, row in stock_data.tail(100).iterrows():
            chart_data.append({
                'x': index.strftime('%Y-%m-%d'),
                'o': float(row['Open']),
                'h': float(row['High']),
                'l': float(row['Low']), 
                'c': float(row['Close']),
                'volume': float(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
            })
        
        current_price = float(stock_data['Close'].iloc[-1])
        current_rsi = float(stock_data['RSI'].iloc[-1]) if not pd.isna(stock_data['RSI'].iloc[-1]) else 50
        
        # Генерация сигналов
        signals = generate_signals(current_rsi, stock_data)
        
        return jsonify({
            'symbol': symbol,
            'period': period,
            'data': chart_data,
            'patterns': patterns,
            'signals': signals,
            'current_price': current_price,
            'rsi': current_rsi,
            'analysis': generate_analysis(current_rsi, signals, patterns),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {str(e)}")
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500

def generate_signals(rsi, df):
    signals = []
    
    if rsi > 70:
        signals.append({
            'type': 'SELL',
            'indicator': 'RSI',
            'message': 'Перекупленность - RSI выше 70',
            'strength': 'HIGH'
        })
    elif rsi < 30:
        signals.append({
            'type': 'BUY', 
            'indicator': 'RSI',
            'message': 'Перепроданность - RSI ниже 30',
            'strength': 'HIGH'
        })
    
    # MACD сигналы
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if len(df) >= 2:
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_Signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_Signal'].iloc[-2]
            
            if not pd.isna(current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal and prev_macd <= prev_signal:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'MACD',
                        'message': 'Бычье пересечение MACD',
                        'strength': 'MEDIUM'
                    })
                elif current_macd < current_signal and prev_macd >= prev_signal:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'MACD', 
                        'message': 'Медвежье пересечение MACD',
                        'strength': 'MEDIUM'
                    })
    
    return signals

def generate_analysis(rsi, signals, patterns):
    if not signals:
        return "Рынок в консолидации. Рекомендуется наблюдение."
    
    buy_signals = [s for s in signals if s['type'] == 'BUY']
    sell_signals = [s for s in signals if s['type'] == 'SELL']
    
    if buy_signals and sell_signals:
        return "Смешанные сигналы. Рекомендуется осторожность."
    elif buy_signals:
        return "Преобладают бычьи сигналы. Возможен рост."
    elif sell_signals:
        return "Преобладают медвежьи сигналы. Возможна коррекция."
    else:
        return "Сигналы отсутствуют. Рынок в нейтральной зоне."

@app.route('/api/symbols')
def get_popular_symbols():
    symbols = {
        'Акции США': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'NFLX'],
        'Криптовалюты': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD'],
        'Индексы': ['^GSPC', '^DJI', '^IXIC'],
        'Товары': ['GC=F', 'CL=F'],
        'Форекс': ['EURUSD=X', 'GBPUSD=X']
    }
    return jsonify(symbols)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)