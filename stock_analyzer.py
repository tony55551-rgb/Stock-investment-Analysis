import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob
from prophet import Prophet
from datetime import datetime
import plotly.graph_objs as go

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Pro Analyzer", page_icon="🔮", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    
    .score-box { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .score-high { background: linear-gradient(135deg, #059669, #10b981); } 
    .score-med { background: linear-gradient(135deg, #d97706, #f59e0b); } 
    .score-low { background: linear-gradient(135deg, #dc2626, #ef4444); }
    
    .ai-card {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .forecast-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #334155;
    }

    .metric-card {
        background-color: white;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eee;
        border-left: 8px solid #ccc;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .metric-card strong { color: #000 !important; font-size: 1.1rem; }
    .metric-card small { color: #666 !important; font-size: 0.9rem; display: block; margin-top: 5px; font-style: italic;}
    
    .card-pass { border-left-color: #059669; background-color: #f0fdf4; }
    .card-fail { border-left-color: #dc2626; background-color: #fef2f2; }
    
    .price-tag { font-size: 1.5rem; font-weight: bold; color: #333; text-align: center; padding: 10px; background: #e0f2fe; border-radius: 10px; margin-bottom: 20px; }

    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f8f9fa; color: #666; text-align: center; padding: 10px; font-size: 0.8rem; z-index: 100; border-top: 1px solid #e5e7eb; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_symbol_from_name(query):
    query = query.strip()
    if query.isupper() and len(query) <= 6 and " " not in query: return query
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except: pass
    return query.upper()

def get_currency_symbol(currency_code):
    if currency_code == 'INR': return '₹'
    if currency_code == 'USD': return '$'
    if currency_code == 'EUR': return '€'
    if currency_code == 'GBP': return '£'
    return f"{currency_code} "

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.financials, stock.balance_sheet, stock.cashflow

# --- AI LAYER 1: NEWS SENTIMENT ---
def analyze_ai_sentiment(stock):
    try:
        news = stock.news
        if not news: return "Neutral / No News", "#9ca3af", "No recent news headlines found."
        score_total = 0
        count = 0
        for item in news[:7]:
            title = item.get('title', '')
            if title:
                analysis = TextBlob(title)
                score_total += analysis.sentiment.polarity
                count += 1
        if count == 0: return "Neutral", "#9ca3af", "Could not analyze news text."
        avg_score = score_total / count
        if avg_score > 0.1: return "Positive (Bullish) 🐂", "High", "News headlines are optimistic."
        elif avg_score < -0.1: return "Negative (Bearish) 🐻", "Low", "News headlines are negative."
        else: return "Neutral 😐", "Med", "News is mixed."
    except: return "AI Error", "Med", "Could not run AI analysis."

# --- AI LAYER 2: PRICE PREDICTION (PROPHET) ---
def predict_future_price(ticker):
    """
    Uses Facebook Prophet to predict stock price 5 years into the future.
    """
    try:
        # 1. Fetch historical data (last 5 years for training)
        df = yf.download(ticker, period="5y")
        df.reset_index(inplace=True)
        
        # 2. Prepare for Prophet (needs 'ds' and 'y' columns)
        # Handle MultiIndex if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        data = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        data['ds'] = data['ds'].dt.tz_localize(None) # Remove timezone for Prophet

        # 3. Train Model
        m = Prophet(daily_seasonality=True)
        m.fit(data)

        # 4. Create Future Dates (5 Years = 365 * 5 days)
        future = m.make_future_dataframe(periods=365*5)
        forecast = m.predict(future)

        # 5. Extract Key Data Points
        current_price = forecast['yhat'].iloc[-365*5] # Approx 'today' in forecast
        future_price = forecast['yhat'].iloc[-1]      # 5 years later
        
        roi = ((future_price - current_price) / current_price) * 100
        
        return forecast, roi, future_price
    except Exception as e:
        return None, 0, 0

def run_pro_analysis(symbol):
    results = []
    score = 0
    stock, info, financials, balance_sheet, cashflow = get_financial_data(symbol)
    
    company_name = info.get('longName', symbol)
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    currency = info.get('currency', 'USD')
    curr_sym = get_currency_symbol(currency)
    
    ai_verdict, ai_strength, ai_msg = analyze_ai_sentiment(stock)
    
    # [Insert all 8 Fundamental Checks here - same as before]
    # 1. REV GROWTH
    try:
        revs = financials.loc['Total Revenue'].iloc[::-1]
        if len(revs) > 1:
            cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / (len(revs) - 1)) - 1
            if cagr >= 0.10: 
                results.append({"step": "Rev Growth > 10%", "status": "PASS", "val": f"{cagr:.2%}", "english": "Sales growing fast (+10%/yr)."})
                score += 1
            else: results.append({"step": "Rev Growth > 10%", "status": "FAIL", "val": f"{cagr:.2%}", "english": "Sales growing slowly."})
        else: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "No Data"})
    except: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # 2. P/E
    pe = info.get('trailingPE')
    if pe and pe < 30: 
        results.append({"step": "P/E < 30", "status": "PASS", "val": f"{pe:.2f}", "english": "Fair price vs profit."}); score += 1
    else: results.append({"step": "P/E < 30", "status": "FAIL", "val": f"{pe if pe else 'N/A'}", "english": "Expensive price."})

    # 3. ROE
    roe = info.get('returnOnEquity')
    if roe and roe > 0.10: 
        results.append({"step": "ROE > 10%", "status": "PASS", "val": f"{roe:.2%}", "english": "High efficiency."}); score += 1
    else: results.append({"step": "ROE > 10%", "status": "FAIL", "val": f"{roe:.2%}" if roe else 'N/A', "english": "Low efficiency."})

    # 4. DEBT/EQUITY
    de = info.get('debtToEquity')
    if de is not None:
        ratio = de/100
        if ratio < 1.0: results.append({"step": "Debt/Eq < 1.0", "status": "PASS", "val": f"{ratio:.2f}", "english": "Safe debt levels."}); score += 1
        else: results.append({"step": "Debt/Eq < 1.0", "status": "FAIL", "val": f"{ratio:.2f}", "english": "High debt risk."})
    else: results.append({"step": "Debt/Eq", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # 5. FCF
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf and mcap:
        if (fcf/mcap) > 0.03: results.append({"step": "FCF Yield > 3%", "status": "PASS", "val": f"{fcf/mcap:.2%}", "english": "Cash machine!"}); score += 1
        else: results.append({"step": "FCF Yield > 3%", "status": "FAIL", "val": f"{fcf/mcap:.2%}", "english": "Low cash generation."})
    else: results.append({"step": "FCF Yield", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # 6. UPSIDE
    tgt = info.get('targetMeanPrice')
    if tgt and current_price:
        up = (tgt - current_price)/current_price
        if up > 0.10: results.append({"step": "Analyst Upside", "status": "PASS", "val": f"+{up:.1%}", "english": "Experts predict rise."}); score += 1
        else: results.append({"step": "Analyst Upside", "status": "FAIL", "val": f"{up:.1%}", "english": "Experts are cautious."})
    else: results.append({"step": "Analyst Upside", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # 7. PEG
    peg = info.get('pegRatio')
    if peg and peg < 2.0: results.append({"step": "PEG < 2", "status": "PASS", "val": f"{peg:.2f}", "english": "Undervalued growth."}); score += 1
    else: results.append({"step": "PEG < 2", "status": "FAIL", "val": f"{peg if peg else 'N/A'}", "english": "Overvalued growth."})

    # 8. CURRENT
    curr = info.get('currentRatio')
    if curr and curr > 1.5: results.append({"step": "Curr Ratio > 1.5", "status": "PASS", "val": f"{curr:.2f}", "english": "Good liquidity."}); score += 1
    else: results.append({"step": "Curr Ratio > 1.5", "status": "FAIL", "val": f"{curr if curr else 'N/A'}", "english": "Tight liquidity."})

    return score, results, company_name, current_price, curr_sym, ai_verdict, ai_msg

# --- MAIN APP ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔮 YnotAI Login")
        with st.form("login"):
            # These are the input fields you were looking for:
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            
            if st.form_submit_button("Access Terminal"):
                # --- YOUR CREDENTIALS ARE HERE ---
                if user == "ynot" and pw == "Str0ng@Pulse#884":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied. Incorrect username or password.")

def footer():
    st.markdown('<div class="footer">© 2026 ynotAIbundle | Pro + Predictive AI</div>', unsafe_allow_html=True)

def main_app():
    with st.sidebar:
        st.write("User: **ynot_admin**")
        if st.button("Logout"): st.session_state.authenticated = False; st.rerun()
        st.info("Features:\n1. 8-Point Check\n2. Sentiment AI\n3. 5-Year Forecast")

    st.markdown('<div class="main-header">YnotAI Ultimate</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Search Ticker/Company", placeholder="e.g. Nvidia, Reliance").strip()
    with col2: st.write(""); st.write(""); btn = st.button("Analyze 🚀", type="primary", use_container_width=True)

    if btn and query:
        with st.spinner(f"Running AI Models for '{query}'..."):
            symbol = get_symbol_from_name(query)
            try:
                # 1. Fundamental Analysis
                score, trace, name, price, sym, ai_verdict, ai_msg = run_pro_analysis(symbol)
                
                st.markdown(f"### 🏢 {name} ({symbol})")
                st.markdown(f'<div class="price-tag">Price: {sym}{price:,.2f}</div>', unsafe_allow_html=True)

                # 2. AI Cards (Sentiment)
                st.markdown(f"""
                    <div class="ai-card">
                        <h3>🧠 News Sentiment: {ai_verdict}</h3>
                        <p>{ai_msg}</p>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Scorecard
                s_class = "score-high" if score >= 7 else "score-med" if score >= 5 else "score-low"
                verdict = "STRONG BUY" if score >= 7 else "HOLD" if score >= 5 else "AVOID"
                st.markdown(f'<div class="score-box {s_class}"><h1>{score}/8</h1><h3>{verdict}</h3></div>', unsafe_allow_html=True)

                # 4. Fundamental Breakdown
                c1, c2 = st.columns(2)
                for i, item in enumerate(trace):
                    css = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "⚠️"
                    html = f'<div class="metric-card {css}"><strong>{icon} {item["step"]}</strong><br><small>{item["english"]}</small><div style="text-align:right"><strong>{item["val"]}</strong></div></div>'
                    if i%2==0: c1.markdown(html, unsafe_allow_html=True)
                    else: c2.markdown(html, unsafe_allow_html=True)
                
                # 5. PREDICTIVE LAYER (The New Part)
                st.markdown("---")
                st.subheader(f"🔮 5-Year AI Price Prediction (Prophet Model)")
                with st.spinner("Calculating future trajectory..."):
                    forecast, roi, fut_price = predict_future_price(symbol)
                    
                    if forecast is not None:
                        # Display ROI Box
                        roi_color = "#10b981" if roi > 0 else "#ef4444"
                        st.markdown(f"""
                            <div class="forecast-box">
                                <h2>Projected Price (2029): {sym}{fut_price:,.2f}</h2>
                                <h3 style="color:{roi_color}">Expected ROI: {roi:+.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Plot interactive chart
                        fig = go.Figure()
                        # Historical Data
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Trend Line', line=dict(color='#3b82f6')))
                        # Confidence Interval (Upper/Lower bounds)
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.2)', name='Confidence Range'))
                        
                        fig.update_layout(title="AI Projected Growth Path", xaxis_title="Year", yaxis_title="Price", template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("Note: This projection uses the 'Facebook Prophet' model based on historical volatility and seasonal trends. Past performance does not guarantee future results.")
                    else:
                        st.warning("Not enough historical data to generate a confident 5-year forecast.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    footer()

if st.session_state.authenticated: main_app()
else: login_screen(); footer()
