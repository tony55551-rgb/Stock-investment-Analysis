import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob
from prophet import Prophet
from datetime import datetime
import plotly.graph_objs as go

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Forensic", page_icon="🕵️‍♂️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* GLOBAL FONTS & HEADERS */
    .main-header { font-size: 3rem; color: #4F46E5; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 1.2rem; color: #6b7280; text-align: center; margin-bottom: 30px; }
    
    /* SCORECARD BOX */
    .score-box { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; color: white; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .score-high { background: linear-gradient(to right, #059669, #10b981); } 
    .score-med { background: linear-gradient(to right, #d97706, #f59e0b); } 
    .score-low { background: linear-gradient(to right, #dc2626, #ef4444); }
    
    /* AI & FORECAST CARDS */
    .ai-card {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
        color: white !important;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .forecast-box {
        background: #1e293b;
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #334155;
    }

    /* METRIC CARDS - FIXED TEXT VISIBILITY */
    .metric-card {
        background-color: #ffffff !important; /* Force White Background */
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        border-left: 10px solid #ccc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        height: 100%; /* Uniform height */
    }
    
    /* FORCE ALL TEXT INSIDE CARDS TO BE BLACK/DARK */
    .metric-card div, .metric-card strong, .metric-card span, .metric-card small {
        color: #1f2937 !important; /* Dark Gray/Black */
        font-family: sans-serif;
    }
    
    .card-pass { border-left-color: #10b981; } /* Green Border */
    .card-fail { border-left-color: #ef4444; } /* Red Border */
    
    .price-tag { 
        font-size: 2rem; 
        font-weight: bold; 
        color: #111827; 
        background: #f3f4f6; 
        padding: 15px; 
        border-radius: 12px; 
        text-align: center; 
        margin-bottom: 20px; 
        border: 1px solid #d1d5db;
    }

    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #1f2937; color: #9ca3af; text-align: center; padding: 10px; font-size: 0.8rem; z-index: 100; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_symbol_from_name(query):
    query = query.strip()
    if (query.isupper() and len(query) <= 12) or ".NS" in query.upper() or ".BO" in query.upper():
        return query.upper()
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            quotes = data['quotes']
            for q in quotes:
                sym = q.get('symbol', '')
                if sym.endswith('.NS') or sym.endswith('.BO'): return sym
            return quotes[0]['symbol']
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

# --- AI LAYERS ---
def analyze_ai_sentiment(stock):
    try:
        news = stock.news
        if not news: return "Neutral / No News", "#9ca3af", "No recent news found."
        score_total = 0
        count = 0
        for item in news[:7]:
            title = item.get('title', '')
            if title:
                analysis = TextBlob(title)
                score_total += analysis.sentiment.polarity
                count += 1
        if count == 0: return "Neutral", "#9ca3af", "Could not analyze news."
        avg_score = score_total / count
        if avg_score > 0.05: return "Positive (Bullish) 🐂", "High", "News headlines are optimistic."
        elif avg_score < -0.05: return "Negative (Bearish) 🐻", "Low", "News headlines are negative."
        else: return "Neutral 😐", "Med", "News is mixed."
    except: return "AI Error", "Med", "AI unavailable."

def predict_future_price(ticker):
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty or len(df) < 100: return None, 0, 0
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        data = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        data['ds'] = data['ds'].dt.tz_localize(None)
        m = Prophet(daily_seasonality=True)
        m.fit(data)
        future = m.make_future_dataframe(periods=365*5)
        forecast = m.predict(future)
        current_price = forecast['yhat'].iloc[-365*5] 
        future_price = forecast['yhat'].iloc[-1]      
        roi = ((future_price - current_price) / current_price) * 100
        return forecast, roi, future_price
    except: return None, 0, 0

def run_pro_analysis(symbol):
    results = []
    score = 0
    stock, info, financials, balance_sheet, cashflow = get_financial_data(symbol)
    
    company_name = info.get('longName', symbol)
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    currency = info.get('currency', 'USD')
    curr_sym = get_currency_symbol(currency)
    
    ai_verdict, ai_strength, ai_msg = analyze_ai_sentiment(stock)
    
    # --- 15-POINT CHECKLIST ---
    
    # 1-8 (ORIGINAL METRICS)
    # Rev Growth
    try:
        revs = financials.loc['Total Revenue'].iloc[::-1]
        if len(revs) > 1:
            cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / (len(revs) - 1)) - 1
            if cagr >= 0.10: results.append({"step": "Rev Growth > 10%", "status": "PASS", "val": f"{cagr:.2%}", "english": "Sales growing fast."}); score += 1
            else: results.append({"step": "Rev Growth > 10%", "status": "FAIL", "val": f"{cagr:.2%}", "english": "Sales slow."})
        else: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "No Data"})
    except: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # P/E
    pe = info.get('trailingPE')
    if pe and pe < 30: results.append({"step": "P/E < 30", "status": "PASS", "val": f"{pe:.2f}", "english": "Fairly priced."}); score += 1
    else: results.append({"step": "P/E < 30", "status": "FAIL", "val": f"{pe if pe else 'N/A'}", "english": "Expensive."})

    # ROE
    roe = info.get('returnOnEquity')
    if roe and roe > 0.10: results.append({"step": "ROE > 10%", "status": "PASS", "val": f"{roe:.2f}", "english": "High efficiency."}); score += 1
    else: results.append({"step": "ROE > 10%", "status": "FAIL", "val": f"{roe if roe else 'N/A'}", "english": "Low efficiency."})

    # Debt/Equity
    de = info.get('debtToEquity')
    if de is not None:
        ratio = de/100
        if ratio < 1.0: results.append({"step": "Debt/Eq < 1.0", "status": "PASS", "val": f"{ratio:.2f}", "english": "Safe debt."}); score += 1
        else: results.append({"step": "Debt/Eq < 1.0", "status": "FAIL", "val": f"{ratio:.2f}", "english": "High debt."})
    else: results.append({"step": "Debt/Eq", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # FCF
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf is None and not cashflow.empty: # Plan B
        try:
             if 'Free Cash Flow' in cashflow.index: fcf = cashflow.loc['Free Cash Flow'].iloc[0]
             elif 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                 fcf = cashflow.loc['Operating Cash Flow'].iloc[0] + cashflow.loc['Capital Expenditure'].iloc[0]
        except: pass
    if fcf and mcap:
        if (fcf/mcap) > 0.03: results.append({"step": "FCF Yield > 3%", "status": "PASS", "val": f"{fcf/mcap:.2%}", "english": "Real cash!"}); score += 1
        else: results.append({"step": "FCF Yield > 3%", "status": "FAIL", "val": f"{fcf/mcap:.2%}", "english": "Low cash generation."})
    else: results.append({"step": "FCF Yield", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # Upside
    tgt = info.get('targetMeanPrice')
    if tgt and current_price:
        up = (tgt - current_price)/current_price
        if up > 0.10: results.append({"step": "Analyst Upside", "status": "PASS", "val": f"+{up:.1%}", "english": "Analysts bullish."}); score += 1
        else: results.append({"step": "Analyst Upside", "status": "FAIL", "val": f"{up:.1%}", "english": "Analysts cautious."})
    else: results.append({"step": "Analyst Upside", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # PEG
    peg = info.get('pegRatio')
    if peg and peg < 2.0: results.append({"step": "PEG < 2", "status": "PASS", "val": f"{peg:.2f}", "english": "Undervalued growth."}); score += 1
    else: results.append({"step": "PEG < 2", "status": "FAIL", "val": f"{peg if peg else 'N/A'}", "english": "Overvalued growth."})

    # Current Ratio
    curr = info.get('currentRatio')
    if curr and curr > 1.5: results.append({"step": "Curr Ratio > 1.5", "status": "PASS", "val": f"{curr:.2f}", "english": "Safe liquidity."}); score += 1
    else: results.append({"step": "Curr Ratio > 1.5", "status": "FAIL", "val": f"{curr if curr else 'N/A'}", "english": "Tight liquidity."})

    # 9-12 (WALL ST METRICS)
    # EV/EBITDA
    ev = info.get('enterpriseValue')
    ebitda = info.get('ebitda')
    if ev and ebitda:
        ev_ebitda = ev / ebitda
        if ev_ebitda < 20: results.append({"step": "EV/EBITDA < 20", "status": "PASS", "val": f"{ev_ebitda:.2f}", "english": "Good value."}); score += 1
        else: results.append({"step": "EV/EBITDA < 20", "status": "FAIL", "val": f"{ev_ebitda:.2f}", "english": "Overvalued."})
    else: results.append({"step": "EV/EBITDA", "status": "FAIL", "val": "N/A", "english": "No Data"})

    # ROA
    roa = info.get('returnOnAssets')
    if roa and roa > 0.05: results.append({"step": "ROA > 5%", "status": "PASS", "val": f"{roa:.2%}", "english": "Efficient Assets."}); score += 1
    else: results.append({"step": "ROA > 5%", "status": "FAIL", "val": f"{roa if roa else 'N/A'}", "english": "Inefficient Assets."})

    # Gross Margin
    gm = info.get('grossMargins')
    if gm and gm > 0.40: results.append({"step": "Gross Mrg > 40%", "status": "PASS", "val": f"{gm:.2%}", "english": "High pricing power."}); score += 1
    else: results.append({"step": "Gross Mrg > 40%", "status": "FAIL", "val": f"{gm if gm else 'N/A'}", "english": "Low margins."})

    # Institutional Hold
    inst = info.get('heldPercentInstitutions')
    if inst and inst > 0.30: results.append({"step": "Inst. Hold > 30%", "status": "PASS", "val": f"{inst:.2%}", "english": "Big banks buying."}); score += 1
    else: results.append({"step": "Inst. Hold > 30%", "status": "FAIL", "val": f"{inst if inst else 'N/A'}", "english": "Retail owned."})

    # --- 13-15 (FORENSIC METRICS for TEJAS CASE) ---
    
    # 13. DSO (Days Sales Outstanding) - Checks Client Payment Delays
    # Formula: (Receivables / Revenue) * 365
    try:
        if not balance_sheet.empty and not financials.empty:
            # Try to get Net Receivables
            receivables = None
            if 'Net Receivables' in balance_sheet.index: receivables = balance_sheet.loc['Net Receivables'].iloc[0]
            elif 'Accounts Receivable' in balance_sheet.index: receivables = balance_sheet.loc['Accounts Receivable'].iloc[0]
            
            revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else None
            
            if receivables and revenue:
                dso = (receivables / revenue) * 365
                if dso < 90: results.append({"step": "DSO < 90 Days", "status": "PASS", "val": f"{int(dso)} days", "english": "Clients pay on time."}); score += 1
                else: results.append({"step": "DSO < 90 Days", "status": "FAIL", "val": f"{int(dso)} days", "english": "Clients delaying payment!"})
            else: results.append({"step": "DSO (Client Risk)", "status": "FAIL", "val": "N/A", "english": "Data Unavailable"})
        else: results.append({"step": "DSO (Client Risk)", "status": "FAIL", "val": "N/A", "english": "Data Unavailable"})
    except: results.append({"step": "DSO (Client Risk)", "status": "FAIL", "val": "Error", "english": "Calc Error"})

    # 14. Inventory Turnover Days - Checks Stuck Stock
    # Formula: (Inventory / Cost of Goods Sold) * 365
    try:
        if not balance_sheet.empty and not financials.empty:
            inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else None
            cogs = financials.loc['Cost Of Revenue'].iloc[0] if 'Cost Of Revenue' in financials.index else None
            
            if inventory and cogs:
                inv_days = (inventory / cogs) * 365
                if inv_days < 150: results.append({"step": "Inv Days < 150", "status": "PASS", "val": f"{int(inv_days)} days", "english": "Stock moves fast."}); score += 1
                else: results.append({"step": "Inv Days < 150", "status": "FAIL", "val": f"{int(inv_days)} days", "english": "Stock stuck in warehouse!"})
            else: results.append({"step": "Inv. Turnover", "status": "FAIL", "val": "N/A", "english": "Data Unavailable"})
        else: results.append({"step": "Inv. Turnover", "status": "FAIL", "val": "N/A", "english": "Data Unavailable"})
    except: results.append({"step": "Inv. Turnover", "status": "FAIL", "val": "Error", "english": "Calc Error"})

    # 15. Operating Cash Flow - Checks Real Money vs Paper Profit
    try:
        if not cashflow.empty and 'Operating Cash Flow' in cashflow.index:
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
            if ocf > 0: results.append({"step": "Op Cash Flow > 0", "status": "PASS", "val": "Positive", "english": "Core biz generating cash."}); score += 1
            else: results.append({"step": "Op Cash Flow > 0", "status": "FAIL", "val": "Negative", "english": "Burning cash!"})
        else: results.append({"step": "Op Cash Flow", "status": "FAIL", "val": "N/A", "english": "Data Unavailable"})
    except: results.append({"step": "Op Cash Flow", "status": "FAIL", "val": "Error", "english": "Calc Error"})

    return score, results, company_name, current_price, curr_sym, ai_verdict, ai_msg

# --- MAIN APP ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🕵️‍♂️ YnotAI Stock Analyzer")
        with st.form("login"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Access Terminal"):
                # CREDENTIALS
                if user == "ynot" and pw == "Str0ng@Pulse#884":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Access Denied.")

def footer():
    st.markdown('<div class="footer">© 2026 ynotAIbundle | Forensic Edition</div>', unsafe_allow_html=True)

def main_app():
    with st.sidebar:
        st.write("User: **ynot_admin**")
        if st.button("Logout"): st.session_state.authenticated = False; st.rerun()
        st.info("Features:\n1. 15-Point Check\n2. Forensic Ratios\n3. AI Prediction")

    st.markdown('<div class="main-header">YnotAI Stock Analyzer Pro</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Search Ticker/Company", placeholder="e.g. Tejas Networks, Reliance").strip()
    with col2: st.write(""); st.write(""); btn = st.button("Analyze 🚀", type="primary", use_container_width=True)

    if btn and query:
        with st.spinner(f"Running Forensic Analysis for '{query}'..."):
            symbol = get_symbol_from_name(query)
            try:
                score, trace, name, price, sym, ai_verdict, ai_msg = run_pro_analysis(symbol)
                
                st.markdown(f"### 🏢 {name} ({symbol})")
                st.markdown(f'<div class="price-tag">Price: {sym}{price:,.2f}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="ai-card">
                        <h3>🧠 News Sentiment: {ai_verdict}</h3>
                        <p>{ai_msg}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Adjusted Score thresholds for 15 parameters
                if score >= 12: s_class = "score-high"; verdict = "STRONG BUY (Clean)"
                elif score >= 8: s_class = "score-med"; verdict = "CAUTIOUS / HOLD"
                else: s_class = "score-low"; verdict = "HIGH RISK / AVOID"

                st.markdown(f'<div class="score-box {s_class}"><h1>{score}/15</h1><h3>{verdict}</h3></div>', unsafe_allow_html=True)

                # 3 columns for 15 cards
                c1, c2, c3 = st.columns(3)
                for i, item in enumerate(trace):
                    css = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "❌"
                    html = f"""
                    <div class="metric-card {css}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <strong>{icon} {item['step']}</strong><br>
                                <small>{item['english']}</small>
                            </div>
                            <div style="text-align:right;">
                                <strong>{item['val']}</strong>
                            </div>
                        </div>
                    </div>
                    """
                    if i % 3 == 0: c1.markdown(html, unsafe_allow_html=True)
                    elif i % 3 == 1: c2.markdown(html, unsafe_allow_html=True)
                    else: c3.markdown(html, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader(f"🔮 5-Year AI Price Prediction")
                with st.spinner("Calculating trajectory..."):
                    forecast, roi, fut_price = predict_future_price(symbol)
                    
                    if forecast is not None:
                        roi_color = "#10b981" if roi > 0 else "#ef4444"
                        st.markdown(f"""
                            <div class="forecast-box">
                                <h2>Projected Price (2031): {sym}{fut_price:,.2f}</h2>
                                <h3 style="color:{roi_color}">Expected ROI: {roi:+.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Trend', line=dict(color='#3b82f6')))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(59, 130, 246, 0.2)', name='Range'))
                        fig.update_layout(title="AI Growth Path", xaxis_title="Year", yaxis_title="Price", template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data for 5-year forecast.")

            except Exception as e:
                st.error(f"Could not analyze '{symbol}'. Try the exact ticker.")
                st.write(e)

    footer()

if st.session_state.authenticated: main_app()
else: login_screen(); footer()
