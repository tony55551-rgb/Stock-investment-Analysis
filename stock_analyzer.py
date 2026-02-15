import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob
from prophet import Prophet
from datetime import datetime
import plotly.graph_objs as go
import os

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Ultimate Dashboard", page_icon="🕵️‍♂️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #4F46E5; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .score-box { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; color: white; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .score-high { background: linear-gradient(to right, #059669, #10b981); } 
    .score-med { background: linear-gradient(to right, #d97706, #f59e0b); } 
    .score-low { background: linear-gradient(to right, #dc2626, #ef4444); }
    .profile-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e5e7eb; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); color: #1f2937 !important; }
    .profile-card h4 { color: #4F46E5 !important; margin-bottom: 10px; }
    .ceo-tag { font-weight: bold; color: #111827; background: #e0f2fe; padding: 5px 12px; border-radius: 20px; display: inline-block; margin-top: 10px; }
    .ai-card { background: linear-gradient(to right, #6366f1, #8b5cf6); color: white !important; padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center; }
    .forecast-box { background: #1e293b; color: white; padding: 25px; border-radius: 15px; margin-top: 20px; text-align: center; border: 1px solid #334155; }
    .metric-card { background-color: #ffffff !important; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb; border-left: 10px solid #ccc; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 15px; height: 100%; }
    .metric-card div, .metric-card strong, .metric-card span, .metric-card small { color: #1f2937 !important; font-family: sans-serif; }
    .card-pass { border-left-color: #10b981; } 
    .card-fail { border-left-color: #ef4444; } 
    .price-tag { font-size: 2rem; font-weight: bold; color: #111827; background: #f3f4f6; padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 20px; border: 1px solid #d1d5db; }
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
            for q in data['quotes']:
                sym = q.get('symbol', '')
                if sym.endswith('.NS') or sym.endswith('.BO'): return sym
            return data['quotes'][0]['symbol']
    except: pass
    return query.upper()

def get_currency_symbol(currency_code):
    symbols = {'INR': '₹', 'USD': '$', 'EUR': '€', 'GBP': '£'}
    return symbols.get(currency_code, f"{currency_code} ")

def analyze_ai_sentiment(stock):
    try:
        news = stock.news
        if not news: return "Neutral", "#9ca3af", "No recent news found."
        score_total = 0
        count = 0
        for item in news[:7]:
            title = item.get('title', '')
            if title:
                analysis = TextBlob(title)
                total_polarity = analysis.sentiment.polarity
                score_total += total_polarity
                count += 1
        if count == 0: return "Neutral", "#9ca3af", "Could not analyze news."
        avg_score = score_total / count
        if avg_score > 0.05: return "Positive (Bullish) 🐂", "High", "Optimistic headlines."
        elif avg_score < -0.05: return "Negative (Bearish) 🐻", "Low", "Negative headlines."
        else: return "Neutral 😐", "Med", "Mixed news."
    except: return "AI Error", "Med", "Sentiment analysis failed."

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
        current_p = forecast['yhat'].iloc[-365*5] 
        future_p = forecast['yhat'].iloc[-1]      
        roi = ((future_p - current_p) / current_p) * 100
        return forecast, roi, future_p
    except: return None, 0, 0

def run_full_intelligence(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    results = []
    score = 0
    
    name = info.get('longName', symbol)
    summary = info.get('longBusinessSummary', "Description unavailable.")
    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    curr_sym = get_currency_symbol(info.get('currency', 'USD'))
    
    ceo = "N/A"
    officers = info.get('companyOfficers', [])
    for off in officers:
        if 'CEO' in off.get('title', '') or 'Chief Executive' in off.get('title', ''):
            ceo = off.get('name', 'N/A')
            break
    
    # 1. Revenue Growth
    try:
        revs = financials.loc['Total Revenue'].iloc[::-1]
        cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / (len(revs) - 1)) - 1
        if cagr >= 0.10:
            results.append({"step": "Rev Growth > 10%", "status": "PASS", "val": f"{cagr:.2%}", "eng": "Sales growing fast."})
            score += 1
        else:
            results.append({"step": "Rev Growth > 10%", "status": "FAIL", "val": f"{cagr:.2%}", "eng": "Sales slowing."})
    except:
        results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "eng": "No Data"})

    # 2. P/E Ratio
    pe = info.get('trailingPE')
    if pe and pe < 30:
        results.append({"step": "P/E < 30", "status": "PASS", "val": f"{pe:.2f}", "eng": "Fairly priced."})
        score += 1
    else:
        results.append({"step": "P/E < 30", "status": "FAIL", "val": f"{pe if pe else 'N/A'}", "eng": "Expensive."})

    # 3. ROE
    roe = info.get('returnOnEquity')
    if roe and roe > 0.10:
        results.append({"step": "ROE > 10%", "status": "PASS", "val": f"{roe:.2%}", "eng": "Efficient management."})
        score += 1
    else:
        results.append({"step": "ROE > 10%", "status": "FAIL", "val": f"{roe if roe else 'N/A'}", "eng": "Low efficiency."})

    # 4. Debt/Equity
    de = info.get('debtToEquity')
    if de is not None and (de/100) < 1.0:
        results.append({"step": "Debt/Eq < 1.0", "status": "PASS", "val": f"{de/100:.2f}", "eng": "Safe debt levels."})
        score += 1
    else:
        results.append({"step": "Debt/Eq < 1.0", "status": "FAIL", "val": f"{de/100 if de else 'N/A'}", "eng": "Highly leveraged."})

    # 5. FCF Yield
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf and mcap and (fcf/mcap) > 0.03:
        results.append({"step": "FCF Yield > 3%", "status": "PASS", "val": f"{fcf/mcap:.2%}", "eng": "Real cash machine!"})
        score += 1
    else:
        results.append({"step": "FCF Yield > 3%", "status": "FAIL", "val": "Weak", "eng": "Low cash flow."})

    # 6. Analyst Upside
    try:
        tgt = info.get('targetMeanPrice')
        if tgt and price:
            up = (tgt - price)/price
            if up > 0.10:
                results.append({"step": "Analyst Upside", "status": "PASS", "val": f"+{up:.1%}", "eng": "Experts bullish."})
                score += 1
            else:
                results.append({"step": "Analyst Upside", "status": "FAIL", "val": f"{up:.1%}", "eng": "Experts cautious."})
        else:
            results.append({"step": "Analyst Upside", "status": "FAIL", "val": "N/A", "eng": "No Target"})
    except:
        results.append({"step": "Analyst Upside", "status": "FAIL", "val": "Error", "eng": "No Target"})

    # 7. PEG Ratio
    peg = info.get('pegRatio')
    if peg and peg < 2.0:
        results.append({"step": "PEG < 2.0", "status": "PASS", "val": f"{peg:.2f}", "eng": "Cheap vs Growth."})
        score += 1
    else:
        results.append({"step": "PEG < 2.0", "status": "FAIL", "val": f"{peg if peg else 'N/A'}", "eng": "Overvalued growth."})

    # 8. Current Ratio
    curr = info.get('currentRatio')
    if curr and curr > 1.5:
        results.append({"step": "Curr Ratio > 1.5", "status": "PASS", "val": f"{curr:.2f}", "eng": "Strong liquidity."})
        score += 1
    else:
        results.append({"step": "Curr Ratio > 1.5", "status": "FAIL", "val": f"{curr if curr else 'N/A'}", "eng": "Tight liquidity."})

    # 9. EV/EBITDA
    ev = info.get('enterpriseValue')
    ebit = info.get('ebitda')
    if ev and ebit and (ev/ebit) < 20:
        results.append({"step": "EV/EBITDA < 20", "status": "PASS", "val": f"{ev/ebit:.2f}", "eng": "Good enterprise value."})
        score += 1
    else:
        results.append({"step": "EV/EBITDA", "status": "FAIL", "val": "High", "eng": "Enterprise overpriced."})

    # 10. ROA
    roa = info.get('returnOnAssets')
    if roa and roa > 0.05:
        results.append({"step": "ROA > 5%", "status": "PASS", "val": f"{roa:.2%}", "eng": "Assets used well."})
        score += 1
    else:
        results.append({"step": "ROA > 5%", "status": "FAIL", "val": f"{roa if roa else 'N/A'}", "eng": "Asset inefficient."})

    # 11. Gross Margin
    gm = info.get('grossMargins')
    if gm and gm > 0.40:
        results.append({"step": "Gross Mrg > 40%", "status": "PASS", "val": f"{gm:.2%}", "eng": "High pricing power."})
        score += 1
    else:
        results.append({"step": "Gross Mrg > 40%", "status": "FAIL", "val": f"{gm if gm else 'N/A'}", "eng": "Thin margins."})

    # 12. Institutional Hold
    inst = info.get('heldPercentInstitutions')
    if inst and inst > 0.30:
        results.append({"step": "Inst. Hold > 30%", "status": "PASS", "val": f"{inst:.2%}", "eng": "Banks are buying."})
        score += 1
    else:
        results.append({"step": "Inst. Hold > 30%", "status": "FAIL", "val": f"{inst if inst else 'N/A'}", "eng": "Retail heavy."})

    # 13. DSO (Forensic)
    try:
        if 'Net Receivables' in balance_sheet.index:
            rec = balance_sheet.loc['Net Receivables'].iloc[0]
        else:
            rec = balance_sheet.loc['Accounts Receivable'].iloc[0]
        rev = financials.loc['Total Revenue'].iloc[0]
        dso = (rec / rev) * 365
        if dso < 90:
            results.append({"step": "DSO < 90 Days", "status": "PASS", "val": f"{int(dso)}d", "eng": "Clean collection."})
            score += 1
        else:
            results.append({"step": "DSO < 90 Days", "status": "FAIL", "val": f"{int(dso)}d", "eng": "Client risk!"})
    except:
        results.append({"step": "DSO", "status": "FAIL", "val": "N/A", "eng": "No Data"})

    # 14. Inv Days (Forensic)
    try:
        inv = balance_sheet.loc['Inventory'].iloc[0]
        cogs = financials.loc['Cost Of Revenue'].iloc[0]
        idys = (inv / cogs) * 365
        if idys < 150:
            results.append({"step": "Inv Days < 150", "status": "PASS", "val": f"{int(idys)}d", "eng": "Fast inventory."})
            score += 1
        else:
            results.append({"step": "Inv Days < 150", "status": "FAIL", "val": f"{int(idys)}d", "eng": "Stuck stock!"})
    except:
        results.append({"step": "Inv Days", "status": "FAIL", "val": "N/A", "eng": "No Data"})

    # 15. Op Cash Flow
    try:
        ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
        if ocf > 0:
            results.append({"step": "Cash Flow > 0", "status": "PASS", "val": "Positive", "eng": "Real money earned."})
            score += 1
        else:
            results.append({"step": "Cash Flow > 0", "status": "FAIL", "val": "Negative", "eng": "Burning cash!"})
    except:
        results.append({"step": "Op Cash Flow", "status": "FAIL", "val": "N/A", "eng": "No Data"})

    return score, results, name, summary, ceo, price, curr_sym, stock

# --- APP FLOW ---

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🕵️‍♂️ YnotAI Terminal")
        with st.form("login"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Access Data"):
                if user == "ynot_admin" and pw == "ynot_secure_pass":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Access Denied.")

def main_app():
    with st.sidebar:
        st.write("Logged: **ynot_admin**")
        if st.button("Logout"): st.session_state.authenticated = False; st.rerun()
        st.info("**Intelligence Stack:**\n1. 15-Point Check\n2. AI News Mood\n3. 5-Year Prophet Forecast")

    st.markdown('<div class="main-header">YnotAI Ultimate Dashboard</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1: query = st.text_input("Search Ticker/Company", placeholder="e.g. Reliance, Tejas Networks").strip()
    with col_s2: st.write(""); st.write(""); btn = st.button("Analyze 🚀", type="primary", use_container_width=True)

    if btn and query:
        with st.spinner("Compiling Intelligence..."):
            symbol = get_symbol_from_name(query)
            try:
                score, trace, name, summary, ceo, price, sym, stock_obj = run_full_intelligence(symbol)
                ai_v, ai_c, ai_m = analyze_ai_sentiment(stock_obj)
                st.markdown(f"### 🏢 {name} ({symbol})")
                st.markdown(f'<div class="price-tag">Price: {sym}{price:,.2f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-card"><h4>📝 Summary</h4><p>{summary}</p><div class="ceo-tag">👤 CEO: {ceo}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-card"><h3>🧠 Market Mood: {ai_v}</h3><p>{ai_m}</p></div>', unsafe_allow_html=True)
                
                s_class = "score-high" if score >= 12 else "score-med" if score >= 8 else "score-low"
                v_text = "STRONG BUY" if score >= 12 else "HOLD/CAUTIOUS" if score >= 8 else "HIGH RISK"
                st.markdown(f'<div class="score-box {s_class}"><h1>{score}/15</h1><h3>{v_text}</h3></div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                for i, item in enumerate(trace):
                    css = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "❌"
                    html = f'<div class="metric-card {css}"><div><strong>{icon} {item["step"]}</strong><br><small>{item["eng"]}</small></div><div style="text-align:right"><strong>{item["val"]}</strong></div></div>'
                    if i % 3 == 0: c1.markdown(html, unsafe_allow_html=True)
                    elif i % 3 == 1: c2.markdown(html, unsafe_allow_html=True)
                    else: c3.markdown(html, unsafe_allow_html=True)
                
                st.markdown("---")
                forecast, roi, f_price = predict_future_price(symbol)
                if forecast is not None:
                    roi_c = "#10b981" if roi > 0 else "#ef4444"
                    st.markdown(f'<div class="forecast-box"><h2>Projected 2031: {sym}{f_price:,.2f}</h2><h3 style="color:{roi_c}">ROI: {roi:+.2f}%</h3></div>', unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Trend'))
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")
    st.markdown('<br><br><div class="footer">© 2026 ynotAIbundle | Advanced Forensic</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if st.session_state.authenticated: main_app()
    else: login_screen()
