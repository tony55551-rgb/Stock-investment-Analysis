import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob  # The AI Library

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Pro Analyzer", page_icon="🧠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    
    .score-box { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .score-high { background: linear-gradient(135deg, #059669, #10b981); } 
    .score-med { background: linear-gradient(135deg, #d97706, #f59e0b); } 
    .score-low { background: linear-gradient(135deg, #dc2626, #ef4444); }
    
    /* AI CARD STYLING */
    .ai-card {
        background: linear-gradient(135deg, #6366f1, #8b5cf6); /* Purple Gradient */
        color: white !important;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        border: 1px solid #7c3aed;
    }
    .ai-card h3 { color: white !important; margin: 0; }
    .ai-card p { color: #e0e7ff !important; margin: 5px 0 0 0; font-size: 1.1rem; }

    /* METRIC CARDS */
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

# --- AI LAYER: NEWS SENTIMENT ---
def analyze_ai_sentiment(stock):
    """
    Fetches latest news and runs Sentiment Analysis using TextBlob.
    Returns: Label, Color, Explanation
    """
    try:
        news = stock.news
        if not news:
            return "Neutral / No News", "#9ca3af", "No recent news headlines found to analyze."
        
        score_total = 0
        count = 0
        
        # Analyze up to 7 recent headlines
        for item in news[:7]:
            title = item.get('title', '')
            if title:
                analysis = TextBlob(title)
                score_total += analysis.sentiment.polarity
                count += 1
        
        if count == 0: return "Neutral", "#9ca3af", "Could not analyze news text."
        
        avg_score = score_total / count
        
        # Determine Verdict
        if avg_score > 0.1:
            return "Positive (Bullish) 🐂", "High", "News headlines are optimistic and positive."
        elif avg_score < -0.1:
            return "Negative (Bearish) 🐻", "Low", "News headlines contain negative sentiment."
        else:
            return "Neutral 😐", "Med", "News is mixed or strictly factual."
            
    except Exception as e:
        return "AI Error", "Med", "Could not run AI analysis."

def run_pro_analysis(symbol):
    results = []
    score = 0
    
    stock, info, financials, balance_sheet, cashflow = get_financial_data(symbol)
    
    # Metadata
    company_name = info.get('longName', symbol)
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    currency = info.get('currency', 'USD')
    curr_sym = get_currency_symbol(currency)
    
    # --- AI ANALYSIS ---
    ai_verdict, ai_strength, ai_msg = analyze_ai_sentiment(stock)
    
    # --- 1. REVENUE GROWTH ---
    try:
        revs = financials.loc['Total Revenue'].iloc[::-1]
        if len(revs) > 1:
            cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / (len(revs) - 1)) - 1
            val_str = f"{cagr:.2%}"
            if cagr >= 0.10:
                results.append({"step": "Rev Growth > 10%", "status": "PASS", "val": val_str, "english": "Sales growing fast (+10%/yr)."})
                score += 1
            else:
                results.append({"step": "Rev Growth > 10%", "status": "FAIL", "val": val_str, "english": "Sales growing too slowly."})
        else: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "Insufficient data."})
    except: results.append({"step": "Rev Growth", "status": "FAIL", "val": "Error", "english": "Data unavailable."})

    # --- 2. P/E RATIO ---
    pe = info.get('trailingPE')
    if pe and pe < 30: 
        results.append({"step": "P/E Ratio < 30", "status": "PASS", "val": f"{pe:.2f}", "english": "Stock price is fair vs profit."})
        score += 1
    else:
        results.append({"step": "P/E Ratio < 30", "status": "FAIL", "val": f"{pe if pe else 'N/A'}", "english": "Stock might be expensive."})

    # --- 3. ROE ---
    roe = info.get('returnOnEquity')
    if roe and roe > 0.10: 
        results.append({"step": "ROE > 10%", "status": "PASS", "val": f"{roe:.2%}", "english": "High efficiency."})
        score += 1
    else:
        results.append({"step": "ROE > 10%", "status": "FAIL", "val": f"{roe:.2%}" if roe else "N/A", "english": "Low returns on capital."})

    # --- 4. DEBT TO EQUITY ---
    de = info.get('debtToEquity') 
    if de is not None:
        ratio = de / 100
        if ratio < 1.0: 
            results.append({"step": "Debt/Equity < 1.0", "status": "PASS", "val": f"{ratio:.2f}", "english": "Low debt. Safe."})
            score += 1
        else:
            results.append({"step": "Debt/Equity < 1.0", "status": "FAIL", "val": f"{ratio:.2f}", "english": "High debt. Risky."})
    else: results.append({"step": "Debt/Equity", "status": "FAIL", "val": "N/A", "english": "Data Missing"})

    # --- 5. FCF YIELD ---
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf is None and not cashflow.empty:
        try:
            if 'Free Cash Flow' in cashflow.index: fcf = cashflow.loc['Free Cash Flow'].iloc[0]
            elif 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                fcf = cashflow.loc['Operating Cash Flow'].iloc[0] + cashflow.loc['Capital Expenditure'].iloc[0]
        except: pass
    if fcf and mcap:
        fcf_yield = fcf / mcap
        if fcf_yield > 0.03: 
            results.append({"step": "FCF Yield > 3%", "status": "PASS", "val": f"{fcf_yield:.2%}", "english": "Generating strong cash."})
            score += 1
        else:
            results.append({"step": "FCF Yield > 3%", "status": "FAIL", "val": f"{fcf_yield:.2%}", "english": "Low cash generation."})
    else: results.append({"step": "FCF Yield", "status": "FAIL", "val": "N/A", "english": "Data Missing"})

    # --- 6. FUTURE UPSIDE ---
    target_price = info.get('targetMeanPrice')
    if target_price and current_price:
        upside = (target_price - current_price) / current_price
        target_display = f"{curr_sym}{target_price:,.2f}"
        if upside > 0.10: 
            results.append({"step": "Analyst Upside > 10%", "status": "PASS", "val": f"+{upside:.1%}", "english": f"Experts target: {target_display}"})
            score += 1
        else:
            results.append({"step": "Analyst Upside > 10%", "status": "FAIL", "val": f"{upside:.1%}", "english": f"Target too low ({target_display})"})
    else: results.append({"step": "Future Upside", "status": "FAIL", "val": "N/A", "english": "No analyst data."})

    # --- 7. PEG RATIO ---
    peg = info.get('pegRatio')
    if peg and peg < 2.0:
        results.append({"step": "PEG Ratio < 2", "status": "PASS", "val": f"{peg:.2f}", "english": "Undervalued for growth."})
        score += 1
    else: results.append({"step": "PEG Ratio < 2", "status": "FAIL", "val": f"{peg if peg else 'N/A'}", "english": "Overvalued for growth."})

    # --- 8. CURRENT RATIO ---
    curr = info.get('currentRatio')
    if curr and curr > 1.5:
        results.append({"step": "Current Ratio > 1.5", "status": "PASS", "val": f"{curr:.2f}", "english": "Can pay short-term bills."})
        score += 1
    else: results.append({"step": "Current Ratio > 1.5", "status": "FAIL", "val": f"{curr if curr else 'N/A'}", "english": "Liquidity tight."})

    return score, results, company_name, current_price, curr_sym, ai_verdict, ai_msg

# --- MAIN APP LOGIC ---

if 'authenticated' not in st.session_state: st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🧠 YnotAI Stock Analyzer Login")
        with st.form("login"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Access Terminal"):
                if user == "ynot" and pw == "Str0ng@Pulse#884":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Access Denied")

def footer():
    st.markdown('<div class="footer">© 2026 ynotAIbundle | Pro + AI Edition</div>', unsafe_allow_html=True)

def main_app():
    with st.sidebar:
        st.write("User: **ynot_admin**")
        if st.button("Logout"): 
            st.session_state.authenticated = False 
            st.rerun()
        st.markdown("---")
        st.info("**Checklist:**\n1. Rev Growth\n2. P/E Ratio\n3. ROE\n4. Debt/Equity\n5. Free Cash Flow\n6. Future Upside\n7. PEG Ratio\n8. Current Ratio\n\n**+ AI Sentiment**")

    st.markdown('<div class="main-header">YnotAI Professional</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fundamental + AI Sentiment Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Ticker or Company", placeholder="e.g. Reliance, Tata Motors, Apple").strip()
    with col2:
        st.write(""); st.write("")
        btn = st.button("Analyze", type="primary", use_container_width=True)

    if btn and query:
        with st.spinner(f"Running AI & Financial models for '{query}'..."):
            symbol = get_symbol_from_name(query)
            try:
                score, trace, name, price, sym, ai_verdict, ai_msg = run_pro_analysis(symbol)
                
                st.markdown("---")
                
                # HEADER
                st.markdown(f"### 🏢 {name} ({symbol})")
                st.markdown(f'<div class="price-tag">Current Price: {sym}{price:,.2f}</div>', unsafe_allow_html=True)
                
                # --- NEW AI CARD ---
                st.markdown(f"""
                    <div class="ai-card">
                        <h3>🧠 AI Market Mood: {ai_verdict}</h3>
                        <p>{ai_msg}</p>
                    </div>
                """, unsafe_allow_html=True)

                # SCORECARD
                if score >= 7:
                    s_class = "score-high"
                    verdict = "STRONG BUY (Institutional Grade)"
                    st.balloons()
                elif score >= 5:
                    s_class = "score-med"
                    verdict = "HOLD / MODERATE BUY"
                else:
                    s_class = "score-low"
                    verdict = "UNDERPERFORM / SELL"
                
                st.markdown(f"""
                    <div class="score-box {s_class}">
                        <div style="font-size: 4rem; font-weight: bold;">{score} / 8</div>
                        <div style="font-size: 1.5rem;">{verdict}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # METRICS GRID
                st.subheader("Fundamental Breakdown")
                c1, c2 = st.columns(2)
                
                for i, item in enumerate(trace):
                    css = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "⚠️"
                    html = f"""
                    <div class="metric-card {css}">
                        <div style="display:flex; justify-content:space-between;">
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
                    if i % 2 == 0: c1.markdown(html, unsafe_allow_html=True)
                    else: c2.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                 st.error(f"Error analyzing {symbol}. Please ensure the ticker is correct.")
                 st.write(f"Debug Info: {e}")

    footer()

if st.session_state.authenticated: main_app()
else: login_screen(); footer()
