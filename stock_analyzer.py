import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Pro Analyzer", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    
    .score-box { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .score-high { background: linear-gradient(135deg, #059669, #10b981); } 
    .score-med { background: linear-gradient(135deg, #d97706, #f59e0b); } 
    .score-low { background: linear-gradient(135deg, #dc2626, #ef4444); }
    
    /* CARDS */
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
    if query.isupper() and len(query) <= 5 and " " not in query: return query
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0: return data['quotes'][0]['symbol']
    except: pass
    return query.upper()

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.financials, stock.balance_sheet

def run_pro_analysis(symbol):
    results = []
    score = 0
    
    stock, info, financials, balance_sheet = get_financial_data(symbol)
    company_name = info.get('longName', symbol)
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    
    # --- 1. REVENUE GROWTH (CAGR) ---
    try:
        revs = financials.loc['Total Revenue'].iloc[::-1]
        if len(revs) > 1:
            cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / (len(revs) - 1)) - 1
            val_str = f"{cagr:.2%}"
            if cagr >= 0.10:
                results.append({"step": "Rev Growth > 10%", "status": "PASS", "val": val_str, "english": "Company sales are growing fast (+10%/yr)."})
                score += 1
            else:
                results.append({"step": "Rev Growth > 10%", "status": "FAIL", "val": val_str, "english": "Sales are growing too slowly."})
        else: results.append({"step": "Rev Growth", "status": "FAIL", "val": "N/A", "english": "Not enough data to tell."})
    except: results.append({"step": "Rev Growth", "status": "FAIL", "val": "Error", "english": "Data unavailable."})

    # --- 2. P/E RATIO ---
    pe = info.get('trailingPE')
    if pe and pe < 30: 
        results.append({"step": "P/E Ratio < 30", "status": "PASS", "val": f"{pe:.2f}", "english": "Stock price is fair compared to its profit."})
        score += 1
    else:
        results.append({"step": "P/E Ratio < 30", "status": "FAIL", "val": f"{pe if pe else 'N/A'}", "english": "Stock might be too expensive right now."})

    # --- 3. ROE (Return on Equity) ---
    roe = info.get('returnOnEquity')
    if roe and roe > 0.10: 
        results.append({"step": "ROE > 10%", "status": "PASS", "val": f"{roe:.2%}", "english": "Management is very efficient at using your money."})
        score += 1
    else:
        results.append({"step": "ROE > 10%", "status": "FAIL", "val": f"{roe:.2%}" if roe else "N/A", "english": "Management isn't generating enough profit from investments."})

    # --- 4. DEBT TO EQUITY (Solvency) ---
    de = info.get('debtToEquity') 
    if de is not None:
        ratio = de / 100
        if ratio < 1.0: 
            results.append({"step": "Debt/Equity < 1.0", "status": "PASS", "val": f"{ratio:.2f}", "english": "Company has low debt. Very safe."})
            score += 1
        else:
            results.append({"step": "Debt/Equity < 1.0", "status": "FAIL", "val": f"{ratio:.2f}", "english": "Company has high debt. Risky."})
    else: results.append({"step": "Debt/Equity", "status": "FAIL", "val": "N/A", "english": "Data Missing"})

    # --- 5. FREE CASH FLOW YIELD ---
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    if fcf and mcap:
        fcf_yield = fcf / mcap
        if fcf_yield > 0.03: 
            results.append({"step": "FCF Yield > 3%", "status": "PASS", "val": f"{fcf_yield:.2%}", "english": "Company generates lots of real cash (not just paper profit)."})
            score += 1
        else:
            results.append({"step": "FCF Yield > 3%", "status": "FAIL", "val": f"{fcf_yield:.2%}", "english": "Company is not generating enough liquid cash."})
    else: results.append({"step": "FCF Yield", "status": "FAIL", "val": "N/A", "english": "Data Missing"})

    # --- 6. FUTURE UPSIDE (Proxy for Order Book/Future Orders) ---
    # We use Analyst Target Price vs Current Price to see if "Future Orders" are expected to drive value
    target_price = info.get('targetMeanPrice')
    if target_price and current_price:
        upside = (target_price - current_price) / current_price
        if upside > 0.10: # Analysts expect 10% growth
            results.append({"step": "Analyst Upside > 10%", "status": "PASS", "val": f"+{upside:.1%}", "english": f"Experts predict price will rise to ${target_price}."})
            score += 1
        else:
            results.append({"step": "Analyst Upside > 10%", "status": "FAIL", "val": f"{upside:.1%}", "english": f"Experts think price might stay flat or drop (Target: ${target_price})."})
    else:
        results.append({"step": "Future Upside", "status": "FAIL", "val": "N/A", "english": "No analyst predictions available."})


    # --- 7. PEG RATIO ---
    peg = info.get('pegRatio')
    if peg and peg < 2.0:
        results.append({"step": "PEG Ratio < 2", "status": "PASS", "val": f"{peg:.2f}", "english": "Price is cheap considering how fast it's growing."})
        score += 1
    else:
        results.append({"step": "PEG Ratio < 2", "status": "FAIL", "val": f"{peg if peg else 'N/A'}", "english": "Price is too high for the current growth rate."})

    # --- 8. CURRENT RATIO ---
    curr = info.get('currentRatio')
    if curr and curr > 1.5:
        results.append({"step": "Current Ratio > 1.5", "status": "PASS", "val": f"{curr:.2f}", "english": "Can easily pay off short-term bills."})
        score += 1
    else:
        results.append({"step": "Current Ratio > 1.5", "status": "FAIL", "val": f"{curr if curr else 'N/A'}", "english": "Might struggle to pay bills if business slows down."})

    return score, results, company_name, current_price

# --- MAIN APP LOGIC ---

if 'authenticated' not in st.session_state: st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🏛️ ynotAI Stock Analyzer")
        with st.form("login"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Access Terminal"):
                if user == "ynot" and pw == "Str0ng@Pulse#884":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Access Denied")

def footer():
    st.markdown('<div class="footer">© 2024 ynotAIbundle | Professional Edition</div>', unsafe_allow_html=True)

def main_app():
    with st.sidebar:
        st.write("User: **ynot_admin**")
        if st.button("Logout"): 
            st.session_state.authenticated = False 
            st.rerun()
        st.markdown("---")
        st.info("**Analysis Checklist:**\n1. Revenue Growth\n2. P/E Ratio\n3. ROE\n4. Debt/Equity\n5. Free Cash Flow\n6. Future Upside\n7. PEG Ratio\n8. Current Ratio")

    st.markdown('<div class="main-header">YnotAI Professional Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Institutional Grade Fundamental Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Ticker or Company", placeholder="e.g. MSFT, JP Morgan, Pfizer").strip()
    with col2:
        st.write(""); st.write("")
        btn = st.button("Analyze", type="primary", use_container_width=True)

    if btn and query:
        with st.spinner(f"Fetching institutional data for '{query}'..."):
            symbol = get_symbol_from_name(query)
            try:
                score, trace, name, price = run_pro_analysis(symbol)
                
                st.markdown("---")
                
                # HEADER WITH PRICE
                st.markdown(f"### 🏢 {name} ({symbol})")
                st.markdown(f'<div class="price-tag">Current Price: ${price:,.2f}</div>', unsafe_allow_html=True)
                
                # SCORECARD LOGIC
                if score >= 7:
                    st.balloons()
                    s_class = "score-high"
                    verdict = "STRONG BUY (Institutional Grade)"
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
                 st.write(e)

    footer()

if st.session_state.authenticated: main_app()
else: login_screen(); footer()
