import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YnotAI Stock Analyzer", page_icon="📈", layout="centered")

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    .score-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        color: white; 
    }
    .score-high { background-color: #059669; } /* Green */
    .score-med { background-color: #d97706; }  /* Orange */
    .score-low { background-color: #dc2626; }  /* Red */
    
    /* FIXED SECTION: FORCE BLACK TEXT ON CARDS */
    .metric-card {
        background-color: #ffffff;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 10px solid #ccc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card strong {
        color: #000000 !important;
        font-size: 1.1em;
    }
    .metric-card small {
        color: #555555 !important;
    }
    
    .card-pass { border-left-color: #059669; }
    .card-fail { border-left-color: #dc2626; }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_symbol_from_name(query):
    """
    Attempts to find a ticker symbol from a company name query.
    Uses a standard lookup approach.
    """
    query = query.strip()
    
    # If it looks like a ticker (short, uppercase), return it directly
    if query.isupper() and len(query) <= 5 and " " not in query:
        return query
    
    try:
        # Use Yahoo Finance auto-complete API to find the ticker
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'quotes' in data and len(data['quotes']) > 0:
            # Return the first matching symbol
            return data['quotes'][0]['symbol']
    except:
        pass
    
    # Fallback: Return original query (assume user typed ticker)
    return query.upper()

def get_financial_data(ticker):
    """Fetches necessary data objects from yfinance."""
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.financials, stock.balance_sheet

def calculate_cagr(financials):
    try:
        if financials.empty or 'Total Revenue' not in financials.index:
            return None, "Data Missing"
        revenues = financials.loc['Total Revenue'].iloc[::-1]
        years = len(revenues)
        if years < 2: return None, "Insufficient History"
        start_rev = revenues.iloc[0]
        end_rev = revenues.iloc[-1]
        cagr = (end_rev / start_rev) ** (1 / (years - 1)) - 1
        return cagr, f"{cagr:.2%}"
    except:
        return None, "Error"

def calculate_avg_roe(stock):
    try:
        fin = stock.financials
        bal = stock.balance_sheet
        if not fin.empty and not bal.empty and 'Net Income' in fin.index and 'Stockholders Equity' in bal.index:
            net_income = fin.loc['Net Income']
            equity = bal.loc['Stockholders Equity']
            roes = []
            for date in net_income.index:
                if date in equity.index:
                    roes.append(net_income[date] / equity[date])
            if not roes: return None
            return sum(roes) / len(roes)
        return None
    except:
        return None

def run_analysis(symbol):
    results = []
    score = 0
    stock, info, financials, balance_sheet = get_financial_data(symbol)
    
    # Get Company Name for display
    company_name = info.get('longName', symbol)
    
    # 1. REVENUE GROWTH CHECK
    cagr, cagr_str = calculate_cagr(financials)
    if cagr is not None and cagr >= 0.10:
        results.append({"step": "Revenue Growth > 10%", "status": "PASS", "value": cagr_str, "msg": "High Growth 🚀"})
        score += 1
    else:
        val = cagr_str if cagr is not None else "N/A"
        results.append({"step": "Revenue Growth > 10%", "status": "FAIL", "value": val, "msg": "Growth too slow 🐢"})

    # 2. P/E RATIO CHECK
    pe = info.get('trailingPE')
    if pe is not None and pe < 25:
        results.append({"step": "P/E Ratio < 25", "status": "PASS", "value": f"{pe:.2f}", "msg": "Good Value 💰"})
        score += 1
    else:
        val = f"{pe:.2f}" if pe is not None else "N/A"
        results.append({"step": "P/E Ratio < 25", "status": "FAIL", "value": val, "msg": "Expensive 💸"})

    # 3. PEG RATIO CHECK
    peg = info.get('pegRatio')
    if peg is not None and peg < 2:
        results.append({"step": "PEG Ratio < 2", "status": "PASS", "value": f"{peg:.2f}", "msg": "Price justifies growth ✅"})
        score += 1
    else:
        val = f"{peg:.2f}" if peg is not None else "N/A"
        results.append({"step": "PEG Ratio < 2", "status": "FAIL", "value": val, "msg": "Price ahead of growth ❌"})

    # 4. ROE CHECK
    avg_roe = calculate_avg_roe(stock)
    if avg_roe is None: avg_roe = info.get('returnOnEquity')
    if avg_roe is not None and avg_roe > 0.05:
        results.append({"step": "ROE > 5%", "status": "PASS", "value": f"{avg_roe:.2%}", "msg": "Efficient Profits 🏭"})
        score += 1
    else:
        val = f"{avg_roe:.2%}" if avg_roe is not None else "N/A"
        results.append({"step": "ROE > 5%", "status": "FAIL", "value": val, "msg": "Low Efficiency 📉"})

    # 5. QUICK RATIO CHECK
    quick = info.get('quickRatio')
    if quick is not None and quick > 1.5:
        results.append({"step": "Quick Ratio > 1.5", "status": "PASS", "value": f"{quick:.2f}", "msg": "Safe Liquidity 🛡️"})
        score += 1
    else:
        val = f"{quick:.2f}" if quick is not None else "N/A"
        results.append({"step": "Quick Ratio > 1.5", "status": "FAIL", "value": val, "msg": "Debt Risk ⚠️"})

    return score, results, company_name

# --- MAIN UI ---

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 YnotAI Stock Analyzer")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter")
            if submitted:
                if username == "ynot" and password == "Str0ng@Pulse#884":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied.")

def footer():
    st.markdown("""
        <div class="footer">
            <p>Copyright © 2024 ynotAIbundle. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def main_app():
    with st.sidebar:
        st.write("Logged in as: **ynot_admin**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("---")
        st.write("### 📜 Strategy Rules")
        st.write("1. Rev Growth > 10%")
        st.write("2. P/E < 25")
        st.write("3. PEG < 2")
        st.write("4. ROE > 5%")
        st.write("5. Quick Ratio > 1.5")

    st.markdown('<div class="main-header">YnotAI Stock Analyzer</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # UPDATED: Placeholder text changed to encourage Names
        search_query = st.text_input("Enter Company Name or Ticker", placeholder="e.g. Apple, Tesla, Reliance").strip()
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if analyze_btn and search_query:
        # STEP 1: RESOLVE NAME TO TICKER
        with st.spinner(f"Searching for '{search_query}'..."):
            symbol = get_symbol_from_name(search_query)
        
        # STEP 2: RUN ANALYSIS
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                score, trace, full_name = run_analysis(symbol)
                
                # Show resolved name
                st.info(f"Analyzed: **{full_name} ({symbol})**")
                
                if score == 5:
                    res_class = "score-high"
                    verdict = "💎 PERFECT BUY"
                    st.balloons()
                elif score >= 3:
                    res_class = "score-med"
                    verdict = "⚠️ MODERATE BUY"
                else:
                    res_class = "score-low"
                    verdict = "🛑 DO NOT INVEST"

                st.markdown(f"""
                    <div class="score-box {res_class}">
                        <h1>{score} / 5</h1>
                        <h3>{verdict}</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("Detailed Breakdown")
                for item in trace:
                    card_class = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "❌"
                    st.markdown(f"""
                        <div class="metric-card {card_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <strong>{icon} {item['step']}</strong><br>
                                    <small>{item['msg']}</small>
                                </div>
                                <div style="text-align:right;">
                                    <strong>{item['value']}</strong>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not find data for '{search_query}'. Try using the exact ticker symbol.")
                
    footer()

if st.session_state.authenticated:
    main_app()
else:
    login_screen()
    footer()
