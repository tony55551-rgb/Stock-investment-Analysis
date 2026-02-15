import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

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
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid #ccc;
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

def get_financial_data(ticker):
    """Fetches necessary data objects from yfinance."""
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.financials, stock.balance_sheet

def calculate_cagr(financials):
    """Calculates Revenue CAGR over available years."""
    try:
        if financials.empty or 'Total Revenue' not in financials.index:
            return None, "Data Missing"
        
        # Get revenue rows and reverse (oldest to newest)
        revenues = financials.loc['Total Revenue'].iloc[::-1]
        years = len(revenues)
        if years < 2:
            return None, "Insufficient History"
            
        start_rev = revenues.iloc[0]
        end_rev = revenues.iloc[-1]
        
        # CAGR Formula: (End/Start)^(1/n) - 1
        cagr = (end_rev / start_rev) ** (1 / (years - 1)) - 1
        return cagr, f"{cagr:.2%}"
    except:
        return None, "Error"

def calculate_avg_roe(stock):
    """Calculates Average ROE based on available Net Income / Stockholder Equity."""
    try:
        fin = stock.financials
        bal = stock.balance_sheet
        
        if not fin.empty and not bal.empty and 'Net Income' in fin.index and 'Stockholders Equity' in bal.index:
            net_income = fin.loc['Net Income']
            equity = bal.loc['Stockholders Equity']
            
            # Calculate ROE for each matching year
            roes = []
            for date in net_income.index:
                if date in equity.index:
                    roes.append(net_income[date] / equity[date])
            
            if not roes:
                return None
            
            avg_roe = sum(roes) / len(roes)
            return avg_roe
        return None
    except:
        return None

def run_analysis(symbol):
    results = []
    score = 0
    
    # FETCH DATA
    stock, info, financials, balance_sheet = get_financial_data(symbol)
    
    # 1. REVENUE GROWTH CHECK (> 10%)
    cagr, cagr_str = calculate_cagr(financials)
    if cagr is not None and cagr >= 0.10:
        results.append({"step": "Revenue Growth > 10%", "status": "PASS", "value": cagr_str, "msg": "High Growth 🚀"})
        score += 1
    else:
        val = cagr_str if cagr is not None else "N/A"
        results.append({"step": "Revenue Growth > 10%", "status": "FAIL", "value": val, "msg": "Growth too slow 🐢"})

    # 2. P/E RATIO CHECK (< 25)
    pe = info.get('trailingPE')
    if pe is not None and pe < 25:
        results.append({"step": "P/E Ratio < 25", "status": "PASS", "value": f"{pe:.2f}", "msg": "Good Value 💰"})
        score += 1
    else:
        val = f"{pe:.2f}" if pe is not None else "N/A"
        results.append({"step": "P/E Ratio < 25", "status": "FAIL", "value": val, "msg": "Expensive 💸"})

    # 3. PEG RATIO CHECK (< 2)
    peg = info.get('pegRatio')
    if peg is not None and peg < 2:
        results.append({"step": "PEG Ratio < 2", "status": "PASS", "value": f"{peg:.2f}", "msg": "Price justifies growth ✅"})
        score += 1
    else:
        val = f"{peg:.2f}" if peg is not None else "N/A"
        results.append({"step": "PEG Ratio < 2", "status": "FAIL", "value": val, "msg": "Price ahead of growth ❌"})

    # 4. ROE CHECK (> 5%)
    avg_roe = calculate_avg_roe(stock)
    if avg_roe is None: avg_roe = info.get('returnOnEquity') # Fallback to current ROE
    
    if avg_roe is not None and avg_roe > 0.05:
        results.append({"step": "ROE > 5%", "status": "PASS", "value": f"{avg_roe:.2%}", "msg": "Efficient Profits 🏭"})
        score += 1
    else:
        val = f"{avg_roe:.2%}" if avg_roe is not None else "N/A"
        results.append({"step": "ROE > 5%", "status": "FAIL", "value": val, "msg": "Low Efficiency 📉"})

    # 5. QUICK RATIO CHECK (> 1.5)
    quick = info.get('quickRatio')
    if quick is not None and quick > 1.5:
        results.append({"step": "Quick Ratio > 1.5", "status": "PASS", "value": f"{quick:.2f}", "msg": "Safe Liquidity 🛡️"})
        score += 1
    else:
        val = f"{quick:.2f}" if quick is not None else "N/A"
        results.append({"step": "Quick Ratio > 1.5", "status": "FAIL", "value": val, "msg": "Debt Risk ⚠️"})

    return score, results

# --- MAIN UI ---

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 YnotAI Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter")
            
            if submitted:
                # --- YOUR DEDICATED CREDENTIALS ---
                if username == "ynot_admin" and password == "ynot_secure_pass":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied. Incorrect credentials.")

def footer():
    st.markdown("""
        <div class="footer">
            <p>Copyright © 2026 ynotAIbundle. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def main_app():
    # Sidebar
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

    # Main Content
    st.markdown('<div class="main-header">YnotAI Stock Analyzer</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Stock Ticker", placeholder="e.g. AAPL, NVDA, TSLA").upper()
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                score, trace = run_analysis(symbol)
                
                # Determine Color & Message
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

                st.markdown("---")
                
                # Display Scorecard
                st.markdown(f"""
                    <div class="score-box {res_class}">
                        <h1>{score} / 5</h1>
                        <h3>{verdict}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Display Individual Steps
                st.subheader("Detailed Breakdown")
                
                for item in trace:
                    # CSS logic for individual cards
                    card_class = "card-pass" if item['status'] == "PASS" else "card-fail"
                    icon = "✅" if item['status'] == "PASS" else "❌"
                    
                    st.markdown(f"""
                        <div class="metric-card {card_class}">
                            <div style="display:flex; justify-content:space-between;">
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
                st.error(f"Error analyzing {symbol}. Please check if the ticker is correct.")
                st.write(f"Debug: {e}")
                
    footer()

# --- APP FLOW CONTROL ---
if st.session_state.authenticated:
    main_app()
else:
    login_screen()
    footer()
