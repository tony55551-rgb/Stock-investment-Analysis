pip install streamlit yfinance pandas
import streamlit as st
import yfinance as yf
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Strategic Stock Analyzer", page_icon="📊", layout="wide")

# --- CSS STYLING (Clean & Professional) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
    }
    .status-card {
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .pass {
        background-color: #d1fae5;
        color: #065f46;
        border-left: 5px solid #059669;
    }
    .fail {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 5px solid #dc2626;
    }
    .neutral {
        background-color: #f3f4f6;
        color: #374151;
        border-left: 5px solid #6b7280;
    }
    </style>
""", unsafe_allow_html=True)

# --- ANALYSIS LOGIC ENGINE ---

def get_financial_data(ticker):
    """Fetches necessary data objects from yfinance."""
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.financials, stock.balance_sheet

def calculate_cagr(financials):
    """Calculates Revenue CAGR over available years."""
    try:
        if 'Total Revenue' not in financials.index:
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
        
        if 'Net Income' in fin.index and 'Stockholders Equity' in bal.index:
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
    
    # FETCH DATA
    stock, info, financials, balance_sheet = get_financial_data(symbol)
    
    # 1. REVENUE CHECK
    cagr, cagr_str = calculate_cagr(financials)
    if cagr is not None and cagr >= 0.10:
        results.append({"step": "Revenue Growth > 10%", "status": "PASS", "value": cagr_str, "msg": "Growth is healthy."})
    else:
        val = cagr_str if cagr is not None else "N/A"
        return "NO_INVEST", "Low Revenue Growth", results + [{"step": "Revenue Growth > 10%", "status": "FAIL", "value": val, "msg": "Growth is too slow."}]

    # 2. P/E RATIO CHECK
    pe = info.get('trailingPE')
    if pe is not None and pe < 25:
        results.append({"step": "P/E Ratio < 25", "status": "PASS", "value": f"{pe:.2f}", "msg": "Valuation is reasonable."})
    else:
        val = f"{pe:.2f}" if pe is not None else "N/A"
        return "NO_INVEST", "Likely Overvalued", results + [{"step": "P/E Ratio < 25", "status": "FAIL", "value": val, "msg": "Stock is too expensive."}]

    # 3. PEG RATIO CHECK
    peg = info.get('pegRatio')
    if peg is not None and peg < 2:
        results.append({"step": "PEG Ratio < 2", "status": "PASS", "value": f"{peg:.2f}", "msg": "Price aligns with growth."})
    else:
        val = f"{peg:.2f}" if peg is not None else "N/A"
        return "NO_INVEST", "Low Profit Growth (vs Price)", results + [{"step": "PEG Ratio < 2", "status": "FAIL", "value": val, "msg": "PEG is too high."}]

    # 4. ROE CHECK (Average > 5%)
    avg_roe = calculate_avg_roe(stock)
    # Fallback to current ROE if historical calc fails
    if avg_roe is None: 
        avg_roe = info.get('returnOnEquity')
    
    if avg_roe is not None and avg_roe > 0.05:
        results.append({"step": "Avg ROE > 5%", "status": "PASS", "value": f"{avg_roe:.2%}", "msg": "Profitability is stable."})
    else:
        val = f"{avg_roe:.2%}" if avg_roe is not None else "N/A"
        return "NO_INVEST", "Weak Profitability", results + [{"step": "Avg ROE > 5%", "status": "FAIL", "value": val, "msg": "Return on Equity is too low."}]

    # 5. QUICK RATIO CHECK
    quick = info.get('quickRatio')
    if quick is not None and quick > 1.5:
        results.append({"step": "Quick Ratio > 1.5", "status": "PASS", "value": f"{quick:.2f}", "msg": "Strong liquidity."})
    else:
        val = f"{quick:.2f}" if quick is not None else "N/A"
        return "NO_INVEST", "Liquidity Issues", results + [{"step": "Quick Ratio > 1.5", "status": "FAIL", "value": val, "msg": "Company may struggle to pay short-term debts."}]

    # IF ALL PASSED
    return "INVEST", "Solid Investment Candidate", results

# --- UI IMPLEMENTATION ---

# Session State for Login
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_screen():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 Stock Analyzer Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                # Basic mock authentication
                if username == "admin" and password == "password":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. (Try: admin / password)")

def main_app():
    # Sidebar
    with st.sidebar:
        st.write(f"User: **Admin**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.info("This tool follows the strict 'Growth at a Reasonable Price' flowchart.")

    st.markdown('<div class="main-header">Strategic Stock Analyzer</div>', unsafe_allow_html=True)
    st.write("Enter a stock ticker to run the 5-step analysis framework.")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Stock Symbol", placeholder="e.g. AAPL, MSFT, TSLA").upper()
    with col2:
        st.write("") # Spacer
        st.write("")
        analyze_btn = st.button("Run Analysis", type="primary")

    if analyze_btn and symbol:
        with st.spinner(f"Retrieving data for {symbol}..."):
            try:
                decision, reason, trace = run_analysis(symbol)
                
                # --- RESULT DISPLAY ---
                st.markdown("---")
                if decision == "INVEST":
                    st.success(f"### ✅ RECOMMENDATION: {decision}")
                    st.write("This stock passed all 5 checks in the flowchart.")
                else:
                    st.error(f"### 🛑 RECOMMENDATION: DO NOT INVEST")
                    st.warning(f"**Reason:** {reason}")

                # --- STEP-BY-STEP BREAKDOWN ---
                st.subheader("Analysis Breakdown")
                
                for item in trace:
                    css_class = "pass" if item['status'] == "PASS" else "fail"
                    icon = "✅" if item['status'] == "PASS" else "❌"
                    
                    st.markdown(f"""
                        <div class="status-card {css_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <strong>{icon} {item['step']}</strong><br>
                                    <span style="font-size:0.9em">{item['msg']}</span>
                                </div>
                                <div style="text-align:right;">
                                    <h2>{item['value']}</h2>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error("Error retrieving data. Please check the ticker symbol.")
                st.write(e)

# --- APP FLOW CONTROL ---
if st.session_state.authenticated:
    main_app()
else:
    login_screen()
