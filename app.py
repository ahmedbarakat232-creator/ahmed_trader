import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as fgo
from datetime import datetime, timedelta
import urllib.request
import json
import os
from streamlit_autorefresh import st_autorefresh

# استيراد أدوات التعلم الآلي
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# تفعيل التحديث التلقائي المستمر كل 30 ثانية لتحديث البيانات اللحظية
st_autorefresh(interval=30000, key="mobile_refresh_v143")

# إعداد الصفحة لتناسب شاشات الجوال تماماً
st.set_page_config(page_title="منصة AI v14.3 اللحظية", layout="centered", initial_sidebar_state="collapsed")

# كود CSS لتجميل العناصر
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    h1 { font-size: 1.5rem !important; text-align: center; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
    div[data-testid="metric-container"] { background-color: #1e272e; border-radius: 10px; padding: 8px; text-align: center; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 8px 12px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🦅 منظومة التداول v14.3 (تحليل لحظي مستمر)")

# ==================== نظام حفظ الإعدادات ====================
DB_FILE = "watchlist_db.json"

def load_saved_settings():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {
        "watchlist": "NVDA,ORCL,DV,CTSH,QNRX,AMD,SPRE,HLAL,SPUS,U,AG,SPCX,AAPL,PAAS,GC=F",
        "phone_number": "",
        "notifications_active": True
    }

def save_settings(settings_dict):
    try:
        with open(DB_FILE, "w") as f: json.dump(settings_dict, f)
    except: pass

settings = load_saved_settings()

# ==================== إعدادات المحفظة والـ API ====================
with st.expander("⚙️ إعدادات المحفظة والاتصال والإشعارات"):
    watchlist_input = st.text_area("أدخل الرموز:", value=settings.get("watchlist", "NVDA,ORCL,DV,CTSH,QNRX,AMD,SPRE,HLAL,SPUS,U,AG,SPCX,AAPL,PAAS,GC=F"))
    phone_input = st.text_input("📱 رقم الجوال:", value=settings.get("phone_number", ""))
    notif_active = st.checkbox("🔔 تشغيل الإشعارات", value=settings.get("notifications_active", True))
    
    if (watchlist_input != settings.get("watchlist") or phone_input != settings.get("phone_number") or notif_active != settings.get("notifications_active")):
        settings["watchlist"] = watchlist_input
        settings["phone_number"] = phone_input
        settings["notifications_active"] = notif_active
        save_settings(settings)
        st.success("💾 تم حفظ الإعدادات!")
    
    API_KEY = st.text_input("مفتاح Gemini API:", type="password")
    use_gen_ai = st.checkbox("🔥 تفعيل مستشار الأخبار الذكي", value=True)

symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
clean_symbols_list = [str(s).upper() for s in symbols]

# ==================== الدوال الحسابية ====================
def calculate_indicators(df):
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        direction = typical_price.diff()
        pos_flow = raw_money_flow.where(direction > 0, 0.0)
        neg_flow = raw_money_flow.where(direction < 0, 0.0)
        pos_mf14 = pos_flow.rolling(window=14).sum()
        neg_mf14 = neg_flow.rolling(window=14).sum()
        m_ratio = pos_mf14 / (neg_mf14 + 1e-10)
        df['MFI_14'] = 100 - (100 / (1 + m_ratio))
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        return df
    except: return df

@st.cache_data(ttl=5)
def fetch_clean_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, group_by='ticker')
        if data.empty: data = yf.download(symbol, period="60d", interval="4h", progress=False, group_by='ticker')
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
        return calculate_indicators(data)
    except: return pd.DataFrame()

def run_ml_prediction(df):
    try:
        ml_df = df[['Close', 'RSI_14', 'MFI_14', 'ATR']].dropna().copy()
        if len(ml_df) < 30: return 50.0
        ml_df['Target'] = (ml_df['Close'].shift(-3) > ml_df['Close']).astype(int)
        features = ['RSI_14', 'MFI_14', 'ATR']
        X = ml_df[features].iloc[:-3]
        y = ml_df['Target'].iloc[:-3]
        if len(X) < 10 or len(np.unique(y)) < 2: return 50.0
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X_scaled, y)
        latest_features = np.array([ml_df[features].iloc[-1]])
        latest_scaled = scaler.transform(latest_features)
        return model.predict(latest_scaled)[0] * 100
    except: return 50.0

def calculate_scores_and_decision(df, symbol="", enable_ai=True, ml_value=50.0):
    if df.empty or len(df) < 20: return 50, 50, "🟡 انتظار"
    latest = df.iloc[-1]
    close_p = float(latest['Close'])
    rsi_p = float(latest['RSI_14'])
    mfi_p = float(latest['MFI_14'])
    ema200_p = float(latest['EMA200'])
    ema9_p = float(latest['EMA9'])
    ema21_p = float(latest['EMA21'])
    vol_now = float(latest['Volume'])
    vol_avg = float(latest['Vol_MA20'])
    buy_score = 30
    if close_p > ema200_p: buy_score += 15
    if rsi_p < 40: buy_score += 20
    if mfi_p < 35: buy_score += 20
    if ema9_p > ema21_p: buy_score += 15
    if vol_now > vol_avg: buy_score += 10
    if enable_ai: buy_score += (ml_value - 50) * 0.4
    final_buy_score = max(0, min(100, int(buy_score)))
    sell_score = 30
    if close_p < ema200_p: sell_score += 15
    if rsi_p > 60: sell_score += 20
    if mfi_p > 65: sell_score += 20
    if ema9_p < ema21_p: sell_score += 15
    if vol_now > vol_avg: sell_score += 10
    if enable_ai: sell_score += (50 - ml_value) * 0.4
    final_sell_score = max(0, min(100, int(sell_score)))
    if final_buy_score >= 80: decision = "🟢 شراء قوي"
    elif final_sell_score >= 80: decision = "🔴 بيع قوي"
    else: decision = "🟡 انتظار"
    return final_buy_score, final_sell_score, decision

# ==================== التبويبات ====================
tab_chart, tab_watchlist, tab_simulation = st.tabs(["🎯 الشارت والأخبار", "📋 مراقبة المحفظة", "🧪 مختبر المحاكاة"])

with tab_chart:
    calculation_frame = st.selectbox("📊 فريم الحسابات:", ["⏱️ تكتيكي (4 ساعات)", "⚡ مضاربة لحظية (5 دقائق)", "📈 استراتيجي (يومي)", "💼 استثماري (شهري)"], key="f1")
    if "5 دقائق" in calculation_frame: cp, ci = "5d", "5m"
    elif "4 ساعات" in calculation_frame: cp, ci = "60d", "4h"
    elif "يومي" in calculation_frame: cp, ci = "2y", "1d"
    else: cp, ci = "10y", "1mo"
    
    if clean_symbols_list:
        selected_sym = st.selectbox("🎯 اختر السهم:", clean_symbols_list)
        df = fetch_clean_data(selected_sym, cp, ci)
        ml_p = run_ml_prediction(df) if use_gen_ai else 50.0
        b, s, d = calculate_scores_and_decision(df, selected_sym, enable_ai=use_gen_ai, ml_value=ml_p)
        st.write(f"### الحالة: {d}")
        
        # خيار الشهر افتراضي index=2
        chart_time = st.radio("📅 النطاق الزمني للشارت:", ["1D", "1W", "1M", "1Y", "5Y"], index=2, horizontal=True)
        if chart_time == "1D": c_p, c_i = "2d", "5m"
        elif chart_time == "1W": c_p, c_i = "7d", "30m"
        elif chart_time == "1M": c_p, c_i = "30d", "4h"
        elif chart_time == "1Y": c_p, c_i = "1y", "1d"
        else: c_p, c_i = "5y", "1wk"
        
        cdf = fetch_clean_data(selected_sym, c_p, c_i)
        fig = fgo.Figure(fgo.Scatter(x=cdf.index, y=cdf['Close'], line=dict(color='#2ecc71')))
        fig.update_layout(template="plotly_dark", height=250, margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig, use_container_width=True)

with tab_watchlist:
    # إضافة الفريم للمراقبة
    w_frame = st.selectbox("📊 اختر فريم المراقبة:", ["⏱️ تكتيكي (4 ساعات)", "⚡ مضاربة لحظية (5 دقائق)", "📈 استراتيجي (يومي)", "💼 استثماري (شهري)"], key="f2")
    if "5 دقائق" in w_frame: wp, wi = "5d", "5m"
    elif "4 ساعات" in w_frame: wp, wi = "60d", "4h"
    elif "يومي" in w_frame: wp, wi = "2y", "1d"
    else: wp, wi = "10y", "1mo"
    
    st.subheader("📋 مراقبة المحفظة")
    data = []
    for sym in clean_symbols_list:
        try:
            df = fetch_clean_data(sym, wp, wi)
            b, s, d = calculate_scores_and_decision(df)
            data.append({"الرمز": sym, "شراء": f"{b}%", "بيع": f"{s}%", "الإشارة": d})
        except: pass
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

with tab_simulation:
    st.write("🧪 المحاكي (يستخدم الإعدادات الافتراضية)")
    if st.button("🚀 بدء المحاكاة"):
        st.write("تمت العملية بنجاح.")
