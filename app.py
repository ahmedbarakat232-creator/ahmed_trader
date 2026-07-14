import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as fgo
from datetime import datetime, timedelta
import json
import os
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- نظام منع تكرار الإشعارات (ساعة واحدة) ---
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = {}

def check_and_alert(symbol, decision):
    now = datetime.now()
    last_time = st.session_state.last_alert.get(symbol)
    if last_time is None or (now - last_time) > timedelta(hours=1):
        st.session_state.last_alert[symbol] = now
        return True
    return False

# إعداد الصفحة
st_autorefresh(interval=30000, key="refresh_v143")
st.set_page_config(page_title="منصة AI v14.3", layout="centered", initial_sidebar_state="collapsed")

# --- نظام الإعدادات والبيانات ---
DB_FILE = "watchlist_db.json"
def load_settings():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {"watchlist": "NVDA,TSLA,AAPL,GC=F", "phone_number": "", "notifications_active": False}

settings = load_settings()

with st.expander("⚙️ إعدادات المحفظة والاتصال والإشعارات"):
    watchlist_input = st.text_area("الرموز:", value=settings.get("watchlist", "NVDA,TSLA,AAPL,GC=F"))
    phone_input = st.text_input("رقم الجوال:", value=settings.get("phone_number", ""))
    notif_active = st.checkbox("🔔 تفعيل الإشعارات", value=settings.get("notifications_active", False))
    use_gen_ai = st.checkbox("🔥 تفعيل الذكاء الاصطناعي", value=True)

symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

# --- دالة جلب البيانات (مع تصحيح الخطأ) ---
@st.cache_data(ttl=5)
def fetch_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
        if 'Close' not in data.columns and 'Adj Close' in data.columns: data['Close'] = data['Adj Close']
        
        # حساب المؤشرات
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['RSI_14'] = 100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(14).mean() / -data['Close'].diff().clip(upper=0).rolling(14).mean())))
        return data
    except Exception: return pd.DataFrame()

# دالة الحساب والقرار
def calculate_scores_and_decision(df, enable_ai):
    if df.empty or len(df) < 20: return 50, 50, "🟡 انتظار"
    return 60, 40, "🟢 شراء قوي" # (ضع منطقك هنا)

# --- الواجهة ---
tab1, tab2, tab3 = st.tabs(["🎯 الشارت", "📋 المحفظة", "🧪 مختبر الدقة"])

with tab1:
    calc_frame = st.selectbox("📊 الفريم:", ["⏱️ 4 ساعات", "⚡ 5 دقائق"], index=0)
    per, inter = ("60d", "4h") if "4 ساعات" in calc_frame else ("5d", "5m")

with tab2:
    st.subheader("📋 حالة الأسهم")
    results = []
    for sym in symbols:
        df = fetch_data(sym, per, inter)
        _, _, d = calculate_scores_and_decision(df, use_gen_ai)
        if ("شراء" in d or "بيع" in d) and notif_active:
            if check_and_alert(sym, d):
                st.sidebar.warning(f"🔔 {sym}: {d}")
        results.append({"الرمز": sym, "الإشارة": d})
    st.table(pd.DataFrame(results))

with tab3:
    st.subheader("🧪 قياس دقة الإشارات")
    if st.button("🚀 فحص الدقة"):
        acc_results = []
        for sym in symbols:
            df = fetch_data(sym, "10d", "1d")
            if len(df) > 5:
                _, _, d_past = calculate_scores_and_decision(df.iloc[:-3], use_gen_ai)
                price_past, price_now = float(df.iloc[-3]['Close']), float(df.iloc[-1]['Close'])
                res = "✅" if ("شراء" in d_past and price_now > price_past) or ("بيع" in d_past and price_now < price_past) else "❌"
                acc_results.append({"السهم": sym, "النتيجة": res})
        st.table(pd.DataFrame(acc_results))
