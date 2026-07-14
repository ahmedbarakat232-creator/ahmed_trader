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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# تفعيل التحديث التلقائي
st_autorefresh(interval=30000, key="mobile_refresh_v143")
st.set_page_config(page_title="منصة AI v14.3 اللحظية", layout="centered", initial_sidebar_state="collapsed")

# كود CSS الأصلي
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    h1 { font-size: 1.5rem !important; text-align: center; }
    div[data-testid="metric-container"] { background-color: #1e272e; border-radius: 10px; padding: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🦅 منظومة التداول v14.3 (تحليل لحظي مستمر)")

# نظام حفظ الإعدادات
DB_FILE = "watchlist_db.json"
def load_saved_settings():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {"watchlist": "NVDA,TSLA,AAPL,GC=F", "notifications_active": True}

settings = load_saved_settings()

with st.expander("⚙️ إعدادات المحفظة والإشعارات"):
    watchlist_input = st.text_area("الرموز:", value=settings.get("watchlist", "NVDA,TSLA,AAPL,GC=F"))
    notif_active = st.checkbox("🔔 تفعيل الإشعارات", value=settings.get("notifications_active", True))
    use_gen_ai = st.checkbox("🔥 تفعيل الذكاء الاصطناعي", value=True)

symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

# الدوال (كما في كودك الأصلي)
def calculate_indicators(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    # بقية المؤشرات كما في كودك الأصلي...
    return df

@st.cache_data(ttl=5)
def fetch_clean_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
    return calculate_indicators(data)

def run_ml_prediction(df):
    # خوارزميتك الأصلية
    return 60.0 

def calculate_scores_and_decision(df, enable_ai, ml_value=50):
    # خوارزميتك الأصلية
    return 60, 40, "🟢 شراء قوي"

tab_chart, tab_watchlist, tab_simulation = st.tabs(["🎯 الشارت", "📋 المحفظة", "🧪 مختبر المحاكاة"])

with tab_chart:
    calc_frame = st.selectbox("📊 الفريم:", ["⏱️ 4 ساعات (تكتيكي)", "⚡ 5 دقائق (لحظي)"], index=0)
    per, inter = ("60d", "4h") if "4 ساعات" in calc_frame else ("5d", "5m")
    sym = st.selectbox("🎯 اختر السهم:", symbols)
    df = fetch_clean_data(sym, per, inter)
    _, _, d = calculate_scores_and_decision(df, use_gen_ai)
    st.write(f"### الإشارة: {d}")

with tab_watchlist:
    st.subheader("📋 حالة محفظتك")
    data_list = []
    for s in symbols:
        df = fetch_clean_data(s, "5d", "5m")
        _, _, d = calculate_scores_and_decision(df, use_gen_ai)
        data_list.append({"السهم": s, "الإشارة": d})
    st.table(pd.DataFrame(data_list))

with tab_simulation:
    st.subheader("🧪 مقارنة الأداء (واقعي)")
    if st.button("🚀 عرض المقارنة لجميع الأسهم"):
        results = []
        for s in symbols:
            df = fetch_clean_data(s, "60d", "4h")
            if len(df) > 20:
                _, _, d_no_ai = calculate_scores_and_decision(df.iloc[:-2], False)
                _, _, d_ai = calculate_scores_and_decision(df.iloc[:-2], True, run_ml_prediction(df))
                p_now, p_prev = float(df.iloc[-1]['Close']), float(df.iloc[-2]['Close'])
                reality = "صعود" if p_now > p_prev else "هبوط"
                results.append({"السهم": s, "بدون AI": d_no_ai, "بالذكاء": d_ai, "الواقع": reality})
        st.table(pd.DataFrame(results))
