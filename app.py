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

# كود CSS لتجميل العناصر على شاشات الجوال والتطابق التام
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    h1 {
        font-size: 1.5rem !important;
        text-align: center;
    }
    h2 {
        font-size: 1.2rem !important;
    }
    h3 {
        font-size: 1.0rem !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e272e;
        border-radius: 10px;
        padding: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem !important;
        padding: 8px 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦅 منظومة التداول v14.3 (تحليل لحظي مستمر)")

# ==================== نظام حفظ الإعدادات تلقائياً ====================
DB_FILE = "watchlist_db.json"

def load_saved_settings():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "watchlist": "NVDA,TSLA,AAPL,GC=F",
        "phone_number": "",
        "notifications_active": False
    }

def save_settings(settings_dict):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(settings_dict, f)
    except:
        pass

settings = load_saved_settings()

# ==================== إعدادات المحفظة والـ API ====================
with st.expander("⚙️ إعدادات المحفظة والاتصال والإشعارات"):
    watchlist_input = st.text_area("أدخل الرموز لقائمة مراقبتك (مثل: NVDA,TSLA,GC=F):", value=settings.get("watchlist", "NVDA,TSLA,AAPL,GC=F"))
    phone_input = st.text_input("📱 رقم الجوال مع رمز الدولة (مثال: 9665xxxxxxxx):", value=settings.get("phone_number", ""))
    notif_active = st.checkbox("🔔 تشغيل الإشعارات الفورية عند الإشارات القوية", value=settings.get("notifications_active", False))
    
    if (watchlist_input != settings.get("watchlist") or 
        phone_input != settings.get("phone_number") or 
        notif_active != settings.get("notifications_active")):
        settings["watchlist"] = watchlist_input
        settings["phone_number"] = phone_input
        settings["notifications_active"] = notif_active
        save_settings(settings)
        st.success("💾 تم حفظ الإعدادات بنجاح!")
    
    API_KEY = st.text_input("مفتاح الـ Gemini API (اختياري للأخبار والمحاكاة الذكية):", type="password")
    use_gen_ai = st.checkbox("🔥 تفعيل مستشار الأخبار الذكي التلقائي")

# معالجة قائمة الرموز
symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
clean_symbols_list = [str(s).upper() for s in symbols]

# ==================== احتساب المؤشرات الفنية للنموذج (لحظي) ====================
def calculate_indicators(df):
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BBU_20'] = df['MA20'] + (df['STD20'] * 2)
        df['BBL_20'] = df['MA20'] - (df['STD20'] * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # MFI
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        direction = typical_price.diff()
        pos_flow = raw_money_flow.where(direction > 0, 0.0)
        neg_flow = raw_money_flow.where(direction < 0, 0.0)
        pos_mf14 = pos_flow.rolling(window=14).sum()
        neg_mf14 = neg_flow.rolling(window=14).sum()
        m_ratio = pos_mf14 / (neg_mf14 + 1e-10)
        df['MFI_14'] = 100 - (100 / (1 + m_ratio))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        return df
    except:
        return df

# تم خفض الـ TTL إلى 5 ثوانٍ فقط لضمان تحديث البيانات اللحظية والأسعار بشكل مباشر وفوري
@st.cache_data(ttl=5)
def fetch_clean_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, group_by='ticker')
        if data.empty:
            data = yf.download(symbol, period="60d", interval="4h", progress=False, group_by='ticker')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)
        return calculate_indicators(data)
    except:
        return pd.DataFrame()

# ==================== خوارزمية التعلم الآلي للتنبؤ ====================
def run_ml_prediction(df):
    try:
        ml_df = df[['Close', 'RSI_14', 'MFI_14', 'ATR']].dropna().copy()
        if len(ml_df) < 30:
            return 50.0
        
        ml_df['Target'] = (ml_df['Close'].shift(-3) > ml_df['Close']).astype(int)
        features = ['RSI_14', 'MFI_14', 'ATR']
        X = ml_df[features].iloc[:-3]
        y = ml_df['Target'].iloc[:-3]
        
        if len(X) < 10 or len(np.unique(y)) < 2:
             return 50.0
             
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X_scaled, y)
        
        latest_features = np.array([ml_df[features].iloc[-1]])
        latest_scaled = scaler.transform(latest_features)
        prediction_prob = model.predict(latest_scaled)[0] * 100
        return prediction_prob
    except:
        return 50.0

# ==================== خوارزمية التقييم الثنائي المرنة ====================
def calculate_scores_and_decision(df, symbol="", enable_ai=True, ml_value=50.0):
    if df.empty or len(df) < 20:
        return 50, 50, "🟡 انتظار"
    
    latest = df.iloc[-1]
    close_p = float(latest['Close'])
    rsi_p = float(latest['RSI_14']) if 'RSI_14' in latest else 50
    mfi_p = float(latest['MFI_14']) if 'MFI_14' in latest else 50
    ema200_p = float(latest['EMA200']) if 'EMA200' in latest else close_p
    ema9_p = float(latest['EMA9']) if 'EMA9' in latest else close_p
    ema21_p = float(latest['EMA21']) if 'EMA21' in latest else close_p
    vol_now = float(latest['Volume']) if 'Volume' in latest else 0
    vol_avg = float(latest['Vol_MA20']) if 'Vol_MA20' in latest else 1
    
    # 1. حساب تقييم الشراء (Buy Score)
    buy_score = 30
    if close_p > ema200_p: buy_score += 15
    if rsi_p < 40: buy_score += 20
    if mfi_p < 35: buy_score += 20
    if ema9_p > ema21_p: buy_score += 15
    if vol_now > vol_avg: buy_score += 10
    
    if enable_ai:
        buy_score += (ml_value - 50) * 0.4
    
    final_buy_score = max(0, min(100, int(buy_score)))
    
    # 2. حساب تقييم البيع (Sell Score)
    sell_score = 30
    if close_p < ema200_p: sell_score += 15
    if rsi_p > 60: sell_score += 20
    if mfi_p > 65: sell_score += 20
    if ema9_p < ema21_p: sell_score += 15
    if vol_now > vol_avg: sell_score += 10
    
    if enable_ai:
        sell_score += (50 - ml_value) * 0.4
        
    final_sell_score = max(0, min(100, int(sell_score)))
    
    if final_buy_score >= 80:
        decision = "🟢 شراء قوي"
    elif final_sell_score >= 80:
        decision = "🔴 بيع قوي"
    else:
        decision = "🟡 انتظار"
        
    return final_buy_score, final_sell_score, decision

# ==================== تقسيم الواجهة إلى تبويبات مريحة للجوال ====================
tab_chart, tab_watchlist, tab_simulation = st.tabs(["🎯 الشارت والأخبار", "📋 مراقبة المحفظة", "🧪 مختبر المحاكاة الشامل"])

# ----------------- التبويب الأول: تحليل السهم المنفرد والشارت (لحظي تماماً) -----------------
with tab_chart:
    st.write("")
    calculation_frame = st.selectbox(
        "📊 فريم حسابات قوة البيع/الشراء والجدول السفلي:",
        ["⏱️ تكتيكي (4 ساعات)", "⚡ مضاربة لحظية (5 دقائق)", "📈 استراتيجي (يومي)", "💼 استثماري (شهري)"],
        key="main_calc_frame"
    )

    if "5 دقائق" in calculation_frame:
        calc_period, calc_interval = "5d", "5m"
    elif "4 ساعات" in calculation_frame:
        calc_period, calc_interval = "60d", "4h"
    elif "يومي" in calculation_frame:
        calc_period, calc_interval = "2y", "1d"
    else:
        calc_period, calc_interval = "10y", "1mo"

    if clean_symbols_list:
        selected_sym = st.selectbox("🎯 اختر السهم للتحليل وعرض الشارت الفعلي:", clean_symbols_list, key="main_sym")

        if selected_sym:
            target_clean = str(selected_sym).strip().upper()
            # استدعاء مباشر ولحظي للبيانات بدون أي حفظ مؤقت دائم
            calc_df = fetch_clean_data(target_clean, calc_period, calc_interval)
            
            # حساب التنبؤ بالذكاء الاصطناعي لحظياً
            ml_p = run_ml_prediction(calc_df) if use_gen_ai else 50.0
            f_buy, f_sell, f_decision = calculate_scores_and_decision(calc_df, target_clean, enable_ai=use_gen_ai, ml_value=ml_p)
            
            decision_color = "#2ecc71" if "شراء" in f_decision else ("#e74c3c" if "بيع" in f_decision else "#f1c40f")
            bg_decision = "rgba(46, 204, 113, 0.15)" if "شراء" in f_decision else ("rgba(231, 76, 60, 0.15)" if "بيع" in f_decision else "rgba(241, 196, 15, 0.12)")

            col_b, col_s = st.columns(2)
            with col_b:
                st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.08); border-top: 4px solid #2ecc71; padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="color: #bdc3c7; font-size: 0.8rem;">قوة الشراء</span><br>
                        <strong style="color: #2ecc71; font-size: 1.3rem;">{f_buy} / 100</strong>
                    </div>
                """, unsafe_allow_html=True)
            with col_s:
                st.markdown(f"""
                    <div style="background-color: rgba(231, 76, 60, 0.08); border-top: 4px solid #e74c3c; padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="color: #bdc3c7; font-size: 0.8rem;">قوة البيع</span><br>
                        <strong style="color: #e74c3c; font-size: 1.3rem;">{f_sell} / 100</strong>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color: {bg_decision}; border: 1px solid {decision_color}; padding: 12px; border-radius: 8px; margin-top: 10px; text-align: center;">
                    <span style="color: #bdc3c7; font-size: 0.85rem;">🎯 الإشارة الحالية (فريم القوة) - تتحدث كل 30ث:</span>
                    <h3 style="margin: 3px 0 0 0; color: {decision_color}; font-weight: bold;">{f_decision}</h3>
                </div>
            """, unsafe_allow_html=True)

            # شريط التحكم المستقل بالشارت
            st.write("")
            chart_time_choice = st.radio(
                "📅 اختر النطاق الزمني للشارت فقط:",
                ["يوم واحد (1D)", "أسبوع واحد (1W)", "شهر واحد (1M)", "سنة واحدة (1Y)", "5 سنوات (5Y)"],
                horizontal=True, key="chart_period_selector"
            )
            
            if "1D" in chart_time_choice:
                chart_period, chart_interval = "2d", "5m"
            elif "1W" in chart_time_choice:
                chart_period, chart_interval = "7d", "30m"
            elif "1M" in chart_time_choice:
                chart_period, chart_interval = "30d", "4h"
            elif "1Y" in chart_time_choice:
                chart_period, chart_interval = "1y", "1d"
            else:
                chart_period, chart_interval = "5y", "1wk"
                
            chart_df = fetch_clean_data(target_clean, chart_period, chart_interval)
            
            if not chart_df.empty:
                fig = fgo.Figure()
                fig.add_trace(fgo.Scatter(
                    x=chart_df.index, 
                    y=chart_df['Close'], 
                    mode='lines', 
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.05)',
                    line=dict(color='#2ecc71', width=2.5), 
                    name='السعر'
                ))
                
                fig.update_layout(
                    height=240, 
                    margin=dict(l=5, r=5, t=5, b=5),
                    xaxis=dict(showgrid=False, color="white", tickformat='%m-%d' if "Y" in chart_time_choice else '%H:%M'),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="white", side="right"),
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ----------------- التبويب الثاني: لوحة مراقبة المحفظة (لحظي تماماً) -----------------
with tab_watchlist:
    st.write("")
    st.subheader("📋 حالة محفظتك الحالية")
    st.caption(f"تحديث تلقائي مستمر ومباشر كل 30 ثانية للفريم: ({calculation_frame})")
    
    if clean_symbols_list:
        watchlist_data = []
        with st.spinner("جاري تحديث البيانات اللحظية..."):
            for sym in clean_symbols_list:
                sym_clean = str(sym).strip().upper()
                try:
                    # جلب لحظي للبيانات وتحديث القيم فورا
                    sym_df = fetch_clean_data(sym_clean, calc_period, calc_interval)
                    if not sym_df.empty and len(sym_df) >= 10:
                        last_row = sym_df.iloc[-1]
                        price_now = float(last_row['Close'])
                        ml_p = run_ml_prediction(sym_df) if use_gen_ai else 50.0
                        b_val, s_val, d_val = calculate_scores_and_decision(sym_df, sym_clean, enable_ai=use_gen_ai, ml_value=ml_p)
                        watchlist_data.append({
                            "الرمز": sym_clean, "السعر": f"${price_now:.2f}",
                            "قوة الشراء": f"{b_val}%", "قوة البيع": f"{s_val}%", "الإشارة": d_val
                        })
                except:
                    pass
        if watchlist_data:
            st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True, hide_index=True)

# ----------------- التبويب الثالث: مختبر المحاكاة الشامل (معزول لحل الاختفاء) -----------------
with tab_simulation:
    st.write("")
    st.subheader("🧪 محاكي التداول والاختبار العكسي لـ 10 أسهم")
    st.markdown("""
    يقوم المحاكي باختبار 10 أسهم قيادية ومقارنتها. لحل مشكلة اختفاء البيانات، يتم حفظ نتائج هذا التبويب بشكل منفصل تماماً دون التأثير على لحظية التبويبات الأخرى.
    """)
    
    simulation_stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "AMD", "BABA"]
    
    test_intervals = {
        "⏱️ 5 دقائق (لحظي)": ("5d", "5m"),
        "⚡ 4 ساعات (تكتيكي)": ("60d", "4h"),
        "📈 يومي (استراتيجي)": ("2y", "1d"),
        "💼 أسبوعي (استثماري)": ("5y", "1wk")
    }

    # العزل التام لجدول المحاكاة باستخدام session_state خاص به فقط
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None

    if st.button("🚀 بدء المحاكاة الشاملة الآن"):
        results_simulation = []
        progress_bar = st.progress(0)
        total_steps = len(simulation_stocks) * len(test_intervals)
        step = 0
        
        with st.spinner("جاري التحليل التاريخي..."):
            for stock in simulation_stocks:
                for int_name, (per, inter) in test_intervals.items():
                    step += 1
                    progress_bar.progress(step / total_steps)
                    
                    try:
                        sim_df = fetch_clean_data(stock, per, inter)
                        if sim_df.empty or len(sim_df) < 20:
                            continue
                            
                        last_close = float(sim_df.iloc[-1]['Close'])
                        
                        # 1. بدون تفعيل الذكاء الاصطناعي
                        b_no_ai, s_no_ai, d_no_ai = calculate_scores_and_decision(
                            sim_df, stock, enable_ai=False, ml_value=50.0
                        )
                        
                        # 2. مع تفعيل الذكاء الاصطناعي والتعلم الآلي
                        ml_val_sim = run_ml_prediction(sim_df)
                        b_with_ai, s_with_ai, d_with_ai = calculate_scores_and_decision(
                            sim_df, stock, enable_ai=True, ml_value=ml_val_sim
                        )
                        
                        results_simulation.append({
                            "السهم": stock,
                            "الفريم": int_name,
                            "السعر": f"${last_close:.2f}",
                            "شراء (بدون AI)": f"{b_no_ai}%",
                            "شراء (بالذكاء)": f"{b_with_ai}%",
                            "الإشارة (بدون AI)": d_no_ai,
                            "الإشارة (بالذكاء)": d_with_ai,
                        })
                    except:
                        pass
                        
        progress_bar.empty()
        
        if results_simulation:
            st.session_state.sim_results = pd.DataFrame(results_simulation)
            st.success("✅ تمت المحاكاة وحفظ النتائج في هذا التبويب بنجاح!")
        else:
            st.error("فشلت المحاكاة، يرجى التحقق من الاتصال بالإنترنت.")

    # عرض جدول المحاكاة المحفوظ بالكامل (دون التأثير على تحديث التبويبات الأخرى)
    if st.session_state.sim_results is not None:
        sim_result_df = st.session_state.sim_results
        st.markdown("### 📊 ملخص مقارنة الإشارات الفورية:")
        total_computed = len(sim_result_df)
        changes_detected = sim_result_df[sim_result_df["الإشارة (بدون AI)"] != sim_result_df["الإشارة (بالذكاء)"]]
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("إجمالي سيناريوهات الفحص", f"{total_computed} سيناريو")
        with col_stat2:
            st.metric("الإشارات المصححة بالذكاء الاصطناعي", f"{len(changes_detected)} إشارة")
            
        st.write("---")
        st.subheader("📋 تفاصيل جدول المحاكاة الشامل")
        st.dataframe(sim_result_df, use_container_width=True, hide_index=True)
