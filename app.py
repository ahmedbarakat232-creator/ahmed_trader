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

# تفعيل التحديث التلقائي المستمر كل 30 ثانية لتحديث الأسعار الفورية
st_autorefresh(interval=30000, key="mobile_refresh_v120")

# إعداد الصفحة لتناسب شاشات الجوال تماماً
st.set_page_config(page_title="منصة AI v12.0", layout="centered", initial_sidebar_state="collapsed")

# كود CSS مخصص لتجميل الألوان والأزرار على الجوال
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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦅 منظومة التداول v12.0 (الشارت المرن والمستشار التلقائي)")

# ==================== نظام حفظ الإعدادات تلقائياً ====================
DB_FILE = "watchlist_db.json"

def load_saved_watchlist():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                return data.get("watchlist", "NVDA,TSLA,AAPL,GC=F")
        except:
            return "NVDA,TSLA,AAPL,GC=F"
    return "NVDA,TSLA,AAPL,GC=F"

def save_watchlist(watchlist_str):
    try:
        with open(DB_FILE, "w") as f:
            json.dump({"watchlist": watchlist_str}, f)
    except:
        pass

saved_watchlist = load_saved_watchlist()

# ==================== إعدادات الواجهة الرئيسية لجوالك ====================
with st.expander("⚙️ إعدادات المحفظة والـ API"):
    watchlist_input = st.text_area("أدخل الرموز (مثل: NVDA,TSLA,GC=F):", value=saved_watchlist)
    if watchlist_input != saved_watchlist:
        save_watchlist(watchlist_input)
        st.success("💾 تم الحفظ والمزامنة!")
    
    symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
    
    API_KEY = st.text_input("مفتاح الـ Gemini API (مطلوب لمستشار الأخبار التلقائي):", type="password")
    use_gen_ai = st.checkbox("🔥 تفعيل مستشار الأخبار الذكي التلقائي (سحب مباشر بدون روابط)")

# ==================== احتساب المؤشرات الفنية للنموذج ====================
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

@st.cache_data(ttl=15)
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

# ==================== خوارزمية التقييم الاحترافي المنفصل لشاشات الجوال ====================
def calculate_scores_and_decision(df):
    if df.empty or len(df) < 20:
        return 50, 50, "🟡 وضع الانتظار (HOLD)"
    
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
    final_buy_score = max(0, min(100, int(buy_score)))
    
    # 2. حساب تقييم البيع (Sell Score)
    sell_score = 30
    if close_p < ema200_p: sell_score += 15
    if rsi_p > 60: sell_score += 20
    if mfi_p > 65: sell_score += 20
    if ema9_p < ema21_p: sell_score += 15
    if vol_now > vol_avg: sell_score += 10
    final_sell_score = max(0, min(100, int(sell_score)))
    
    # تحديد الإشارة
    if final_buy_score >= 80:
        decision = "🟢 شراء قوي (BUY)"
    elif final_sell_score >= 80:
        decision = "🔴 بيع قوي (SELL)"
    else:
        decision = "🟡 انتظار (HOLD)"
        
    return final_buy_score, final_sell_score, decision

# ==================== معالجة السهم الفردي المختار ====================
st.write("---")
clean_symbols_list = [str(s).upper() for s in symbols]
if clean_symbols_list:
    selected_sym = st.selectbox("🎯 اختر الأصل لتحليله بالشارت والأخبار التلقائية:", clean_symbols_list)

    if selected_sym:
        target_clean = str(selected_sym).strip().upper()
        
        # أزرار اختيار المدة الزمنية للشارت بمرونة فائقة
        time_frame_choice = st.radio(
            "📅 اختر النطاق الزمني للشارت:",
            ["يوم واحد (1D)", "أسبوع واحد (1W)", "شهر واحد (1M)", "سنة واحدة (1Y)", "5 سنوات (5Y)"],
            horizontal=True
        )
        
        if "1D" in time_frame_choice:
            p_period, p_interval = "2d", "5m"
        elif "1W" in time_frame_choice:
            p_period, p_interval = "7d", "30m"
        elif "1M" in time_frame_choice:
            p_period, p_interval = "30d", "4h"
        elif "1Y" in time_frame_choice:
            p_period, p_interval = "1y", "1d"
        else:
            p_period, p_interval = "5y", "1wk"
            
        df = fetch_clean_data(target_clean, p_period, p_interval)
        
        if not df.empty and len(df) >= 5:
            latest = df.iloc[-1]
            close_p = float(latest['Close'])
            mfi_p = float(latest['MFI_14']) if 'MFI_14' in latest else 50
            atr_p = float(latest['ATR']) if 'ATR' in latest else (close_p * 0.02)
            
            # حساب الأرقام والقرارات الفورية
            f_buy, f_sell, f_decision = calculate_scores_and_decision(df)
            
            decision_color = "#2ecc71" if "BUY" in f_decision else ("#e74c3c" if "SELL" in f_decision else "#f1c40f")
            bg_decision = "rgba(46, 204, 113, 0.15)" if "BUY" in f_decision else ("rgba(231, 76, 60, 0.15)" if "SELL" in f_decision else "rgba(241, 196, 15, 0.12)")

            # عرض التقييمات الثنائية والقرار الفوري على الجوال
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
                    <span style="color: #bdc3c7; font-size: 0.85rem;">🎯 الإشارة الفورية الحالية:</span>
                    <h3 style="margin: 3px 0 0 0; color: {decision_color}; font-weight: bold;">{f_decision}</h3>
                </div>
            """, unsafe_allow_html=True)

            # ==================== الشارت التفاعلي المطور جداً (Area Chart) ====================
            st.write("")
            fig = fgo.Figure()
            # رسم حركة السعر كـ Area Chart لتعبئة الفراغ بشكل احترافي
            fig.add_trace(fgo.Scatter(
                x=df.index, 
                y=df['Close'], 
                mode='lines', 
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.05)' if "BUY" in f_decision else 'rgba(231, 76, 60, 0.05)',
                line=dict(color=decision_color, width=2.5), 
                name='السعر'
            ))
            
            fig.update_layout(
                height=280, 
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(showgrid=False, color="white", tickformat='%m-%d' if "Y" in time_frame_choice else '%H:%M'),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="white", side="right"),
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ==================== المستشار التلقائي للأخبار الذكي (بدون روابط يدوية) ====================
            if use_gen_ai:
                if not API_KEY:
                    st.warning("⚠️ يرجى إدخال مفتاح الـ Gemini API أولاً في الإعدادات بالأعلى لتفعيل هذا الخيار.")
                else:
                    st.write("---")
                    st.subheader("📰 مستشار الذكاء الاصطناعي والأخبار (تلقائي الحلب)")
                    with st.spinner("جاري سحب أحدث الأخبار وتلخيص نبرتها الاقتصادية والجيوسياسية..."):
                        try:
                            ticker_obj = yf.Ticker(target_clean)
                            raw_news = ticker_obj.news
                            
                            news_context = ""
                            if raw_news:
                                for idx, article in enumerate(raw_news[:3]):
                                    title = article.get('title', 'عنوان غير معروف')
                                    link = article.get('link', '#')
                                    publisher = article.get('publisher', 'مصدر مالي')
                                    news_context += f"- العنوان: {title} (الناشر: {publisher})\n"
                                    # عرض الخبر بصفته رابطاً تفاعلياً تلقائياً في الواجهة
                                    st.markdown(f"🔗 [{title}]({link})")
                            else:
                                news_context = "لا توجد أخبار اقتصادية عاجلة متداولة حالياً لهذا الرمز في ياهو فاينانس."
                                st.info("ℹ️ لا توجد مقالات إخبارية حيوية منشورة في الساعات الأخيرة.")
                            
                            if news_context:
                                prompt = (
                                    f"أنت مستشار مالي وخبير أسواق. قم بتحليل الأصل {target_clean} "
                                    f"بناءً على التقييمين الحاليين (قوة الشراء: {f_buy}%، قوة البيع: {f_sell}%) "
                                    f"وهذه الأخبار الاقتصادية التي جلبناها تلقائياً:\n{news_context}\n\n"
                                    f"اكتب تلخيصاً دقيقاً في 3 نقاط محددة وسريعة:\n"
                                    f"1. هل الأخبار تضغط إيجابياً أم سلبياً حالياً؟\n"
                                    f"2. نصيحة تداول ذكية ومحددة للمتداول عبر الجوال."
                                )
                                
                                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
                                payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')
                                req = urllib.request.Request(api_url, data=payload, headers={'Content-Type': 'application/json'})
                                with urllib.request.urlopen(req) as response:
                                    res_data = json.loads(response.read().decode('utf-8'))
                                    ai_response = res_data['candidates'][0]['content']['parts'][0]['text']
                                    st.markdown(f"""
                                        <div style="background-color: #1a252f; padding: 12px; border-radius: 8px; margin-top: 10px; border-right: 4px solid #f1c40f;">
                                            <span style="font-size: 0.85rem; color: #bdc3c7;">🔮 تحليل الروبوت الاستثماري للأخبار:</span><br>
                                            <span style="font-size: 0.9rem;">{ai_response}</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error("حدث خطأ في عملية الاتصال بمزود الذكاء الاصطناعي.")

# ==================== لوحة المراقبة الشاملة لجميع الأسهم المضافة (شاشتك المطلوبة) ====================
st.write("---")
st.subheader("📋 لوحة المراقبة الشاملة للمحفظة")

if clean_symbols_list:
    watchlist_data = []
    
    with st.spinner("جاري تحديث لوحة المراقبة لجميع الأسهم..."):
        for sym in clean_symbols_list:
            sym_clean = str(sym).strip().upper()
            try:
                # سحب بيانات سريعة للفريم الحالي لكل سهم لحساب إشارته
                sym_df = fetch_clean_data(sym_clean, "30d", "4h")
                if not sym_df.empty and len(sym_df) >= 10:
                    last_row = sym_df.iloc[-1]
                    price_now = float(last_row['Close'])
                    
                    buy_val, sell_val, decision_val = calculate_scores_and_decision(sym_df)
                    
                    watchlist_data.append({
                        "الرمز": sym_clean,
                        "السعر الحالي": f"${price_now:.2f}",
                        "قوة الشراء": f"{buy_val}%",
                        "قوة البيع": f"{sell_val}%",
                        "الإشارة الاستراتيجية": decision_val
                    })
                else:
                    watchlist_data.append({
                        "الرمز": sym_clean,
                        "السعر الحالي": "N/A",
                        "قوة الشراء": "-",
                        "قوة البيع": "-",
                        "الإشارة الاستراتيجية": "⚠️ بيانات غير كافية"
                    })
            except:
                pass
                
    if watchlist_data:
        # تحويل البيانات إلى جدول جميل ومنظم
        summary_df = pd.DataFrame(watchlist_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("أدخل الرموز في الإعدادات بالأعلى لملء لوحة المراقبة.")
