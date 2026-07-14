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
st_autorefresh(interval=30000, key="mobile_refresh_v110")

# إعداد الصفحة لتناسب شاشات الجوال تماماً
st.set_page_config(page_title="منصة AI v11.0", layout="centered", initial_sidebar_state="collapsed")

# كود CSS مخصص لتظبيط أحجام العناصر والخطوط لتناسب شاشات الموبايل تماماً
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
    </style>
""", unsafe_allow_value=True)

st.title("🦅 منظومة التداول v11.0 (التقييم المنفصل للجوال)")

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
    except Exception as e:
        pass

saved_watchlist = load_saved_watchlist()

# ==================== إعدادات مبسطة في الواجهة الرئيسية لجوالك ====================
with st.expander("⚙️ إعدادات المحفظة والـ API"):
    watchlist_input = st.text_area("أدخل الرموز (مثل: NVDA,TSLA,GC=F):", value=saved_watchlist)
    if watchlist_input != saved_watchlist:
        save_watchlist(watchlist_input)
        st.success("💾 تم الحفظ والمزامنة!")
    
    symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
    
    API_KEY = st.text_input("مفتاح الـ Gemini API (اختياري):", type="password")
    use_gen_ai = st.checkbox("تفعيل مستشار الأخبار بالذكاء الاصطناعي")

# اختيار نمط التداول والفريم الزمني للجوال ليكون سريعاً وسهلاً باللمس
trading_style = st.selectbox(
    "اختر فريم العمل:",
    [
        "⚡ مضاربة متطورة (4 ساعات)",
        "🔥 مضاربة سريعة (5 دقائق)",
        "📈 مضاربة متوسطة (يومي)",
        "💼 استثمار طويل (أسبوعي)"
    ]
)

if "🔥" in trading_style:
    p_period, p_interval = "5d", "5m"
elif "⚡" in trading_style:
    p_period, p_interval = "60d", "4h"
elif "📈" in trading_style:
    p_period, p_interval = "2y", "1d"
else:
    p_period, p_interval = "5y", "1wk"

# ==================== احتساب المؤشرات الفنية للنموذج ====================
def calculate_indicators(df):
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # 1. Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BBU_20'] = df['MA20'] + (df['STD20'] * 2)
        df['BBL_20'] = df['MA20'] - (df['STD20'] * 2)
        
        # 2. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 3. EMAs
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # 4. MFI (Money Flow Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        direction = typical_price.diff()
        pos_flow = raw_money_flow.where(direction > 0, 0.0)
        neg_flow = raw_money_flow.where(direction < 0, 0.0)
        pos_mf14 = pos_flow.rolling(window=14).sum()
        neg_mf14 = neg_flow.rolling(window=14).sum()
        m_ratio = pos_mf14 / (neg_mf14 + 1e-10)
        df['MFI_14'] = 100 - (100 / (1 + m_ratio))
        
        # 5. ATR
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

# ==================== معالجة السهم المختار ====================
st.write("---")
clean_symbols_list = [str(s).upper() for s in symbols]
if clean_symbols_list:
    selected_sym = st.selectbox("اختر الرمز المراد تحليله:", clean_symbols_list)

    if selected_sym:
        target_clean = str(selected_sym).strip().upper()
        df = fetch_clean_data(target_clean, p_period, p_interval)
        
        if not df.empty and len(df) >= 20:
            latest = df.iloc[-1]
            close_p = float(latest['Close'])
            bbl_p = float(latest['BBL_20']) if 'BBL_20' in latest else close_p
            bbu_p = float(latest['BBU_20']) if 'BBU_20' in latest else close_p
            rsi_p = float(latest['RSI_14']) if 'RSI_14' in latest else 50
            mfi_p = float(latest['MFI_14']) if 'MFI_14' in latest else 50
            ema200_p = float(latest['EMA200']) if 'EMA200' in latest else close_p
            ema9_p = float(latest['EMA9']) if 'EMA9' in latest else close_p
            ema21_p = float(latest['EMA21']) if 'EMA21' in latest else close_p
            atr_p = float(latest['ATR']) if 'ATR' in latest else (close_p * 0.02)
            vol_now = float(latest['Volume']) if 'Volume' in latest else 0
            vol_avg = float(latest['Vol_MA20']) if 'Vol_MA20' in latest else 1
            
            ml_prob = run_ml_prediction(df)
            
            # ==================== 🧮 خوارزمية التقييم الثنائي المنفصل (من 100) ====================
            
            # 1. حساب تقييم الشراء (Buy Score):
            buy_score = 30  # الأساس الافتراضي المعتدل للشراء
            
            if close_p > ema200_p: buy_score += 15       # تريند صاعد يدعم الشراء
            if rsi_p < 40: buy_score += 20               # مؤشر الزخم في مناطق قريبة من القاع
            if mfi_p < 35: buy_score += 20               # تجميع سيولة ذكي
            if ema9_p > ema21_p: buy_score += 15         # تقاطع زخم صاعد سريع
            if vol_now > vol_avg: buy_score += 10         # حجم تداول يدعم الصعود
            buy_score += (ml_prob - 50) * 0.4            # إضافة وزن توقع الذكاء الاصطناعي
            
            final_buy_score = max(0, min(100, int(buy_score)))
            
            # 2. حساب تقييم البيع (Sell Score):
            sell_score = 30  # الأساس الافتراضي المعتدل للبيع
            
            if close_p < ema200_p: sell_score += 15      # تريند هابط يدعم الخروج والبيع
            if rsi_p > 60: sell_score += 20              # مؤشر الزخم متشبع بالقمة
            if mfi_p > 65: sell_score += 20              # تصريف سيولة عند القمة
            if ema9_p < ema21_p: sell_score += 15         # تقاطع زخم هابط سريع
            if vol_now > vol_avg: sell_score += 10         # حجم تداول عالي يدعم الهبوط
            sell_score += (50 - ml_prob) * 0.4           # قوة توقع الهبوط من الذكاء الاصطناعي
            
            final_sell_score = max(0, min(100, int(sell_score)))

            # ==================== 🚦 نظام الإشارة الحاسم الفوري ====================
            if final_buy_score >= 80:
                final_decision = "🟢 إشارة شراء فورية (BUY)"
                decision_color = "#2ecc71"
                bg_decision = "rgba(46, 204, 113, 0.2)"
            elif final_sell_score >= 80:
                final_decision = "🔴 إشارة بيع فورية (SELL)"
                decision_color = "#e74c3c"
                bg_decision = "rgba(231, 76, 60, 0.2)"
            else:
                final_decision = "🟡 وضع الانتظار والمراقبة (HOLD)"
                decision_color = "#f1c40f"
                bg_decision = "rgba(241, 196, 15, 0.15)"

            # عرض التقييمين المنفصلين بوضوح في شاشتين متجاورتين مريحة للجوال
            st.markdown("### 📊 التقييمات الثنائية للعملية:")
            col_b, col_s = st.columns(2)
            with col_b:
                st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.1); border-top: 5px solid #2ecc71; padding: 12px; border-radius: 8px; text-align: center;">
                        <span style="color: #bdc3c7; font-size: 0.85rem;">🔥 قوة الشراء</span><br>
                        <strong style="color: #2ecc71; font-size: 1.5rem;">{final_buy_score} / 100</strong>
                    </div>
                """, unsafe_allow_value=True)
            with col_s:
                st.markdown(f"""
                    <div style="background-color: rgba(231, 76, 60, 0.1); border-top: 5px solid #e74c3c; padding: 12px; border-radius: 8px; text-align: center;">
                        <span style="color: #bdc3c7; font-size: 0.85rem;">📉 قوة البيع</span><br>
                        <strong style="color: #e74c3c; font-size: 1.5rem;">{final_sell_score} / 100</strong>
                    </div>
                """, unsafe_allow_value=True)

            # عرض القرار النهائي الكبير والمريح للعين باللمس
            st.markdown(f"""
                <div style="background-color: {bg_decision}; border: 1.5px solid {decision_color}; padding: 15px; border-radius: 8px; margin-top: 15px; text-align: center;">
                    <span style="color: #bdc3c7; font-size: 0.9rem;">📢 الإشارة الفورية الحالية:</span>
                    <h2 style="margin: 5px 0 0 0; color: {decision_color}; font-weight: bold;">{final_decision}</h2>
                </div>
            """, unsafe_allow_value=True)
            
            # عرض بيانات سريعة (Metrics)
            st.write("")
            col_p, col_v = st.columns(2)
            with col_p:
                st.metric("السعر الحالي", f"${close_p:.2f}")
            with col_v:
                st.metric("حجم السيولة MFI", f"{mfi_p:.1f}%")

            # ==================== شارت اتجاه السعر البسيط جداً ====================
            st.write("---")
            st.subheader("📈 مسار حركة السعر بالنسبة للزمن")
            
            fig = fgo.Figure()
            fig.add_trace(fgo.Scatter(
                x=df.index, 
                y=df['Close'], 
                mode='lines', 
                line=dict(color='#2ecc71' if close_p > ema200_p else '#e74c3c', width=2), 
                name='السعر'
            ))
            fig.add_trace(fgo.Scatter(
                x=df.index, 
                y=df['EMA200'], 
                mode='lines', 
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'), 
                name='EMA 200'
            ))
            
            fig.update_layout(
                height=300, 
                margin=dict(l=10, r=10, t=5, b=5),
                xaxis=dict(showgrid=False, color="white"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="white"),
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # مستويات وقف الخسارة والأهداف التكيفية المحسوبة
            if final_buy_score >= 80 or final_sell_score >= 80:
                stop_loss = close_p - (atr_p * 1.5) if final_buy_score >= 80 else close_p + (atr_p * 1.5)
                target_1 = close_p + (atr_p * 1.5) if final_buy_score >= 80 else close_p - (atr_p * 1.5)
                
                st.markdown(f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;">
                        <span style="color: #bdc3c7; font-size: 0.8rem;">🎯 أهداف الـ ATR المقترحة:</span><br>
                        <strong style="color: #e74c3c; font-size: 0.9rem;">وقف خسارة: ${stop_loss:.2f}</strong> | 
                        <strong style="color: #2ecc71; font-size: 0.9rem;">هدف: ${target_1:.2f}</strong>
                    </div>
                """, unsafe_allow_value=True)

            # ==================== تحليل الأخبار بالذكاء الاصطناعي ====================
            if use_gen_ai and API_KEY:
                st.write("---")
                with st.expander("📰 تحليل سريع للأخبار والمشاعر"):
                    with St.spinner("جاري جمع البيانات الاقتصادية..."):
                        try:
                            ticker_obj = yf.Ticker(target_clean)
                            news_list = ticker_obj.news[:3]
                            news_context = ""
                            for i, news in enumerate(news_list):
                                news_context += f"{i+1}. العنوان: {news.get('title')}\n"
                            
                            if not news_context:
                                news_context = "لا توجد أخبار اقتصادية عاجلة متداولة حالياً."
                                
                            prompt = (
                                f"أنت مستشار مالي محترف. قم بتحليل السهم أو الأصل {target_clean} "
                                f"بناءً على التقييمين الحاليين (قوة الشراء: {final_buy_score}%، قوة البيع: {final_sell_score}%) "
                                f"وهذه الأخبار العاجلة:\n{news_context}\n\n"
                                f"اعطني في 3 سطور قصيرة جداً:\n"
                                f"1. نبرة ومشاعر الأخبار العاجلة.\n"
                                f"2. توصيتك السريعة والمباشرة لمستخدم يتصفح من هاتفه المحمول."
                            )
                            
                            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
                            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')
                            req = urllib.request.Request(api_url, data=payload, headers={'Content-Type': 'application/json'})
                            with urllib.request.urlopen(req) as response:
                                res_data = json.loads(response.read().decode('utf-8'))
                                ai_response = res_data['candidates'][0]['content']['parts'][0]['text']
                                st.markdown(ai_response)
                        except:
                            st.error("فشل جلب الأخبار، تأكد من الـ API Key وصلاحية الاتصال.")
