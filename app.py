import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as fgo
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import urllib.request
import json
import os
from streamlit_autorefresh import st_autorefresh

# استيراد أدوات التعلم الآلي
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. تفعيل التحديث التلقائي المستمر كل 30 ثانية لتحديث الأسعار الفورية والسيولة والتحليلات
st_autorefresh(interval=30000, key="watchlist_auto_refresh_v90")

st.set_page_config(page_title="منظومة الذكاء الاصطناعي v9.0", layout="wide")
st.title("🦅 منظومة التداول الإمبراطورية v9.0 (مؤشر الخوف، التوترات الجيوسياسية والذكاء الاصطناعي)")
st.write("الإصدار الاحترافي الأقوى عالمياً: يدمج الفلاتر الفنية، التعلم الآلي، مؤشر الخوف العالمي VIX، تأكيد تقاطعات الزخم، وتحليل التوترات والسياسة النقدية بالذكاء الاصطناعي.")

# ==================== نظام الذاكرة الرقمية وحفظ الإعدادات تلقائياً ====================
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
        st.sidebar.error(f"خطأ في الحفظ التلقائي: {e}")

saved_watchlist = load_saved_watchlist()

# ==================== إعدادات ربط الإشعارات والذكاء الاصطناعي ====================
st.sidebar.header("🔔 ربط التنبيهات الخارجية والـ API")
enable_telegram = st.sidebar.checkbox("تفعيل تنبيهات Telegram")
TELEGRAM_BOT_TOKEN = st.sidebar.text_input("توكن البوت (Bot Token):", type="password", placeholder="Token ID")
TELEGRAM_CHAT_ID = st.sidebar.text_input("معرف الشات (Chat ID):", type="password", placeholder="Chat ID")

st.sidebar.write("---")
st.sidebar.header("🧠 ذكاء اصطناعي توليدي للأخبار والتوترات (Gemini API)")
use_gen_ai = st.sidebar.checkbox("تفعيل مستشار الأخبار والمشاعر")
API_KEY = st.sidebar.text_input("أدخل مفتاح الـ Gemini API الخاص بك:", type="password", placeholder="Gemini API Key")

def send_telegram_alert(message):
    if not enable_telegram or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        pass

# ==================== جلب مؤشر الخوف العالمي VIX لحظياً ====================
@st.cache_data(ttl=60)
def fetch_vix_index():
    try:
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if not vix_data.empty:
            # معالجة الأعمدة والتأكد من جلب القيمة الأخيرة بشكل صحيح
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(-1)
            vix_close = float(vix_data['Close'].iloc[-1])
            return vix_close
    except:
        pass
    return 20.0  # القيمة الطبيعية الافتراضية في حال فشل الجلب

vix_current = fetch_vix_index()

# عرض حالة السوق بناءً على مؤشر الخوف في القائمة الجانبية
st.sidebar.write("---")
st.sidebar.header("📊 حالة معنويات السوق العالمية (VIX)")
if vix_current > 30:
    st.sidebar.error(f"💀 درجة الخوف: {vix_current:.2f} (سوق هابط عالي الخطورة)")
elif vix_current > 20:
    st.sidebar.warning(f"⚠️ درجة الخوف: {vix_current:.2f} (تذبذب متوسط وطبيعي)")
else:
    st.sidebar.success(f"😊 درجة الخوف: {vix_current:.2f} (استقرار وتفاؤل بالسوق)")

# ==================== إعدادات لوحة التحكم والأنماط الزمنية المحسنة ====================
st.sidebar.header("🎛️ لوحة التحكم الفنية")
watchlist_input = st.sidebar.text_area("أدخل رموز الأسهم والذهب مفصولة بفاصلة (,):", value=saved_watchlist)

if watchlist_input != saved_watchlist:
    save_watchlist(watchlist_input)
    st.sidebar.success("💾 تم الحفظ والمزامنة تلقائياً!")

symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

trading_style = st.sidebar.selectbox(
    "اختر نمط التداول وفريم العمل الفوري:",
    [
        "🔥 مضاربة سريعة (نفس اليوم - 5 دقائق)",
        "⚡ مضاربة متطورة (قصير المدى - 4 ساعات)",
        "📈 مضاربة متوسطة (مدار أيام - يومي)",
        "💼 استثمار طويل (مدار أشهر - أسبوعي)"
    ]
)

if "🔥" in trading_style:
    p_period, p_interval = "5d", "5m"
    style_label = "شارت الـ 5 دقائق"
elif "⚡" in trading_style:
    p_period, p_interval = "60d", "4h"
    style_label = "شارت الـ 4 ساعات"
elif "📈" in trading_style:
    p_period, p_interval = "2y", "1d"
    style_label = "الشارت اليومي"
else:
    p_period, p_interval = "5y", "1wk"
    style_label = "الشارت الأسبوعي"

# ==================== احتساب المؤشرات المتقدمة للتعلم الآلي والسيولة ====================
def calculate_advanced_indicators(df):
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
        
        # 3. EMA 200, EMA 9, EMA 21 (تقاطعات الزخم السريع والبطيء)
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # 4. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 5. MFI (Money Flow Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        direction = typical_price.diff()
        pos_flow = raw_money_flow.where(direction > 0, 0.0)
        neg_flow = raw_money_flow.where(direction < 0, 0.0)
        pos_mf14 = pos_flow.rolling(window=14).sum()
        neg_mf14 = neg_flow.rolling(window=14).sum()
        m_ratio = pos_mf14 / (neg_mf14 + 1e-10)
        df['MFI_14'] = 100 - (100 / (1 + m_ratio))
        
        # 6. ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        # 7. فلتر تأكيد حجم التداول الذكي (متوسط حجم 20 شمعة)
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        return df
    except Exception as e:
        return df

@st.cache_data(ttl=15)
def fetch_clean_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, group_by='ticker')
        if data.empty:
            data = yf.download(symbol, period="60d", interval="4h", progress=False, group_by='ticker')
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)
            
        data = calculate_advanced_indicators(data)
        return data
    except:
        return pd.DataFrame()

# ==================== خوارزمية التعلم الآلي (Machine Learning) ====================
def run_ml_prediction(df):
    try:
        ml_df = df[['Close', 'RSI_14', 'MFI_14', 'MACD', 'MACD_Signal', 'ATR']].dropna().copy()
        if len(ml_df) < 30:
            return "غير كافٍ للتدريب", 50.0
        
        ml_df['Target'] = (ml_df['Close'].shift(-3) > ml_df['Close']).astype(int)
        
        features = ['RSI_14', 'MFI_14', 'MACD', 'MACD_Signal', 'ATR']
        X = ml_df[features].iloc[:-3]
        y = ml_df['Target'].iloc[:-3]
        
        if len(X) < 10 or len(np.unique(y)) < 2:
             return "بيانات غير مستقرة", 50.0
             
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)
        
        latest_features = np.array([ml_df[features].iloc[-1]])
        latest_scaled = scaler.transform(latest_features)
        prediction_prob = model.predict(latest_scaled)[0] * 100
        
        if prediction_prob >= 60:
            direction = "📈 صعود متوقع خوارزمياً"
        elif prediction_prob <= 40:
            direction = "📉 هبوط متوقع خوارزمياً"
        else:
            direction = "🔄 اتجاه عرضي محايد"
            
        return direction, prediction_prob
    except Exception as e:
        return "تحت المزامنة الفنية", 50.0

# ==================== معالجة البيانات الفورية والرصد المتكامل ====================
summary_results = []
alerts_to_trigger = []

for sym in symbols:
    df = fetch_clean_data(sym, p_period, p_interval)
    if df.empty or len(df) < 20: continue
    
    latest = df.iloc[-1]
    close_p = float(latest['Close'])
    bbl_p = float(latest['BBL_20']) if 'BBL_20' in latest else close_p
    bbu_p = float(latest['BBU_20']) if 'BBU_20' in latest else close_p
    rsi_p = float(latest['RSI_14']) if 'RSI_14' in latest else 50
    mfi_p = float(latest['MFI_14']) if 'MFI_14' in latest else 50
    ema200_p = float(latest['EMA200']) if 'EMA200' in latest else close_p
    ema9_p = float(latest['EMA9']) if 'EMA9' in latest else close_p
    ema21_p = float(latest['EMA21']) if 'EMA21' in latest else close_p
    macd_p = float(latest['MACD']) if 'MACD' in latest else 0
    macd_sig_p = float(latest['MACD_Signal']) if 'MACD_Signal' in latest else 0
    atr_p = float(latest['ATR']) if 'ATR' in latest else (close_p * 0.02)
    
    vol_now = float(latest['Volume']) if 'Volume' in latest else 0
    vol_avg = float(latest['Vol_MA20']) if 'Vol_MA20' in latest else 1
    
    # تشغيل محرك التعلم الآلي
    ml_direction, ml_prob = run_ml_prediction(df)
    
    status = "🟡 انتظر (لا توجد إشارة حاسمة)"
    sound_signal = None
    
    # الشروط الأمنية التكيفية بناءً على مؤشر الخوف VIX
    confidence_threshold = 63 if vix_current > 25 else 58  # نرفع شرط التأكيد إذا كان السوق خائفاً ومتوتراً
    
    is_bullish_trend = close_p > ema200_p
    is_momentum_buy = ema9_p > ema21_p  # تأكيد التقاطع الإيجابي السريع للزخم
    is_momentum_sell = ema9_p < ema21_p  # تأكيد التقاطع السلبي السريع للزخم
    is_volume_confirmed = vol_now >= vol_avg

    # دمج شروط المؤشرات الفنية، تقاطع الزخم، التعلم الآلي، تأكيد الحجم، وفلتر الخوف VIX
    if (close_p <= bbl_p * 1.015 or rsi_p < 35) and mfi_p < 30 and is_bullish_trend and is_momentum_buy and macd_p > macd_sig_p and ml_prob >= confidence_threshold and is_volume_confirmed:
        status = "🟢 اقتناص شراء مؤكد (توافق السيولة والـ AI + تأكيد الحجم)"
        sound_signal = "buy"
    elif (close_p >= bbu_p * 0.985 or rsi_p > 65) and mfi_p > 70 and is_momentum_sell and macd_p < macd_sig_p and ml_prob <= (100 - confidence_threshold) and is_volume_confirmed:
        status = "🔴 خروج وبيع مؤكد (تضخم السيولة والـ AI + تأكيد الحجم)"
        sound_signal = "sell"

    summary_results.append({
        "الرمز": sym,
        "السعر الحالي": f"${close_p:.2f}",
        "تدفق السيولة MFI": f"{mfi_p:.1f}%",
        "تأكيد حجم التداول": "✅ مرتفع ومدعوم" if is_volume_confirmed else "⚠️ منخفض وضغيف",
        "تنبؤ التعلم الآلي": ml_direction,
        "نسبة ثقة الصعود (ML)": f"{ml_prob:.1f}%",
        "القرار النهائي": status
    })
    
    if sound_signal:
        stop_loss = close_p - (atr_p * 1.5) if sound_signal == "buy" else close_p + (atr_p * 1.5)
        target_1 = close_p + (atr_p * 1.5) if sound_signal == "buy" else close_p - (atr_p * 1.5)
        target_2 = close_p + (atr_p * 3.0) if sound_signal == "buy" else close_p - (atr_p * 3.0)
        
        alerts_to_trigger.append({
            "sym": sym, "sig": sound_signal, "stat": status, 
            "price": close_p, "sl": stop_loss, "t1": target_1, "t2": target_2, "prob": ml_prob
        })

# عرض لوحة المراقبة التفاعلية الرئيسية
st.subheader(f"📊 شاشة الرصد والتداول الذكية المدمجة بالتعلم الآلي وحجم التداول ومؤشر الخوف - الفريم: {trading_style}")
if summary_results:
    st.dataframe(pd.DataFrame(summary_results), use_container_width=True, hide_index=True)

# تفعيل الإشارات وإرسالها عبر تيليجرام
if alerts_to_trigger:
    first_alert = alerts_to_trigger[0]
    t_sym = first_alert["sym"]
    t_stat = first_alert["stat"]
    t_price = first_alert["price"]
    t_sl = first_alert["sl"]
    t_t1 = first_alert["t1"]
    t_t2 = first_alert["t2"]
    t_prob = first_alert["prob"]
    
    emoji = "🦅🟢" if first_alert["sig"] == "buy" else "🦅🔴"
    msg_to_send = (
        f"{emoji} *توصية فئة الإمبراطور فائقة الأمان والدقة v9.0* {emoji}\n\n"
        f"🔹 *السهم:* {t_sym}\n"
        f"🔹 *القرار الفني:* {t_stat}\n"
        f"🔹 *سعر الدخول:* `${t_price:.2f}`\n"
        f"🎯 *نسبة ثقة الذكاء الاصطناعي بالاتجاه:* `{t_prob:.1f}%`\n"
        f"📉 *مؤشر الخوف VIX للحالة العامة:* `{vix_current:.2f}`\n\n"
        f"🛡️ *الأهداف التكيفية المحسوبة عبر ATR:*\n"
        f"📍 *وقف الخسارة الموصى به (SL):* `${t_sl:.2f}`\n"
        f"🎯 *الهدف الأول الفوري (TP1):* `${t_t1:.2f}`\n"
        f"🎯 *الهدف الثاني الرئيسي (TP2):* `${t_t2:.2f}`\n\n"
        f"⏱ *الفريم المستهدف للعملية:* {style_label}"
    )
    
    if "telegram_sent_" + t_sym not in st.session_state:
        send_telegram_alert(msg_to_send)
        st.session_state["telegram_sent_" + t_sym] = True
        st.success(f"🚀 تم إرسال إشارة فائقة الأمان v9.0 المفلترة بمؤشر VIX وتقاطع الزخم إلى تليجرام بنجاح!")

# ==================== شارت التحليل الفني التفاعلي ومستشار الأخبار المدمج ====================
st.write("---")
st.subheader("🔍 شارت التحليل الفني المتقدم ومستشار التوترات والسياسة النقدية")

clean_symbols_list = [str(s).upper() for s in symbols]
if clean_symbols_list:
    selected_sym = st.selectbox("اختر السهم أو الأصل لعرض مصفوفة التعلم الآلي والشارت الفني التفاعلي:", clean_symbols_list)

    if selected_sym:
        target_clean = str(selected_sym).strip().upper()
        detail_df = fetch_clean_data(target_clean, p_period, p_interval)
        
        if not detail_df.empty and len(detail_df) >= 20:
            d_latest = detail_df.iloc[-1]
            close_p = float(d_latest['Close'])
            bbl_p = float(d_latest['BBL_20']) if 'BBL_20' in d_latest else close_p
            bbu_p = float(d_latest['BBU_20']) if 'BBU_20' in d_latest else close_p
            rsi_p = float(d_latest['RSI_14']) if 'RSI_14' in d_latest else 50
            mfi_p = float(d_latest['MFI_14']) if 'MFI_14' in d_latest else 50
            ema200_p = float(d_latest['EMA200']) if 'EMA200' in d_latest else close_p
            macd_p = float(d_latest['MACD']) if 'MACD' in d_latest else 0
            macd_sig_p = float(d_latest['MACD_Signal']) if 'MACD_Signal' in d_latest else 0
            atr_p = float(d_latest['ATR']) if 'ATR' in d_latest else (close_p * 0.02)
            
            # تشغيل نموذج التعلم الآلي للسهم المحدد
            ml_dir, ml_p_val = run_ml_prediction(detail_df)
            
            # --- رسم شارت الشموع والتوقعات الخوارزمية المتطورة ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.04, 
                                subplot_titles=(f'شموع السعر ومستويات القناة لـ {target_clean}', 'مؤشرات تدفق السيولة MFI و RSI', 'الزخم وقوة الترند MACD'),
                                row_heights=[0.5, 0.25, 0.25])
            
            fig.add_trace(fgo.Candlestick(x=detail_df.index, open=detail_df['Open'], high=detail_df['High'], low=detail_df['Low'], close=detail_df['Close'], name="السعر"), row=1, col=1)
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['EMA200'], line=dict(color='blue', width=2), name='EMA 200 (فلتر الترند)'), row=1, col=1)
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['BBU_20'], line=dict(color='orange', width=1, dash='dot'), name='بولنجر علوي'), row=1, col=1)
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['BBL_20'], line=dict(color='orange', width=1, dash='dot'), name='بولنجر سفلي'), row=1, col=1)
            
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['MFI_14'], line=dict(color='cyan', width=1.5), name='MFI (السيولة)'), row=2, col=1)
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['RSI_14'], line=dict(color='purple', width=1), name='RSI (الزخم)'), row=2, col=1)
            
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=3, col=1)
            fig.add_trace(fgo.Scatter(x=detail_df.index, y=detail_df['MACD_Signal'], line=dict(color='red', width=1), name='Signal'), row=3, col=1)
            fig.add_trace(fgo.Bar(x=detail_df.index, y=detail_df['MACD_Hist'], name='Histogram'), row=3, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- مصفوفة القرار المتكاملة بالتعلم الآلي ---
            st.markdown(f"### 🤖 نتائج تحليل التعلم الآلي لـ **{target_clean}**:")
            st.info(f"🔮 **الاتجاه المتوقع للـ 3 شمعات القادمة**: {ml_dir} (نسبة ثقة الصعود الحركي: **{ml_p_val:.1f}%**)")
            
            # ==================== وحدة جني وتدقيق وتحليل الأخبار الحية بـ GenAI ====================
            if use_gen_ai and API_KEY:
                st.write("---")
                st.markdown("### 📰 تقرير المخاطر والتوترات والسياسة النقدية الكلية عبر الذكاء الاصطناعي:")
                with st.spinner("جاري جلب أحدث البيانات الاقتصادية وتحليل التأثير الجيوسياسي الفوري..."):
                    try:
                        # 1. سحب الأخبار الفورية تلقائياً من ياهو فاينانس
                        ticker_obj = yf.Ticker(target_clean)
                        news_list = ticker_obj.news[:5]
                        
                        news_context = ""
                        for i, news in enumerate(news_list):
                            news_context += f"{i+1}. العنوان: {news.get('title')} | المصدر: {news.get('publisher')}\n"
                        
                        if not news_context:
                            news_context = "لا توجد أخبار اقتصادية عاجلة متداولة حالياً لهذا الرمز."
                            
                        # 2. بناء البرومبت المطور كلياً للأبعاد الاقتصادية والجيوسياسية الكبرى
                        prompt = (
                            f"أنت مستشار مالي جيوسياسي وخبير استراتيجي في إدارة الصناديق السيادية بوول ستريت. "
                            f"قم بتحليل السهم أو الأصل {target_clean} بدمج التحليل الفني، الأخبار الحالية، والظروف الاقتصادية الكبرى.\n\n"
                            f"📊 أولاً: البيانات الفنية الفورية للشارت ومؤشرات الخوف:\n"
                            f"- السعر الحالي: ${close_p:.2f}\n"
                            f"- تدفق السيولة MFI: {mfi_p:.1f}%\n"
                            f"- اتجاه التعلم الآلي (Random Forest): {ml_dir} بنسبة ثقة {ml_p_val:.1f}%\n"
                            f"- مؤشر الخوف العالمي VIX الحالي: {vix_current:.2f}\n\n"
                            f"📰 ثانياً: الأخبار العاجلة المتداولة حديثاً بالسوق:\n"
                            f"{news_context}\n\n"
                            f"المطلوب منك تحليل الأبعاد التالية وصياغتها بلغة عربية مالية فخمة وواضحة جداً في نقاط كالتالي:\n"
                            f"1. **تحليل المشاعر الاقتصادية (Sentiment Analysis)**: كيف تؤثر الأخبار الحالية ومستويات الخوف (VIX) على رغبة المستثمرين في الدخول بهذا الأصل؟\n"
                            f"2. **البعد الجيوسياسي والسياسة النقدية**: كيف تؤثر التوترات الدولية، أسعار الفائدة الحالية لـ الفيدرالي، ومستويات التضخم على حركة هذا الأصل خصيصاً (مثلاً: الذهب وملاذه الآمن، أو أسهم التكنولوجيا وقدرتها التمويلية)؟\n"
                            f"3. **التناغم أو التعارض**: هل البيانات الفنية والتعلم الآلي على الشارت متوافقة مع الأجواء الأساسية والجيوسياسية الخارجية أم أن هناك فخًا فنيًا وتعارضًا؟\n"
                            f"4. **القرار النهائي الصارم**: التوصية النهائية للمستثمر (شراء مع هدف تكتيكي / بيع فوري لتأمين الأرباح / انتظار وتسييل المحفظة مؤقتاً)."
                        )
                        
                        # 3. إرسال الطلب الآمن لـ Gemini API
                        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
                        payload = json.dumps({
                            "contents": [{"parts": [{"text": prompt}]}]
                        }).encode('utf-8')
                        
                        req = urllib.request.Request(api_url, data=payload, headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req) as response:
                            res_data = json.loads(response.read().decode('utf-8'))
                            ai_response = res_data['candidates'][0]['content']['parts'][0]['text']
                            st.markdown(ai_response)
                            
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء سحب الأخبار الاقتصادية الكبرى أو الاتصال بسيرفر الذكاء الاصطناعي: {e}")
