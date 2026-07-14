import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time
import urllib.request
import json
from streamlit_autorefresh import st_autorefresh

# 1. تفعيل ميزة التحديث التلقائي المستمر كل 30 ثانية لتحديث الأسعار الفورية
st_autorefresh(interval=30000, key="watchlist_auto_refresh_final_v11")

st.set_page_config(page_title="منظومة التداول المتعددة", layout="wide")
st.title("🦅 منظومة مراقبة الأسهم الآلية متعددة الأنماط الاستثمارية")
st.write("النسخة الفائقة المستقرة: تتيح لك التبديل الفوري بين المضاربة اللحظية السريعة والاستثمار بعيد المدى بضغطة زر واحدة.")

# نظام الذاكرة الرقمية عبر الرابط للحفاظ على الأسهم من الاختفاء
query_params = st.query_params
default_watchlist = query_params.get("watchlist", "NVDA,TSLA,ORCL,GLD")

# 2. القائمة الجانبية لإدارة المحفظة
st.sidebar.header("🎛️ لوحة التحكم وإدارة الأنماط")
watchlist_input = st.sidebar.text_area(
    "أدخل رموز الأسهم والذهب مفصولة بفاصلة (,):", 
    value=default_watchlist
)
st.query_params["watchlist"] = watchlist_input
symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

trading_style = st.sidebar.selectbox(
    "اختر نمط التداول المطلوب:",
    [
        "🔥 مضاربة سريعة (نفس اليوم - 5 دقائق)",
        "📈 مضاربة متوسطة (مدار أيام - يومي)",
        "💼 استثمار طويل (مدار أشهر - أسبوعي)",
        "🏆 استثمار طويل جداً (سنوات - شهري)"
    ]
)

# ضبط الفواصل الزمنية برمجياً بناءً على اختيار المستخدم
if "🔥" in trading_style:
    p_period, p_interval = "5d", "5m"
    style_label = "شارت الـ 5 دقائق اللحظي"
elif "📈" in trading_style:
    p_period, p_interval = "1y", "1d"
    style_label = "الشارت اليومي"
elif "💼" in trading_style:
    p_period, p_interval = "3y", "1wk"
    style_label = "الشارت الأسبوعي"
else:
    p_period, p_interval = "max", "1mo"
    style_label = "الشارت الشهري التاريخي"

st.sidebar.subheader("🔕 جدولة ساعات الصمت (كتم التنبيهات)")
enable_dnd = st.sidebar.checkbox("تفعيل خاصية كتم التنبيهات المؤقت")
dnd_start = st.sidebar.time_input("وقت بدء الكتم:", time(23, 0))
dnd_end = st.sidebar.time_input("وقت انتهاء الكتم:", time(7, 0))

current_time = datetime.now().time()
is_silent_hours = False

if enable_dnd:
    if dnd_start <= dnd_end:
        is_silent_hours = dnd_start <= current_time <= dnd_end
    else:
        is_silent_hours = current_time >= dnd_start or current_time <= dnd_end

if is_silent_hours:
    st.sidebar.warning("🌙 وضع الصمت نشط حالياً: تم كتم الأصوات تلقائياً.")
else:
    st.sidebar.success("🔔 نظام التنبيهات اللحظية يعمل الآن بكفاءة.")

def fetch_news_sentiment_fast(query_term):
    try:
        url = f"https://yahoo.com{query_term}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            news_list = data.get('news', [])[:5]
    except:
        return 0, []
        
    positive_keywords = ['earnings beat', 'rate cut', 'stimulus', 'growth', 'demand surge', 'bullish', 'upgrade']
    negative_keywords = ['rate hike', 'inflation spike', 'recession', 'sanctions', 'war', 'bearish', 'downgrade']
    
    score = 0
    news_records = []
    for item in news_list:
        title = item.get('title', '').lower()
        link = item.get('link', '#')
        item_score = 0
        for p_word in positive_keywords:
            if p_word in title: item_score += 2
        for n_word in negative_keywords:
            if n_word in title: item_score -= 2
        score += item_score
        news_records.append({"title": item.get('title', ''), "link": link, "score": item_score})
    return score, news_records

def play_interactive_sound(sound_type):
    if is_silent_hours: return
    sound_url = "https://mixkit.co" if sound_type == "buy" else "https://mixkit.co"
    interactive_audio_html = f"""
    <audio id="traderAudio" loop autoplay><source src="{sound_url}" type="audio/wav"></audio>
    <script>
      function stopAlert() {{
          var audio = document.getElementById("traderAudio");
          if (audio) {{ audio.pause(); audio.currentTime = 0; }}
      }}
      window.parent.document.addEventListener('click', stopAlert);
      window.parent.document.addEventListener('touchstart', stopAlert);
      document.addEventListener('click', stopAlert);
    </script>
    """
    st.components.v1.html(interactive_audio_html, height=0, width=0)

def calculate_indicators_manually(df):
    try:
        # حساب RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # حساب Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BBU_20'] = df['MA20'] + (df['STD20'] * 2)
        df['BBL_20'] = df['MA20'] - (df['STD20'] * 2)
        
        # حساب Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['STOCHK_14'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['STOCHD_14'] = df['STOCHK_14'].rolling(window=3).mean()
        
        # حساب Chaikin Money Flow (CMF) للسيولة
        mfv = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)) * df['Volume']
        df['CMF_20'] = mfv.rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)
        return df
    except:
        return df

@st.cache_data(ttl=15)
def fetch_clean_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = calculate_indicators_manually(data)
        return data
    except:
        return pd.DataFrame()

summary_results = []
alerts_to_trigger = []

for sym in symbols:
    df = fetch_clean_data(sym, p_period, p_interval)
    if df.empty or len(df) < 30 or 'STOCHK_14' not in df.columns: continue
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    status = "🟡 انتظر (لا توجد إشارة حاسمة)"
    sound_signal = None
    
    if float(latest['STOCHK_14']) > 80 and float(latest['STOCHK_14']) < float(latest['STOCHD_14']) and float(previous['STOCHK_14']) >= float(previous['STOCHD_14']):
        status = "🔴 إشارة بيع وتخفيف" if float(latest['CMF_20']) < 0 else "⚠️ تراجع فني وشيك"
        sound_signal = "sell"
    elif float(latest['STOCHK_14']) < 20 and float(latest['STOCHK_14']) > float(latest['STOCHD_14']) and float(previous['STOCHK_14']) <= float(previous['STOCHD_14']):
        status = "🟢 إشارة شراء واقتناص" if float(latest['CMF_20']) > 0 else "🚀 ارتداد صاعد قادم"
        sound_signal = "buy"

    summary_results.append({
        "الرمز": sym,
        "السعر الحالي": f"${float(latest['Close']):.2f}",
        "تدفق السيولة (CMF)": f"{float(latest['CMF_20']):.2f}",
        "الرادار الفوري الحالي": status
    })
    if sound_signal:
        alerts_to_trigger.append((sym, sound_signal, status, float(latest['Close'])))

st.subheader(f"📊 لوحة المراقبة الحية - النمط الحالي: {trading_style}")
if summary_results:
    st.dataframe(pd.DataFrame(summary_results), use_container_width=True, hide_index=True)

if alerts_to_trigger and not is_silent_hours:
    target_sym, sig, stat, price = alerts_to_trigger
    st.subheader("🚨 رادار التنبيهات الصوتية النشط (اضغط على الشاشة لكتم الصوت)")
    if "🔴" in stat or "⚠️" in stat:
        st.error(f"⚠️ تنبيه خروج: السهم {target_sym} دخل منطقة قمة بناءً على {style_label} عند سعر ${price:.2f}!")
        play_interactive_sound("sell")
    else:
        st.success(f"🚀 تنبيه دخول: السهم {target_sym} دخل منطقة قاع بناءً على {style_label} عند سعر ${price:.2f}!")
        play_interactive_sound("buy")

st.write("---")
st.subheader("🔍 مستشار الفحص المخصص وحساب دقة الإشارات من 100")
selected_sym = st.selectbox("اختر السهم الذي تريد الدخول إليه لعرض تقييم البيع والشراء والقرار الحاسم له من 100:", symbols)

if selected_sym:
    detail_df = fetch_clean_data(selected_sym, p_period, p_interval)
    news_score, news_data = fetch_news_sentiment_fast(selected_sym)
    if not detail_df.empty and 'STOCHK_14' in detail_df.columns:
        d_latest = detail_df.iloc[-1]
        
        buy_score = int(5) # قيمة افتراضية دنيا
        sell_score = int(5)
        
        # حماية برمجية لضمان صحة الأرقام والحسابات
        try:
            stoch_k = float(d_latest['STOCHK_14'])
            stoch_d = float(d_latest['STOCHD_14'])
            close_p = float(d_latest['Close'])
            bbl_p = float(d_latest['BBL_20'])
            bbu_p = float(d_latest['BBU_20'])
            rsi_p = float(d_latest['RSI_14'])
            cmf_p = float(d_latest['CMF_20'])
            
            # معادلة الشراء من 100
            b_s = 0
            if stoch_k < 20: b_s += 35
            if stoch_k > stoch_d: b_s += 15
            if close_p <= bbl_p: b_s += 25
            if rsi_p < 35: b_s += 15
            if cmf_p > 0: b_s += 10
            buy_score = int(max(0, min(100, b_s)))
            
            # معادلة البيع من 100
            s_s = 0
            if stoch_k > 80: s_s += 35
            if stoch_k < stoch_d: s_s += 15
            if close_p >= bbu_p: s_s += 25
            if rsi_p > 65: s_s += 15
            if cmf_p < 0: s_s += 10
            sell_score = int(max(0, min(100, s_s)))
        except:
            pass
        
        final_advice = "🟡 انتظر (تذبذب عرضي، لا تدخل السوق الآن)"
        card_color = "info"
        
        if buy_score >= 65 and buy_score > sell_score:
