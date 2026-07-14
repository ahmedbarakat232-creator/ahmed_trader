import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time
import urllib.request
import json
from streamlit_autorefresh import st_autorefresh

# 1. تفعيل ميزة التحديث التلقائي المستمر كل 30 ثانية لتحديث الأسعار الفورية
st_autorefresh(interval=30000, key="watchlist_auto_refresh_final_v20")

st.set_page_config(page_title="منظومة التداول المتعددة", layout="wide")
st.title("🦅 منظومة مراقبة الأسهم الآلية متعددة الأنماط الاستثمارية")
st.write("النسخة الذهبية المستقرة: تم ضبط وعزل عناصر الجوال لعرض كافة التقييمات والشارتات عمودياً بكفاءة كاملة.")

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

def calculate_secure_indicators(df):
    try:
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BBU_20'] = df['MA20'] + (df['STD20'] * 2)
        df['BBL_20'] = df['MA20'] - (df['STD20'] * 2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        return df
    except:
        return df

@st.cache_data(ttl=15)
def fetch_clean_data(symbol, period, interval):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = calculate_secure_indicators(data)
        return data
    except:
        return pd.DataFrame()

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
    
    status = "🟡 انتظر (لا توجد إشارة حاسمة)"
    sound_signal = None
    
    if close_p <= bbl_p * 1.01 or rsi_p < 35:
        status = "🟢 إشارة شراء واقتناص"
        sound_signal = "buy"
    elif close_p >= bbu_p * 0.99 or rsi_p > 65:
        status = "🔴 إشارة بيع وتخفيف"
        sound_signal = "sell"

    summary_results.append({
        "الرمز": sym,
        "السعر الحالي": f"${close_p:.2f}",
        "مؤشر الزخم (RSI)": f"{rsi_p:.1f}" if 'RSI_14' in latest else "50.0",
        "الرادار الفوري الحالي": status
    })
    if sound_signal:
        alerts_to_trigger.append({"sym": sym, "sig": sound_signal, "stat": status, "price": close_p})

st.subheader(f"📊 لوحة المراقبة الحية - النمط الحالي: {trading_style}")
if summary_results:
    st.dataframe(pd.DataFrame(summary_results), use_container_width=True, hide_index=True)

if alerts_to_trigger and not is_silent_hours:
    st.subheader("🚨 رادار التنبيهات الصوتية النشط (اضغط على الشاشة لكتم الصوت)")
    first_alert = alerts_to_trigger[0]
    t_sym = first_alert["sym"]
    t_sig = first_alert["sig"]
    t_stat = first_alert["stat"]
    t_price = first_alert["price"]
    
    if "🔴" in t_stat or "⚠️" in t_stat:
        st.error(f"⚠️ تنبيه خروج: السهم {t_sym} دخل منطقة قمة بناءً على {style_label} عند سعر ${t_price:.2f}!")
        play_interactive_sound("sell")
    else:
        st.success(f"🚀 تنبيه دخول: السهم {t_sym} دخل منطقة قاع بناءً على {style_label} عند سعر ${t_price:.2f}!")
        play_interactive_sound("buy")

st.write("---")
st.subheader("🔍 مستشار الفحص المخصص وحساب دقة الإشارات من 100")

clean_symbols_list = [str(s).upper() for s in symbols]
selected_sym = st.selectbox("اختر السهم الذي تريد الدخول إليه لعرض تقييم البيع والشراء والقرار الحاسم له من 100:", clean_symbols_list)

if selected_sym:
    target_clean = str(selected_sym).strip().upper()
    detail_df = fetch_clean_data(target_clean, p_period, p_interval)
    news_score, news_data = fetch_news_sentiment_fast(target_clean)
    
    if not detail_df.empty:
        d_latest = detail_df.iloc[-1]
        
        close_p = float(d_latest['Close'])
        bbl_p = float(d_latest['BBL_20']) if 'BBL_20' in d_latest else close_p
        bbu_p = float(d_latest['BBU_20']) if 'BBU_20' in d_latest else close_p
        rsi_p = float(d_latest['RSI_14']) if 'RSI_14' in d_latest else 50
        
        # تصفير وحساب فوري نشط يتغير 100% مع تبديل السهم
        buy_score = 15
        sell_score = 15
        
        if close_p <= bbl_p * 1.02: buy_score += 40
        if rsi_p < 40: buy_score += 35
        if news_score > 0: buy_score += 10
        
        if close_p >= bbu_p * 0.98: sell_score += 40
        if rsi_p > 60: sell_score += 35
        if news_score < 0: sell_score += 10
            
        buy_score = int(max(5, min(100, buy_score)))
        sell_score = int(max(5, min(100, sell_score)))
        
        is_buy_approved = (buy_score >= 60 and buy_score > sell_score)
        is_sell_approved = (sell_score >= 60 and sell_score > buy_score)
        
        final_advice = f"🟢 شراء (السعر في أفضل منطقة قاع بناءً على {style_label} بتقييم {buy_score}/100)" if is_buy_approved else (f"🔴 بيع (السعر في أفضل منطقة قمة بناءً على {style_label} بتقييم {sell_score}/100)" if is_sell_approved else "🟡 انتظر (تذبذب عرضي، لا تدخل السوق الآن)")
        
        if is_buy_approved:
            st.success(final_advice)
        elif is_sell_approved:
            st.warning(final_advice)
        else:
            st.info(final_advice)
            
        # --- التحديث الفخم: عزل وعرض العدادات عمودياً لحل مشكلة شاشة الأندرويد 100% ---
        st.markdown(f"### 📊 لوحة البيانات الرقمية الحية لـ **{target_clean}**:")
        st.write(f"💵 **السعر الحالي الفوري**: ${close_p:.2f}")
        st.write(f"🟢 **تقييم دقة الشراء (Buy Score)**: {buy_score}/100")
        st.write(f"🔴 **تقييم دقة البيع (Sell Score)**: {sell_score}/100")
