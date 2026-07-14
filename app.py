import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time
import urllib.request
import json
from streamlit_autorefresh import st_autorefresh

# 1. تفعيل ميزة التحديث التلقائي المستمر كل 60 ثانية بدون تدخل منك
st_autorefresh(interval=60000, key="watchlist_auto_refresh_final_v6")

st.set_page_config(page_title="منظومة التداول الذكية المستقلة", layout="wide")
st.title("🦅 منظومة مراقبة الأسهم الآلية بنظام التحديث المستمر وجدولة الصمت")
st.write("نسخة الجوال الفائقة: تحديث تلقائي، حفظ معزز عبر الرابط، وشارت أصلي مدمج غير قابل للحظر.")

# --- نظام الذاكرة الرقمية عبر الرابط للحفاظ على الأسهم من الاختفاء ---
# قراءة الأسهم من رابط المتصفح إن وجدت، وإلا استخدام القائمة الافتراضية
query_params = st.query_params
default_watchlist = query_params.get("watchlist", "NVDA,TSLA,ORCL,GLD")

# القائمة الجانبية لإدارة المحفظة والأسهم وساعات الصمت
st.sidebar.header("📋 لوحة التحكم والمراقبة")

watchlist_input = st.sidebar.text_area(
    "أدخل رموز الأسهم والذهب مفصولة بفاصلة (,):", 
    value=default_watchlist
)

# حفظ التغييرات فوراً في رابط المتصفح لمنع الاختفاء عند التحديث التلقائي
st.query_params["watchlist"] = watchlist_input

symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
# -------------------------------------------------------------------------

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
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BBU_20'] = df['MA20'] + (df['STD20'] * 2)
        df['BBL_20'] = df['MA20'] - (df['STD20'] * 2)
        
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['STOCHK_14'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        df['STOCHD_14'] = df['STOCHK_14'].rolling(window=3).mean()
        
        mfv = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)) * df['Volume']
        df['CMF_20'] = mfv.rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)
        return df
    except:
        return df

@st.cache_data(ttl=30)
def fetch_clean_data(symbol):
    try:
        end = datetime.today()
        start = end - timedelta(days=150)
        data = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = calculate_indicators_manually(data)
        return data
    except:
        return pd.DataFrame()

summary_results = []
alerts_to_trigger = []

for sym in symbols:
    df = fetch_clean_data(sym)
    if df.empty or len(df) < 30 or 'STOCHK_14' not in df.columns: continue
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    status = "🟡 انتظر (لا توجد إشارة صريحة)"
    sound_signal = None
    
    if float(latest['STOCHK_14']) > 80 and float(latest['STOCHK_14']) < float(latest['STOCHD_14']) and float(previous['STOCHK_14']) >= float(previous['STOCHD_14']):
        status = "🔴 بيع (قمة وتصريف سيولة)" if float(latest['CMF_20']) < 0 else "⚠️ بيع (تراجع فني وشيك)"
        sound_signal = "sell"
    elif float(latest['STOCHK_14']) < 20 and float(latest['STOCHK_14']) > float(latest['STOCHD_14']) and float(previous['STOCHK_14']) <= float(previous['STOCHD_14']):
        status = "🟢 شراء (قاع ذهبي وتجمع)" if float(latest['CMF_20']) > 0 else "🚀 شراء (ارتداد صاعد قادم)"
        sound_signal = "buy"

    summary_results.append({
        "الرمز": sym,
        "السعر الحالي": f"${float(latest['Close']):.2f}",
        "تدفق السيولة (CMF)": f"{float(latest['CMF_20']):.2f}",
        "إشارة الرادار الفورية": status
    })
    if sound_signal:
        alerts_to_trigger.append((sym, sound_signal, status, float(latest['Close'])))

st.subheader(f"📊 لوحة المراقبة الحية لجميع الأسهم (تحديث تلقائي: {datetime.now().strftime('%I:%M:%S %p')})")
if summary_results:
    st.dataframe(pd.DataFrame(summary_results), use_container_width=True, hide_index=True)

if alerts_to_trigger and not is_silent_hours:
    target_sym, sig, stat, price = alerts_to_trigger
    st.subheader("🚨 رادار التنبيهات الصوتية النشط (اضغط على الشاشة لكتم الصوت)")
    if "🔴" in stat or "⚠️" in stat:
        st.error(f"⚠️ تنبيه بيع عاجل: السهم {target_sym} دخل منطقة تراجع عند سعر ${price:.2f}! (انقر لكتم النغمة)")
        play_interactive_sound("sell")
    else:
        st.success(f"🚀 تنبيه شراء ذهبي: السهم {target_sym} دخل منطقة ارتداد عند سعر ${price:.2f}! (انقر لكتم النغمة)")
        play_interactive_sound("buy")

st.write("---")
st.subheader("🔍 مستشار الفحص المخصص وحساب دقة الإشارات من 100")
selected_sym = st.selectbox("اختر السهم الذي تريد الدخول إليه لعرض تقييم البيع والشراء والقرار الحاسم له من 100:", symbols)

if selected_sym:
    detail_df = fetch_clean_data(selected_sym)
    news_score, news_data = fetch_news_sentiment_fast(selected_sym)
    if not detail_df.empty and 'STOCHK_14' in detail_df.columns:
        d_latest = detail_df.iloc[-1]
        
        buy_score = 0
        sell_score = 0
        
        if float(d_latest['STOCHK_14']) < 20: buy_score += 35
        if float(d_latest['STOCHK_14']) > float(d_latest['STOCHD_14']): buy_score += 15
        if float(d_latest['Close']) <= float(d_latest['BBL_20']): buy_score += 25
        if float(d_latest['RSI_14']) < 35: buy_score += 15
        if float(d_latest['CMF_20']) > 0: buy_score += 10
        
        if float(d_latest['STOCHK_14']) > 80: sell_score += 35
        if float(d_latest['STOCHK_14']) < float(d_latest['STOCHD_14']): sell_score += 15
        if float(d_latest['Close']) >= float(d_latest['BBU_20']): sell_score += 25
        if float(d_latest['RSI_14']) > 65: sell_score += 15
        if float(d_latest['CMF_20']) < 0: sell_score += 10
        
        final_advice = "🟡 انتظر (تذبذب عرضي، لا تدخل السوق الآن)"
        card_color = "info"
        
        if buy_score >= 65 and buy_score > sell_score:
            final_advice = f"🟢 شراء (السعر في أفضل نقطة اقتناص صاعدة بنسبة تقييم {buy_score}/100)"
            card_color = "success"
        elif sell_score >= 65 and sell_score > buy_score:
            final_advice = f"🔴 بيع (السعر في أفضل نقطة تراجع وهبوط بنسبة تقييم {sell_score}/100)"
            card_color = "warning"
            
        st.markdown(f"### 🎯 قرار التوجيه الفوري لـ سهم **{selected_sym}**:")
        if card_color == "success": st.success(final_advice)
        elif card_color == "warning": st.warning(final_advice)
        else: st.info(final_advice)
            
        col1, col2, col3 = st.columns(3)
        col1.metric("تقييم دقة الشراء (Buy Score)", f"{buy_score}/100")
        col2.metric("تقييم دقة البيع (Sell Score)", f"{sell_score}/100")
        col3.metric("السعر اللحظي الحالي للسهم", f"${float(d_latest['Close']):.2f}")
        
        # --- التحديث الجوهري: شارت الجوال الأصلي المدمج المقاوم للحظر بنسبة 100% ---
