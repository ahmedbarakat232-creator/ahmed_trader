import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from gnews import GNews
from datetime import datetime, timedelta, time
import smtplib
from email.mime.text import MIMEText
from streamlit_autorefresh import st_autorefresh

# إعداد التحديث التلقائي المستمر للتطبيق كل 60 ثانية (60000 مللي ثانية)
st_autorefresh(interval=60000, key="watchlist_auto_refresh")

st.set_page_config(page_title="منظومة التداول الذكية المستقلة", layout="wide")
st.title("🦅 منظومة مراقبة الأسهم الآلية بنظام التحديث المستمر وجدولة الصمت")
st.write("النسخة المستقلة: يقوم التطبيق بتحديث نفسه كل دقيقة، وإصدار نغمات هادئة تتوقف باللمس، مع إمكانية جدولة ساعات كتم التنبيهات.")

# القائمة الجانبية لإدارة المحفظة والإشعارات
st.sidebar.header("📋 لوحة التحكم والمراقبة")
watchlist_input = st.sidebar.text_area(
    "أدخل رموز الأسهم والذهب مفصولة بفاصلة (,):", 
    value="NVDA, TSLA, AAPL, GC=F"
)
symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

# التحديث الجديد: التحكم في غلق وكتم الإشعارات في أوقات محددة
st.sidebar.subheader("🔕 جدولة ساعات الصمت (كتم التنبيهات)")
enable_dnd = st.sidebar.checkbox("تفعيل خاصية كتم التنبيهات المؤقت")
dnd_start = st.sidebar.time_input("وقت بدء الكتم:", time(23, 0)) # الافتراضي 11 مساءً
dnd_end = st.sidebar.time_input("وقت انتهاء الكتم:", time(7, 0))   # الافتراضي 7 صباحاً

# التحقق مما إذا كان الوقت الحالي يقع ضمن ساعات الصمت المحددة
current_time = datetime.now().time()
is_silent_hours = False

if enable_dnd:
    if dnd_start <= dnd_end:
        is_silent_hours = dnd_start <= current_time <= dnd_end
    else: # في حال كان الكتم يمتد عبر منتصف الليل (مثلاً من 11 م إلى 7 ص)
        is_silent_hours = current_time >= dnd_start or current_time <= dnd_end

if is_silent_hours:
    st.sidebar.warning("🌙 وضع الصمت نشط حالياً: تم كتم الأصوات وإيقاف الإيميلات.")
else:
    st.sidebar.success("🔔 نظام التنبيهات اللحظية يعمل الآن بكفاءة.")

# إعدادات البريد الإلكتروني
st.sidebar.subheader("📧 تنبيهات البريد الإلكتروني (اختياري)")
enable_email = st.sidebar.checkbox("تفعيل تنبيهات البريد")
sender_email = st.sidebar.text_input("بريدك الإلكتروني:")
sender_password = st.sidebar.text_input("كلمة مرور التطبيق:", type="password")
receiver_email = st.sidebar.text_input("البريد المستلم:")

# دالة توليد الأصوات التفاعلية (تتوقف عند اللمس) وتراعي ساعات الصمت
def play_interactive_sound(sound_type):
    if is_silent_hours:
        return # لا تشغل أي صوت إذا كنا في ساعات الصمت
        
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

def send_email_alert(subject, body):
    if is_silent_hours:
        return # لا ترسل إيميل إذا كنا في ساعات الصمت
        
    if enable_email and sender_email and sender_password and receiver_email:
        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = receiver_email
            server = smtplib.SMTP_SSL('://gmail.com', 465)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
        except:
            pass

@st.cache_data(ttl=30) # تقليل الكاش لضمان تحديث مستمر حقيقي كل دقيقة لبيانات السوق
def fetch_clean_data(symbol):
    try:
        end = datetime.today()
        start = end - timedelta(days=150)
        data = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return pd.DataFrame()

summary_results = []
alerts_to_trigger = []

# 1. شاشة المراقبة والفحص الشامل في الخلفية (تتحدث تلقائياً)
for sym in symbols:
    df = fetch_clean_data(sym)
    if df.empty or len(df) < 30: continue
        
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.cmf(length=20, append=True)
    
    rsi_name = [c for c in df.columns if 'RSI_' in c]
    stochk_name = [c for c in df.columns if 'STOCHK_' in c]
    stochd_name = [c for c in df.columns if 'STOCHD_' in c]
    cmf_name = [c for c in df.columns if 'CMF_' in c]
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    status = "🟡 انتظر (لا توجد إشارة صريحة)"
    sound_signal = None
    
    if float(latest[stochk_name]) > 80 and float(latest[stochk_name]) < float(latest[stochd_name]) and float(previous[stochk_name]) >= float(previous[stochd_name]):
        status = "🔴 بيع (قمة وتصريف سيولة)" if float(latest[cmf_name]) < 0 else "⚠️ بيع (تراجع فني وشيك)"
        sound_signal = "sell"
    elif float(latest[stochk_name]) < 20 and float(latest[stochk_name]) > float(latest[stochd_name]) and float(previous[stochk_name]) <= float(previous[stochd_name]):
        status = "🟢 شراء (قاع ذهبي وتجمع)" if float(latest[cmf_name]) > 0 else "🚀 شراء (ارتداد صاعد قادم)"
        sound_signal = "buy"

    summary_results.append({
        "الرمز": sym,
        "السعر الحالي": f"${float(latest['Close']):.2f}",
        "تدفق السيولة (CMF)": f"{float(latest[cmf_name]):.2f}",
        "إشارة الرادار الفورية": status
    })
    if sound_signal:
        alerts_to_trigger.append((sym, sound_signal, status, float(latest['Close'])))

# عرض لوحة المراقبة العامة المحدثة تلقائياً
st.subheader(f"📊 لوحة المراقبة الحية (آخر تحديث آلي: {datetime.now().strftime('%I:%M:%S %p')})")
if summary_results:
    st.dataframe(pd.DataFrame(summary_results), use_container_width=True, hide_index=True)

# تشغيل نغمة التنبيه بشرط عدم تفعيل وضع الصمت
if alerts_to_trigger and not is_silent_hours:
    target_sym, sig, stat, price = alerts_to_trigger
    st.subheader("🚨 رادار التنبيهات الصوتية النشط (اضغط على الشاشة لكتم الصوت)")
    if "🔴" in stat or "⚠️" in stat:
        st.error(f"⚠️ تنبيه بيع عاجل: السهم {target_sym} دخل منطقة تراجع عند سعر ${price:.2f}! (انقر لكتم النغمة)")
        play_interactive_sound("sell")
        send_email_alert(f"🚨 تنبيه بيع عاجل: {target_sym}", f"تم رصد إشارة بيع وتراجع لـ {target_sym}")
    else:
        st.success(f"🚀 تنبيه شراء ذهبي: السهم {target_sym} دخل منطقة ارتداد عند سعر ${price:.2f}! (انقر لكتم النغمة)")
        play_interactive_sound("buy")
        send_email_alert(f"🟢 تنبيه شراء عاجل: {target_sym}", f"تم رصد إشارة شراء وارتداد لـ {target_sym}")

# 2. مستشار الفحص المخصص والتقييم الحسابي من 100
st.write("---")
st.subheader("🔍 مستشار الفحص المخصص وحساب دقة الإشارات من 100")
selected_sym = st.selectbox("اختر السهم الذي تريد الدخول إليه لعرض تقييم البيع والشراء والقرار الحاسم له:", symbols)

if selected_sym:
    detail_df = fetch_clean_data(selected_sym)
    if not detail_df.empty:
        detail_df.ta.bbands(length=20, std=2, append=True)
        detail_df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        detail_df.ta.rsi(length=14, append=True)
        detail_df.ta.cmf(length=20, append=True)
        
        bbl_name = [c for c in detail_df.columns if 'BBL_' in c]
        bbu_name = [c for c in detail_df.columns if 'BBU_' in c]
        rsi_det = [c for c in detail_df.columns if 'RSI_' in c]
        stochk_det = [c for c in detail_df.columns if 'STOCHK_' in c]
        stochd_det = [c for c in detail_df.columns if 'STOCHD_' in c]
        cmf_det = [c for c in detail_df.columns if 'CMF_' in c]
        
        d_latest = detail_df.iloc[-1]
        
        buy_score = 0
        sell_score = 0
        
        # احتساب قوى تقييم الشراء من 100
        if float(d_latest[stochk_det]) < 20: buy_score += 35
        if float(d_latest[stochk_det]) > float(d_latest[stochd_det]): buy_score += 15
        if float(d_latest['Close']) <= float(d_latest[bbl_name]): buy_score += 25
        if float(d_latest[rsi_det]) < 35: buy_score += 15
        if float(d_latest[cmf_det]) > 0: buy_score += 10
        
        # احتساب قوى تقييم البيع من 100
        if float(d_latest[stochk_det]) > 80: sell_score += 35
        if float(d_latest[stochk_det]) < float(d_latest[stochd_det]): sell_score += 15
        if float(d_latest['Close']) >= float(d_latest[bbu_name]): sell_score += 25
        if float(d_latest[rsi_det]) > 65: sell_score += 15
        if float(d_latest[cmf_det]) < 0: sell_score += 10
        
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