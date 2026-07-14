import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as fgo
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import json
import os
from streamlit_autorefresh import st_autorefresh

# استيراد أدوات التعلم الآلي
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# تفعيل التحديث التلقائي المستمر كل 30 ثانية لتحديث الأسعار الفورية ومراقبة الإشارات
st_autorefresh(interval=30000, key="mobile_refresh_v130")

# إعداد الصفحة لتناسب شاشات الجوال تماماً
st.set_page_config(page_title="منصة AI v13.0", layout="centered", initial_sidebar_state="collapsed")

# كود CSS مخصص لتنسيق مريح للعين على شاشات الجوال
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
""", unsafe_allow_html=True)

st.title("🦅 منصة التداول v13.0 (نظام التحكم المنفصل والإشعارات)")

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

# ==================== إعدادات المحفظة، الإشعارات والـ API ====================
with st.expander("⚙️ إعدادات المحفظة والاتصال والإشعارات"):
    watchlist_input = st.text_area("أدخل الرموز (مثل: NVDA,TSLA,GC=F):", value=settings.get("watchlist", "NVDA,TSLA,AAPL,GC=F"))
    
    # حقول إعدادات إشعارات الجوال
    phone_input = st.text_input("📱 رقم الجوال مع رمز الدولة (مثال: 9665xxxxxxxx):", value=settings.get("phone_number", ""))
    notif_active = st.checkbox("🔔 تشغيل الإشعارات الفورية عند نقطة الشراء أو البيع القوية", value=settings.get("notifications_active", False))
    
    # حفظ الإعدادات تلقائياً عند التعديل
    if (watchlist_input != settings.get("watchlist") or 
        phone_input != settings.get("phone_number") or 
        notif_active != settings.get("notifications_active")):
        settings["watchlist"] = watchlist_input
        settings["phone_number"] = phone_input
        settings["notifications_active"] = notif_active
        save_settings(settings)
        st.success("💾 تم حفظ ومزامنة إعداداتك وهاتفك بنجاح!")
    
    symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
    
    API_KEY = st.text_input("مفتاح الـ Gemini API (اختياري للأخبار):", type="password")
    use_gen_ai = st.checkbox("🔥 تفعيل مستشار الأخبار الذكي التلقائي")

# ==================== 🛠️ الخيار الرئيسي لتحديد فريم الحسابات والجدول (فقط) ====================
calculation_frame = st.selectbox(
    "📊 اختر الفريم الخاص بقوة البيع/الشراء والجدول السفلي:",
    [
        "⏱️ تكتيكي متوسط (4 ساعات)",
        "⚡ مضاربة لحظية (5 دقائق)",
        "📈 استراتيجي يومي (Daily)",
        "💼 طويل الأمد شهري (Monthly)"
    ]
)

# تعيين الفترات الزمنية لحسابات القوى والجدول فقط بناء على الاختيار
if "5 دقائق" in calculation_frame:
    calc_period, calc_interval = "5d", "5m"
elif "4 ساعات" in calculation_frame:
    calc_period, calc_interval = "60d", "4h"
elif "يومي" in calculation_frame:
    calc_period, calc_interval = "2y", "1d"
else:
    calc_period, calc_interval = "10y", "1mo"

# ==================== احتساب المؤشرات الفنية ====================
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

# ==================== إرسال إشعارات الجوال التلقائية ====================
def send_sms_notification(phone, symbol, message):
    """
    هنا يتم وضع كود بوابة الإرسال المفضلة لديك (مثل Twilio أو بوابة SMS محلية).
    كمثال برمجي قياسي فعال، سنقوم بالطباعة في واجهة النظام، ويمكنك ربطه بـ API الإرسال الخاص بك بسهولة.
    """
    # مثال للتنبيه على واجهة الويب
    st.info(f"📱 [محاكاة الإشعار]: تم إرسال رسالة إلى {phone} -> {symbol}: {message}")
    
    # إذا كنت ترغب بالربط الفعلي مع Twilio كأكثر البوابات استخداماً:
    # try:
    #     import requests
    #     requests.post("YOUR_SMS_GATEWAY_API_URL", data={"to": phone, "msg": message})
    # except:
    #     pass

# ==================== خوارزمية حساب القوى والقرارات المستقلة ====================
def calculate_scores_and_decision(df, symbol=""):
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
    final_buy_score = max(0, min(100, int(buy_score)))
    
    # 2. حساب تقييم البيع (Sell Score)
    sell_score = 30
    if close_p < ema200_p: sell_score += 15
    if rsi_p > 60: sell_score += 20
    if mfi_p > 65: sell_score += 20
    if ema9_p < ema21_p: sell_score += 15
    if vol_now > vol_avg: sell_score += 10
    final_sell_score = max(0, min(100, int(sell_score)))
    
    # تحديد الإشارة وإطلاق الإشعارات الفورية للجوال
    if final_buy_score >= 80:
        decision = "🟢 شراء قوي (BUY)"
        if settings.get("notifications_active") and settings.get("phone_number"):
            send_sms_notification(settings.get("phone_number"), symbol, f"إشارة شراء قوية على {symbol}! التقييم: {final_buy_score}/100 السعر: ${close_p:.2f}")
    elif final_sell_score >= 80:
        decision = "🔴 بيع قوي (SELL)"
        if settings.get("notifications_active") and settings.get("phone_number"):
            send_sms_notification(settings.get("phone_number"), symbol, f"إشارة بيع قوية على {symbol}! التقييم: {final_sell_score}/100 السعر: ${close_p:.2f}")
    else:
        decision = "🟡 انتظار"
        
    return final_buy_score, final_sell_score, decision

# ==================== معالجة السهم الفردي المختار ====================
st.write("---")
clean_symbols_list = [str(s).upper() for s in symbols]
if clean_symbols_list:
    selected_sym = st.selectbox("🎯 اختر السهم للتحليل وعرض الشارت:", clean_symbols_list)

    if selected_sym:
        target_clean = str(selected_sym).strip().upper()
        
        # 🟢 أولاً: جلب بيانات الحسابات والقوى بناءً على الفريم المستقل (لا علاقة للشارت به)
        calc_df = fetch_clean_data(target_clean, calc_period, calc_interval)
        f_buy, f_sell, f_decision = calculate_scores_and_decision(calc_df, target_clean)
        
        decision_color = "#2ecc71" if "BUY" in f_decision or "شراء" in f_decision else ("#e74c3c" if "SELL" in f_decision or "بيع" in f_decision else "#f1c40f")
        bg_decision = "rgba(46, 204, 113, 0.15)" if "BUY" in f_decision or "شراء" in f_decision else ("rgba(231, 76, 60, 0.15)" if "SELL" in f_decision or "بيع" in f_decision else "rgba(241, 196, 15, 0.12)")

        # عرض درجات قوة الشراء والبيع (المرتبطة بفريم الحسابات والجدول)
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
                <span style="color: #bdc3c7; font-size: 0.85rem;">🎯 الإشارة الحالية (بناءً على فريم القوة المختار):</span>
                <h3 style="margin: 3px 0 0 0; color: {decision_color}; font-weight: bold;">{f_decision}</h3>
            </div>
        """, unsafe_allow_html=True)

        # 🔵 ثانياً: شريط التحكم المستقل كلياً بالشارت (لا يؤثر إطلاقاً على قيم الـ Score)
        st.write("")
        chart_time_choice = st.radio(
            "📅 اختر النطاق الزمني للشارت فقط:",
            ["يوم واحد (1D)", "أسبوع واحد (1W)", "شهر واحد (1M)", "سنة واحدة (1Y)", "5 سنوات (5Y)"],
            horizontal=True
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
        
        # رسم الشارت التفاعلي المستقل
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
                height=280, 
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(showgrid=False, color="white", tickformat='%m-%d' if "Y" in chart_time_choice else '%H:%M'),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="white", side="right"),
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ==================== المستشار التلقائي للأخبار الذكي ====================
            if use_gen_ai:
                if not API_KEY:
                    st.warning("⚠️ يرجى إدخال مفتاح الـ Gemini API أولاً في الإعدادات بالأعلى لتفعيل المستشار التلقائي.")
                else:
                    st.write("---")
                    st.subheader("📰 مستشار الذكاء الاصطناعي للأخبار")
                    with st.spinner("جاري سحب أحدث الأخبار وتلخيص نبرتها الاقتصادية..."):
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
                                    st.markdown(f"🔗 [{title}]({link})")
                            else:
                                news_context = "لا توجد أخبار اقتصادية عاجلة متداولة حالياً لهذا الرمز."
                                st.info("ℹ️ لا توجد مقالات إخبارية حيوية منشورة في الساعات الأخيرة.")
                            
                            if news_context:
                                prompt = (
                                    f"أنت مستشار مالي وخبير أسواق. قم بتحليل الأصل {target_clean} "
                                    f"بناءً على التقييمين الحاليين (قوة الشراء: {f_buy}%، قوة البيع: {f_sell}%) "
                                    f"وهذه الأخبار الاقتصادية:\n{news_context}\n\n"
                                    f"اكتب تلخيصاً دقيقاً في نقطتين سريعتين لمستخدم يتصفح من هاتفه المحمول."
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

# ==================== لوحة المراقبة المرتبطة بفريم القوة والجدول المختار ====================
st.write("---")
st.subheader("📋 لوحة المراقبة الشاملة للمحفظة")
st.caption(f"تتحث وتعمل تلقائياً بناءً على فريم القوة المختار بالأعلى: ({calculation_frame})")

if clean_symbols_list:
    watchlist_data = []
    
    with st.spinner("جاري تحديث لوحة المراقبة لجميع الأسهم..."):
        for sym in clean_symbols_list:
            sym_clean = str(sym).strip().upper()
            try:
                sym_df = fetch_clean_data(sym_clean, calc_period, calc_interval)
                if not sym_df.empty and len(sym_df) >= 10:
                    last_row = sym_df.iloc[-1]
                    price_now = float(last_row['Close'])
                    
                    buy_val, sell_val, decision_val = calculate_scores_and_decision(sym_df, sym_clean)
                    
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
        summary_df = pd.DataFrame(watchlist_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("أدخل الرموز في الإعدادات بالأعلى لملء لوحة المراقبة.")
