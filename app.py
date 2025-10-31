import time
import streamlit as st
import sqlite3
import pandas as pd
import altair as alt
import plotly.express as px
import pydeck as pdk

# Import your chatbot runner (uses your existing chatbot.py)
try:
    from chatbot import run_conversation_groq   # <- keep this as-is (your file is chatbot.py)
except ImportError:
    st.error("Error: Could not import run_conversation_groq from chatbot.py.")
    st.stop()

DB_FILE = "database.db"

# ---------- Page setup ----------
st.set_page_config(
    page_title="Project Samarth",
    page_icon="🇮🇳",
    layout="wide",
)

# ---------- Global styles  ----------
st.markdown("""
<style>
/* ---------------------------------
   GLOBAL SPACING & LAYOUT OPTIMIZATION 
   --- REMOVED ALL DEFAULT PADDING ---
   --------------------------------- */

/* Ensure the main app frame itself is flush */
.stApp {
    margin: 0 !important;
    padding: 0 !important;
}

/* Target Streamlit's main content container and remove default padding */
.main > div {
    padding-left: 0rem;
    padding-right: 0rem;
    padding-top: 0rem;
    padding-bottom: 0rem;
}

/* Ensure the prominent introductory banner is perfectly flush and attractive */
.prominent-intro {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    border-radius: 0 !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); /* Stronger shadow for depth */
    transition: all 0.3s ease;
}
.prominent-intro:hover {
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px); /* Subtle lift on hover */
}


/* Adjust Chat Input Bottom Padding to prevent overlap with fixed input */
/* This specific block is necessary to reserve space for the fixed chat input */
div[data-testid="stVerticalBlock"] > div:first-child {
    # padding-bottom: 120px; 
}

:root {
  --bg: #ffffff;
  --text: #1e40af;  /* Primary Dark Blue */
  --muted: #4b5563;  /* Darker gray for better readability */
  --primary: #1e40af; /* Primary Blue */
  --card: #f8fafc; /* Lighter gray for Explorer tab background */
  --success: #1e40af;
  --danger: #dc2626;
}

.app-title { 
  color: #047857;  /* Dark Green */
}

.app-subtitle, .sidebar-text {
  color: #4b5563;  /* Darker gray */
}

/* Chat bubbles - Tighter spacing (margin: 4px 0) */
.chat-bubble-user {
  background: #ecfdf5;  /* Light green background */
  color: #047857;  /* Darker green text */
  border: 1px solid rgba(16, 185, 129, 0.3);
}
.chat-bubble-bot {
  background: #f0fdf4; /* Very light green */
  color: #047857;
  border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Answer block (not used but kept for completeness) */
.answer-block {
  background: #047857;  
  color: #ffffff;  
}

/* Expanders / code */
.streamlit-expanderHeader {
  font-weight: 700 !important;
  color: #374151 !important; 
}
.streamlit-expanderContent {
  background: #f8fafc; 
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,.20);
  padding: 10px 12px;
}
code, pre {
  color: #374151 !important; 
  background: #f3f4f6 !important; 
  border: 1px solid rgba(148,163,184,.25) !important;
  border-radius: 10px !important;
}

/* Buttons (Green Gradient) */
button[kind="secondary"] {
  border-radius: 12px !important;
}
.stButton > button {
  background: linear-gradient(135deg, #10b981, #047857) !important; 
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 20px rgba(4, 120, 87, 0.35) !important; 
}
.stButton > button:hover {
  filter: brightness(1.04);
}

/* Input (chat) */
[data-testid="stChatInput"] > div {
  background: #ffffff; 
  border: 1px solid rgba(16, 185, 129, 0.30); 
  border-radius: 16px;
  box-shadow: 0 4px 14px rgba(2, 48, 32);
}
[data-testid="stChatInputTextArea"] textarea {
  color: #374151 !important; 
}

/* Footer note */
.footer-note { font-size: 0.9rem; color: #93a3b0; }

/* Subtle link color */
a, .markdown-text-container a { color: #047857; }
a:hover { color: #10b981; }

/* Sidebar card animations (unchanged) */
.sidebar-card {
  background: #ffffff;
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
  transition: all 0.2s ease;
}
.sidebar-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  background: #ecfdf5;
}
.sidebar-title {
  color: #047857;
  font-weight: 700;
  font-size: 1.1rem;
  margin-bottom: 8px;
  text-align: center;
}
.sidebar-text {
  color: #64748b;
  font-size: 0.95rem;
  text-align: center;
  line-height: 1.5;
}

/* Welcome container */
.welcome-container {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 24px;
  margin: 12px 0 24px;
  background: #ffffff;
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 16px;
  animation: fadeIn 0.5s ease-out;
}

/* Prominent Central Introduction */
.prominent-intro {
    background: #006400; /* Dark Green */
    /* Reduced top/bottom padding slightly for a tighter fit */
    padding: 40px 35px 30px 35px; 
    margin: 0;
    text-align: center;
    width: 100%;
    box-sizing: border-box;
}

.prominent-intro-title {
    color: #ffffff; /* White */
    font-size: 2.5rem; 
    font-weight: 900;
    margin-bottom: 15px;
    font-family: 'Times New Roman', serif;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
}
.prominent-intro-bullets {
    color: #ffffff;
    font-size: 1.2rem;
    font-weight: 600;
    display: flex;
    flex-direction: row;
    font-style: italic;    
    align-items: center;
    justify-content: center;    
    gap: 40px;
}
.prominent-intro-text {
    color: #ffffff; /* White */
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.5;
    font-family: 'Times New Roman', serif;
}

/* Chat input at bottom */
div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    padding: 12px 20px 20px 20px;
    z-index: 999;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    border-top: 1px solid #ccc; /* Added border for separation */
}

/* Reset button styling to match chat input */
.stButton > button[key="reset_chat"] {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(16, 185, 129, 0.30);
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    z-index: 1000;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: #047857;
}

/* Tab button styling */
.st-emotion-cache-1ftvsfn { 
    border-bottom: 3px solid #047857 !important;
    color: #047857 !important;
    font-weight: 700 !important;
}

.st-emotion-cache-1ftvsfn:hover { /* This class targets inactive tab button on hover */
    color: #10b981 !important;
    border-bottom: 3px solid #10b981 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- Prominent Central Introduction ----------
st.markdown("""
<div class="prominent-intro">
    <div class="prominent-intro-title">
        🇮🇳 Project Samarth
    </div>
    <div class="prominent-intro-text">
        Empowering India’s Agricultural Intelligence with Data and Climate Insight.<br> Project Samarth transforms vast government datasets into actionable insights that explain how weather trends and rainfall variability drive India’s agricultural economy.
        <div class="prominent-intro-bullets">
            <br>
            • Rainfall data: 1901–2015
            • Crop data: 2009–2015
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    pass

# ---------- Init session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: 'user'|'assistant', content: str}]

# ---------- Cached DB helpers ----------
@st.cache_data(show_spinner=False)
def load_states():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT DISTINCT State, StateNorm FROM rainfall ORDER BY State;", conn)
        conn.close()
        if df.empty:
            return []
        # present display names but also return normalized for queries
        return list(df.apply(lambda r: (r["State"], r["StateNorm"]), axis=1))
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def load_rainfall(state_norms, years):
    if not state_norms or not years:
        return pd.DataFrame()
    ph_s = ",".join("?" * len(state_norms))
    ph_y = ",".join("?" * len(years))
    q = f"""
        SELECT State, StateNorm, Year, Annual_Rainfall
        FROM rainfall
        WHERE StateNorm IN ({ph_s}) AND Year IN ({ph_y})
        ORDER BY State, Year;
    """
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(q, conn, params=tuple(state_norms + years))

@st.cache_data(show_spinner=False)
def load_crop_totals(state_norms, years):
    if not state_norms or not years:
        return pd.DataFrame()
    ph_s = ",".join("?" * len(state_norms))
    ph_y = ",".join("?" * len(years))
    q = f"""
        SELECT State, StateNorm, Year, SUM(ProductionMT) AS Total_Production
        FROM crop_production
        WHERE StateNorm IN ({ph_s}) AND Year IN ({ph_y})
        GROUP BY State, Year
        ORDER BY State, Year;
    """
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(q, conn, params=tuple(state_norms + years))

@st.cache_data(show_spinner=False)
def top_crops_for(state_norm, years, top_n=5):
    ph = ",".join("?" * len(years))
    q = f"""
        SELECT CropSimple AS Crop, SUM(ProductionMT) AS Total_Production
        FROM crop_production
        WHERE StateNorm = ? AND Year IN ({ph})
        GROUP BY CropSimple
        ORDER BY Total_Production DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(q, conn, params=(state_norm, *years, top_n))

# ---------- Small state centroid lookup for map views ----------
STATE_CENTROIDS = {
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Odisha": {"lat": 20.9517, "lon": 85.0985},
    # ...add others as needed...
}

def build_map_df(mean_df):
    # mean_df: DataFrame with columns ['State','Annual_Rainfall' or 'Total_Production']
    rows = []
    for _, r in mean_df.iterrows():
        st = r["State"]
        coords = STATE_CENTROIDS.get(st)
        if not coords:
            continue
        rows.append({"lat": coords["lat"], "lon": coords["lon"], "state": st, "value": r.iloc[1]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ---------- Tab: enhance UI with map, plotly, downloads, time-slider  ----------

tabs = st.tabs(["🔬 Visual Representation", "💬 Chatbot"])

# Custom Altair color scale 
MULTI_COLOR_SCALE = alt.Scale(
    range=['#047857', '#FBBF24', '#A78BFA', '#60A5FA', '#E5E7EB', '#374151'] 
)

# Plotly colors (equivalent to the Altair scale)
PLOTLY_COLORS = ['#047857', '#FBBF24', '#A78BFA', '#60A5FA', '#E5E7EB', '#374151']

# Visual Representation tab
with tabs[0]:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid rgba(16, 185, 129, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #047857; font-size: 1.1rem; font-weight: 700; margin-top: 0;">Integrated Dashboard for Data-Driven Insights</h3>
        <p style="color: #4b5563; font-size: 0.95rem; line-height: 1.6;">
            This interactive Analytical Insights Dashboard is designed to facilitate comparative analysis of climate and agricultural trends across India for easeness. The visualizations are inspired by <strong style="color: #047857;">official government project dashboards</strong>, ensuring clarity, accessibility, and high information density.
            To begin your exploration, utilize the controls below:
        </p>
        <ul style="color: #4b5563; font-size: 0.9rem; margin-top: 10px; padding-left: 20px;">
            <li style="margin-bottom: 5px;"><strong style="color: #10b981;">Select State(s) to Compare:</strong> Choose one or more States/Union Territories to instantly benchmark and compare their climate and production metrics side-by-side.</li>
            <li style="margin-bottom: 5px;"><strong style="color: #10b981;">Year Range Selection:</strong> Specify the temporal scope of your analysis. Note the distinct data availability periods: <span style="font-weight: 700;">Rainfall data spans 1901–2015</span>, and <span style="font-weight: 700;">Crop data is available for 2009–2015</span>.</li>
            <li style="margin-bottom: 5px;"><strong style="color: #10b981;">Mode of Data:</strong> Select the required analytical output: <strong>Rainfall</strong> (time-series), <strong>Crop totals</strong> (aggregated production), <strong>Top crops</strong> (principal production list), or <strong>Rainfall vs Crop correlation</strong> (statistical analysis).</li>
        </ul>
        <p style="color: #4b5563; font-size: 0.95rem; margin-top: 15px;">
            Click <strong>Generate</strong> after setting your parameters to view the corresponding time-series charts, production metrics, or statistical findings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    # --- End of Explanatory Text ---

    # load available states
    states_pairs = load_states()
    if not states_pairs:
        st.warning("No rainfall states found in DB. Please ensure samarth_data.db is populated.")
    else:
        display_to_norm = {d: n for d, n in states_pairs}
        display_names = list(display_to_norm.keys())


        # Controls flow directly. Streamlit's st.columns will manage width.
        cols = st.columns([2,1,1,1])
        with cols[0]:
            sel_states = st.multiselect("Select state(s) to compare", display_names, default=display_names[1:5])
        with cols[1]:
            year_min, year_max = st.slider("Year range (rainfall: 1901–2015, crop: 2009–2015)", 1901, 2015, (2009, 2015))
        with cols[2]:
            mode = st.selectbox("Mode of Data", ["Rainfall", "Crop totals", "Top crops", "Rainfall vs Crop correlation"])
        with cols[3]:
            apply = st.button("Generate", key="generate_explorer")
        

        if apply and sel_states:
            sel_norms = [display_to_norm[s] for s in sel_states]
            years = list(range(year_min, year_max+1))

            chart_lib = "Altair"

            # Rainfall
            if mode == "Rainfall":
                df_r = load_rainfall(sel_norms, years)
                if df_r.empty:
                    st.info("No rainfall rows for that selection.")
                else:
                    # Results flow directly, filling the width
                    st.download_button("Export rainfall CSV", df_r.to_csv(index=False), 
                                         file_name="rainfall_selection.csv", mime="text/csv")
                    
                    if chart_lib == "Altair":
                        chart = alt.Chart(df_r).mark_line(point=True).encode(
                            x="Year:O",
                            y=alt.Y("Annual_Rainfall:Q", title="Annual Rainfall (mm)"),
                            color=alt.Color("State:N", scale=MULTI_COLOR_SCALE), 
                            tooltip=["State", "Year", "Annual_Rainfall"]
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        fig = px.line(df_r, x="Year", y="Annual_Rainfall", 
                                    color="State", markers=True, title="Annual Rainfall",
                                    color_discrete_sequence=PLOTLY_COLORS) 
                        st.plotly_chart(fig, use_container_width=True)
                    st.write("Source: IMD Area Weighted Annual Rainfall (1901–2015)")
                    

            # Crop totals
            elif mode == "Crop totals":
                # clip years to crop data availability (2009–2015)
                crop_years = [y for y in years if 2009 <= y <= 2015]
                if not crop_years:
                    st.warning("Crop data only available for 2009–2015. Adjust your year range.")
                else:
                    df_c = load_crop_totals(sel_norms, crop_years)
                    if df_c.empty:
                        st.info("No crop totals for that selection.")
                    else:
                        # Results flow directly, filling the width
                        st.download_button("Export crop totals CSV", df_c.to_csv(index=False), file_name="crop_totals_selection.csv", mime="text/csv")
                        df_sum = df_c.groupby("State").Total_Production.sum().reset_index()
                        
                        # Use columns for metrics for better visual alignment
                        metric_cols = st.columns(len(df_sum))
                        for i, r in df_sum.iterrows():
                            with metric_cols[i % len(metric_cols)]: # Use modulo to cycle columns
                                st.markdown(f"""
                                    <div style="
                                        background: #fffff; 
                                        border: 1px solid #059669; 
                                        border-radius: 6px; 
                                        padding: 6px 8px; 
                                        margin: 3px 0;
                                        font-size: 0.85rem;
                                        max-width: 180px;
                                        color: #b58900;">
                                        <strong>{r['State']}</strong><br>
                                        <span style="font-size: 1.1rem; font-weight: bold; color: #047857;">{r['Total_Production']:,.0f} MT</span>
                                    </div>
                                """, unsafe_allow_html=True)

                        # Plotly option for stacked bars / small multiples
                        if chart_lib == "Altair":
                            chart = (
                                alt.Chart(df_c).mark_bar(size=12).encode(
                                x=alt.X("Year:O", title="Year"),
                                y=alt.Y("Total_Production:Q", title="Total Production (MT)"),
                                color=alt.Color("State:N", scale=MULTI_COLOR_SCALE), 
                                column="State:N",
                                tooltip=["State", "Year", "Total_Production"]
                            )
                            .properties(width=120, height=180)
                            )
                            st.altair_chart(chart, use_container_width=False)
                        else:
                            fig = px.bar(df_c, x="Year", y="Total_Production", color="State", barmode="group", 
                                         title="Total Production by State & Year",
                                         color_discrete_sequence=PLOTLY_COLORS, 
                                         width=600, height=350
                            )
                            fig.update_traces(marker_line_width=0.5)
                            st.plotly_chart(fig, use_container_width=True)
                        st.write("Source: State/UT-wise Production of Principal Crops (2009–2015)")
                        


            # Top crops
            elif mode == "Top crops":
                # Results flow directly, filling the width
                cols_top = st.columns([3, 1, 1])
                with cols_top[0]:
                    st.write("Top N crops")
                with cols_top[1]:
                    if st.button("-", key="top_n_minus"):
                        if "top_n" not in st.session_state:
                            st.session_state.top_n = 5
                        st.session_state.top_n = max(1, st.session_state.top_n - 1)
                with cols_top[2]:
                    if st.button("+", key="top_n_plus"):
                        if "top_n" not in st.session_state:
                            st.session_state.top_n = 5
                        st.session_state.top_n = min(20, st.session_state.top_n + 1)
                top_n = st.session_state.get("top_n", 5)
                st.write(f"**Current value: {top_n}**")

                crop_years = [y for y in years if 2009 <= y <= 2015]
                if not crop_years:
                    st.warning("Crop data only available for 2009–2015. Adjust your year range.")
                else:
                    all_top = []
                    st.write(f"Top {top_n} crops per selected state (aggregated {min(crop_years)}–{max(crop_years)})")

                    for s, norm in zip(sel_states, sel_norms):
                        df_top = top_crops_for(norm, crop_years, top_n)
                        if df_top.empty:
                            st.write(f"• {s}: No data")
                        else:
                            st.markdown(f"**🟢 {s}:**")
                            all_top.append(df_top.assign(State=s))
                            for idx, row in df_top.iterrows():
                                st.markdown(f"    <span style='color:#047857;'>{idx+1}. {row['Crop']}</span> — <strong style='color:#10b981;'>{row['Total_Production']:,.0f} MT</strong>", unsafe_allow_html=True)
                    if all_top:
                        big = pd.concat(all_top, ignore_index=True)
                        st.download_button("Export top-crops CSV", big.to_csv(index=False), file_name="top_crops.csv", mime="text/csv")
                

            # Correlation
            elif mode == "Rainfall vs Crop correlation":
                # Results flow directly, filling the width
                cols_corr = st.columns([3, 1, 1])
                with cols_corr[0]:
                    st.write("Last N years to analyze")
                with cols_corr[1]:
                    if st.button("-", key="last_n_minus"):
                        if "last_n" not in st.session_state:
                            st.session_state.last_n = 10
                        st.session_state.last_n = max(3, st.session_state.last_n - 1)
                with cols_corr[2]:
                    if st.button("+", key="last_n_plus"):
                        if "last_n" not in st.session_state:
                            st.session_state.last_n = 10
                        st.session_state.last_n = min(115, st.session_state.last_n + 1)
                last_n = st.session_state.get("last_n", 10)
                st.write(f"**Current value: {last_n}**")

                results = []
                charts = []
                for s, norm in zip(sel_states, sel_norms):
                    with sqlite3.connect(DB_FILE) as conn:
                        y_r = pd.read_sql_query("SELECT MAX(Year) AS y FROM rainfall WHERE StateNorm = ?", conn, params=(norm,))
                        y_c = pd.read_sql_query("SELECT MAX(Year) AS y FROM crop_production WHERE StateNorm = ?", conn, params=(norm,))
                        if y_r.empty or pd.isna(y_r.loc[0,"y"]) or y_c.empty or pd.isna(y_c.loc[0,"y"]):
                            results.append((s, None, "No data"))
                            continue
                        ymax = int(min(y_r.loc[0,"y"], y_c.loc[0,"y"]))
                        ymin = ymax - int(last_n) + 1
                        rain = pd.read_sql_query("SELECT Year, Annual_Rainfall AS rain FROM rainfall WHERE StateNorm = ? AND Year BETWEEN ? AND ? ORDER BY Year", conn, params=(norm, ymin, ymax))
                        crops = pd.read_sql_query("SELECT Year, SUM(ProductionMT) AS prod FROM crop_production WHERE StateNorm = ? AND Year BETWEEN ? AND ? GROUP BY Year ORDER BY Year", conn, params=(norm, ymin, ymax))
                    dfm = pd.merge(rain, crops, on="Year", how="inner").dropna()
                    if len(dfm) < 3:
                        results.append((s, None, f"Insufficient overlapping years ({ymin}–{ymax})"))
                        continue
                    r = float(dfm["rain"].corr(dfm["prod"]))
                    results.append((s, r, f"Years used: {dfm['Year'].min()}–{dfm['Year'].max()} (n={len(dfm)})"))
                    # plot scatter
                    if chart_lib == "Altair":
                        chart = alt.Chart(dfm).mark_point(filled=True, size=80, color="#10b981").encode( # Set point color to green
                            x=alt.X("rain:Q", title="Annual Rainfall (mm)"),
                            y=alt.Y("prod:Q", title="Total Production (MT)"),
                            tooltip=["Year", "rain", "prod"]
                        ).properties(title=f"{s}: r={r:.2f}")
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        fig = px.scatter(dfm, x="rain", y="prod", hover_data=["Year"], title=f"{s} (r={r:.2f})",
                                         color_discrete_sequence=['#10b981']) # Set scatter color to green
                        st.plotly_chart(fig, use_container_width=True)
                tbl = pd.DataFrame([{"State": s, "Pearson r": (r if r is not None else None), "Note": note} for (s, r, note) in results])
                st.table(tbl)
                st.download_button("Export correlation CSV", tbl.to_csv(index=False), file_name="correlation_summary.csv", mime="text/csv")


# Chatbot tab
with tabs[1]:
    # Check if any old prefill/submit state exists and clear it just in case
    if "_prefill" in st.session_state: del st.session_state["_prefill"]
    if "_submit_trigger" in st.session_state: del st.session_state["_submit_trigger"]
    
    # --- Chat History (Flows directly, spanning full width) ---
    for m in st.session_state.messages:
        if m["role"] == "user":
            # Right-aligned user message
            st.markdown(
                f"""
                <div style="text-align:right; margin:4px 0;">
                    <div style="
                        display:inline-block;
                        background:#10b981; /* Primary Green */
                        color:#ffffff;
                        padding:10px 14px;
                        border-radius:15px 15px 0px 15px;
                        max-width:75%;
                        word-wrap:break-word;
                        font-size:0.95rem;">
                        {m["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Left-aligned assistant message
            st.markdown(
                f"""
                <div style="text-align:left; margin:4px 0;">
                    <div style="
                        display:inline-block;
                        background:#f0fdf4; /* Very Light Green/White */
                        color:#047857; /* Dark Green text */
                        padding:10px 14px;
                        border-radius:15px 15px 15px 0px;
                        max-width:75%;
                        word-wrap:break-word;
                        font-size:0.95rem;">
                        {m["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    # --- Chat input row (fixed at bottom, outside the padding container) ---
    user_q = None
    
    # Normal chat input processing
    cols = st.columns([15, 1])
    with cols[0]:
        # Updated placeholder to include the required guidance
        user_q = st.chat_input(
            placeholder="Hey! Ask me about rainfall patterns, crop production, or agricultural trends...",
            key="chat_input",
        )
    with cols[1]:
        # Button uses fixed position CSS defined in the main style block
        if st.button("↺", key="reset_chat", help="Reset chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if user_q:
        # Right-aligned user bubble
        st.markdown(
            f"""
            <div style="text-align:right; margin:4px 0;">
                <div style="
                    display:inline-block;
                    background:#10b981; 
                    color:#ffffff;
                    padding:10px 14px;
                    border-radius:15px 15px 0px 15px;
                    max-width:75%;
                    word-wrap:break-word;
                    font-size:0.95rem;">
                    {user_q}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.spinner("Thinking…"):
            bot_text = run_conversation_groq(user_q)

        # Left-aligned bot bubble
        st.markdown(
            f"""
            <div style="text-align:left; margin:4px 0;">
                <div style="
                    display:inline-block;
                    background:#f0fdf4; 
                    color:#047857; 
                    padding:10px 14px;
                    border-radius:15px 15px 15px 0px;
                    max-width:75%;
                    word-wrap:break-word;
                    font-size:0.95rem;">
                    {bot_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.messages.append({"role": "assistant", "content": bot_text})
        st.rerun() # Rerun after new message is added to state