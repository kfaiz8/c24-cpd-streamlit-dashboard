import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CARS24 | Dealer Ticket Impact",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #F5F7FA; }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F0F4FF 100%);
    border-right: 1px solid #E2E8F0;
  }
  [data-testid="stSidebar"] label { color: #374151 !important; font-weight: 500; }
  .header-bar {
    background: linear-gradient(135deg, #FF5200 0%, #FF7A3D 100%);
    padding: 18px 28px; border-radius: 14px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 20px rgba(255,82,0,0.25);
  }
  .header-logo { font-size: 2.6rem; font-weight: 800; color: white; letter-spacing: -1px; }
  .header-logo span { color: #FFD580; }
  .header-subtitle { color: rgba(255,255,255,0.88); font-size: 0.95rem; font-weight: 400; margin-top: 2px; }
  .header-title { color: white; font-size: 1.45rem; font-weight: 700; }
  .section-title {
    font-size: 1.05rem; font-weight: 700; color: #1E293B;
    margin: 20px 0 10px 0; padding-left: 10px; border-left: 4px solid #FF5200;
  }
  .kpi-card {
    background: white; border-radius: 14px; padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border-top: 4px solid #FF5200; height: 100%;
  }
  .kpi-label { font-size: 0.78rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .kpi-value { font-size: 2rem; font-weight: 800; color: #1E293B; line-height: 1.1; }
  .kpi-delta-pos { font-size:0.85rem; color:#10B981; font-weight:600; }
  .kpi-delta-neg { font-size:0.85rem; color:#EF4444; font-weight:600; }
  .kpi-delta-neu { font-size:0.85rem; color:#64748B; font-weight:600; }
  .kpi-icon { font-size:1.6rem; float:right; opacity:0.18; margin-top:-4px; }
  .chart-box { background: white; border-radius: 14px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 16px; }
  .info-banner { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 18px; color: #1D4ED8; font-size: 0.88rem; margin-bottom: 14px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
  .pill-green  { background:#D1FAE5; color:#065F46; }
  .pill-red    { background:#FEE2E2; color:#991B1B; }
  .pill-orange { background:#FFEDD5; color:#9A3412; }
  .divider { border-top: 1px solid #E2E8F0; margin: 18px 0; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MONTH ORDER HELPER
# ─────────────────────────────────────────────
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

def month_sort_key(m):
    try:
        return MONTH_ORDER.index(m)
    except ValueError:
        return 99

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("CPD_data.csv", low_memory=False)
    df["BUCKET"] = df["BUCKET"].fillna("Unknown").astype(str)
    df["TICKET_SUB_CATEGORY"] = df["TICKET_SUB_CATEGORY"].fillna("Unknown").astype(str)
    df["MONTH_NUM"] = df["TICKET_RAISED_MONTH"].apply(month_sort_key)
    df["MONTH_DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH_NUM"].apply(lambda x: str(x+1).zfill(2)),
        format="%Y-%m", errors="coerce"
    )
    df["cpd_before"] = df[["CM-3","CM-2","CM-1"]].mean(axis=1)
    df["cpd_after"]  = df[["CM+1","CM+2","CM+3"]].mean(axis=1)
    df["cpd_change"] = df["cpd_after"] - df["cpd_before"]
    return df

df = load_data()

# ─────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────
ORANGE     = "#FF5200"
COLORS     = ["#FF5200","#3B82F6","#10B981","#F59E0B","#8B5CF6","#EC4899","#06B6D4","#84CC16","#6366F1"]
CHART_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#374151"

def style_fig(fig, title=""):
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        title=dict(
            text=title,
            y=0.99,                 # <-- move title upward
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=14, color="#1E293B")
        ),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#E2E8F0",
            font=dict(size=12, color="#1E293B")
        )
    )

    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div><div class="header-logo">CARS<span>24</span></div></div>
  <div style="border-left:1px solid rgba(255,255,255,0.3);height:52px;margin:0 4px;"></div>
  <div>
    <div class="header-title">Dealer Ticket Impact Dashboard</div>
    <div class="header-subtitle">Track CPD purchase trends before &amp; after dealer ticket resolution · with Sub-Category Intelligence</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    year_options = sorted(df["YEAR"].unique())
    year = st.selectbox("📅 Year", year_options, index=len(year_options)-1)

    months_in_year = sorted(
        df[df["YEAR"] == year]["TICKET_RAISED_MONTH"].unique(),
        key=month_sort_key
    )
    month = st.selectbox("🗓️ Ticket Raised Month", months_in_year)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    dealer_list = sorted(df["DEALER_CODE"].unique().astype(str))
    dealer_selection = st.multiselect(
        "🏪 Dealer Code",
        ["ALL DEALERS"] + dealer_list,
        default=["ALL DEALERS"],
        help="Select one or more dealers, or keep ALL DEALERS"
    )

    bucket_list = sorted(df["BUCKET"].unique())
    bucket_selection = st.multiselect(
        "🪣 Dealer Bucket",
        ["ALL BUCKETS"] + bucket_list,
        default=["ALL BUCKETS"],
        help="Dealer performance tier buckets"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Ticket Sub-Category Filter (dynamic - updates after year/month/dealer/bucket) ──
    pre_filter = df[(df["YEAR"] == year) & (df["TICKET_RAISED_MONTH"] == month)].copy()
    if "ALL DEALERS" not in dealer_selection and len(dealer_selection) > 0:
        pre_filter = pre_filter[pre_filter["DEALER_CODE"].astype(str).isin(dealer_selection)]
    if "ALL BUCKETS" not in bucket_selection and len(bucket_selection) > 0:
        pre_filter = pre_filter[pre_filter["BUCKET"].isin(bucket_selection)]

    sub_cat_list = sorted(pre_filter["TICKET_SUB_CATEGORY"].unique())
    sub_cat_selection = st.multiselect(
        "🏷️ Ticket Sub-Category",
        ["ALL CATEGORIES"] + sub_cat_list,
        default=["ALL CATEGORIES"],
        help="Filter by specific ticket issue type(s). Affects ALL charts and KPIs."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Aggregation (drives ALL charts & KPIs) ──
    agg_function = st.selectbox(
        "📐 CPD Aggregation",
        ["Mean (AVG)", "Sum", "Median", "Min", "Max"],
        help="Applied globally to every chart and KPI metric"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Display Options")
    show_raw_table    = st.toggle("Show Raw Data Table",       value=False)
    show_dealer_drill = st.toggle("Dealer Drill-Down",         value=False)
    show_heatmap      = st.toggle("Month × Bucket Heatmap",    value=True)
    show_subcat_deep  = st.toggle("Sub-Category Deep Dive",    value=True)

# ─────────────────────────────────────────────
# AGG HELPER  (used everywhere)
# ─────────────────────────────────────────────
AGG_KEY = agg_function.split(" ")[0]   # "Mean" | "Sum" | "Median" | "Min" | "Max"

AGG_MAP = {
    "Mean":   "mean",
    "Sum":    "sum",
    "Median": "median",
    "Min":    "min",
    "Max":    "max",
}
pandas_agg = AGG_MAP[AGG_KEY]

def aggregate(series):
    ops = {
        "Mean":   series.mean(),
        "Sum":    series.sum(),
        "Median": series.median(),
        "Min":    series.min(),
        "Max":    series.max(),
    }
    return ops[AGG_KEY]

# ─────────────────────────────────────────────
# FILTER DATA  (all filters including sub-category)
# ─────────────────────────────────────────────
filtered_data = df[
    (df["YEAR"] == year) &
    (df["TICKET_RAISED_MONTH"] == month)
].copy()

if "ALL DEALERS" not in dealer_selection and len(dealer_selection) > 0:
    filtered_data = filtered_data[
        filtered_data["DEALER_CODE"].astype(str).isin(dealer_selection)
    ]

if "ALL BUCKETS" not in bucket_selection and len(bucket_selection) > 0:
    filtered_data = filtered_data[
        filtered_data["BUCKET"].isin(bucket_selection)
    ]

if "ALL CATEGORIES" not in sub_cat_selection and len(sub_cat_selection) > 0:
    filtered_data = filtered_data[
        filtered_data["TICKET_SUB_CATEGORY"].isin(sub_cat_selection)
    ]

if filtered_data.empty:
    st.warning("⚠️ No data matches the current filters. Adjust the sidebar filters and try again.")
    st.stop()

# For sub-category section we also need dealer-level aggregated view
# (each dealer row deduplicated for CPD metrics, since tickets repeat per row)
dealer_agg = (
    filtered_data.drop_duplicates(subset=["DEALER_CODE"])
    [["DEALER_CODE","BUCKET","TICKET_CNT","cpd_before","CPD","cpd_after","cpd_change",
      "CM-3","CM-2","CM-1","CM+1","CM+2","CM+3"]]
    .copy()
)

# Sub-category counts per dealer (all tickets in filtered view)
subcat_counts = (
    filtered_data.groupby(["DEALER_CODE","TICKET_SUB_CATEGORY"])
    .size().reset_index(name="TICKET_COUNT")
)

# ─────────────────────────────────────────────
# Active sub-category label for display
# ─────────────────────────────────────────────
if "ALL CATEGORIES" in sub_cat_selection or len(sub_cat_selection) == 0:
    active_subcat_label = "All Sub-Categories"
else:
    active_subcat_label = ", ".join(sub_cat_selection)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

# KPIs based on dealer-level dedup (CPD metrics should not double-count)
total_tickets    = int(filtered_data["TICKET_ID"].nunique())
total_dealers    = filtered_data["DEALER_CODE"].nunique()
unique_subcats   = filtered_data["TICKET_SUB_CATEGORY"].nunique()

before_agg       = aggregate(dealer_agg["cpd_before"])
after_agg        = aggregate(dealer_agg["cpd_after"])
cpd_ticket_month = aggregate(dealer_agg["CPD"])
improvement      = after_agg - before_agg
pct_improvement  = (improvement / before_agg * 100) if before_agg else 0
positive_dealers = (dealer_agg["cpd_change"] > 0).sum()
pct_positive     = positive_dealers / len(dealer_agg) * 100 if len(dealer_agg) else 0

def delta_html(val, suffix=""):
    icon = "▲" if val > 0 else "▼"
    cls  = "kpi-delta-pos" if val > 0 else ("kpi-delta-neg" if val < 0 else "kpi-delta-neu")
    return f'<span class="{cls}">{icon} {abs(val):.2f}{suffix}</span>'

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">🎫</div>
      <div class="kpi-label">Total Tickets Raised</div>
      <div class="kpi-value">{total_tickets:,}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">{total_dealers} unique dealers</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:#8B5CF6;">
      <div class="kpi-icon">🏷️</div>
      <div class="kpi-label">Active Sub-Categories</div>
      <div class="kpi-value">{unique_subcats}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">issue types in view</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:#3B82F6;">
      <div class="kpi-icon">📦</div>
      <div class="kpi-label">{AGG_KEY} CPD Before Ticket</div>
      <div class="kpi-value">{before_agg:.2f}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">3-month avg (M-3 to M-1)</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:#10B981;">
      <div class="kpi-icon">🚗</div>
      <div class="kpi-label">{AGG_KEY} CPD Ticket Month</div>
      <div class="kpi-value">{cpd_ticket_month:.2f}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">month of ticket raise</div>
    </div>""", unsafe_allow_html=True)

with col5:
    sign       = "+" if improvement >= 0 else ""
    border_col = "#10B981" if improvement >= 0 else "#EF4444"
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:{border_col};">
      <div class="kpi-icon">📈</div>
      <div class="kpi-label">{AGG_KEY} CPD Improvement</div>
      <div class="kpi-value">{sign}{improvement:.2f}</div>
      {delta_html(pct_improvement, suffix="%")}
    </div>""", unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:#8B5CF6;">
      <div class="kpi-icon">✅</div>
      <div class="kpi-label">Dealers with +ve Impact</div>
      <div class="kpi-value">{positive_dealers}</div>
      <div class="kpi-delta-pos" style="font-size:0.82rem;margin-top:4px;">▲ {pct_positive:.1f}% of selected dealers</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1: CPD Trend + Before/After Bar
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🚗 CPD Purchase Trend — Ticket Window</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

PERIOD_COLS   = ["CM-3","CM-2","CM-1","CPD","CM+1","CM+2","CM+3"]
PERIOD_LABELS = ["M-3","M-2","M-1","Ticket Month","M+1","M+2","M+3"]

with col_left:
    values = [aggregate(dealer_agg[c] if c != "CPD" else dealer_agg["CPD"]) for c in PERIOD_COLS]

    point_colors = ["#3B82F6","#3B82F6","#3B82F6","#FF5200","#10B981","#10B981","#10B981"]

    fig_trend = go.Figure()
    fig_trend.add_vrect(x0=-0.5, x1=2.5,
        fillcolor="#EFF6FF", opacity=0.4, layer="below", line_width=0,
        annotation_text="Pre-Ticket", annotation_position="top left",
        annotation=dict(font_size=11, font_color="#3B82F6"))
    fig_trend.add_vrect(x0=3.5, x1=6.5,
        fillcolor="#F0FDF4", opacity=0.4, layer="below", line_width=0,
        annotation_text="Post-Ticket", annotation_position="top right",
        annotation=dict(font_size=11, font_color="#10B981"))

    fig_trend.add_trace(go.Scatter(
        x=PERIOD_LABELS, y=values,
        mode="lines+markers+text",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=11, color=point_colors, line=dict(color="white", width=2)),
        text=[f"{v:.1f}" for v in values],
        textposition="top center",
        textfont=dict(size=11, color="#1E293B"),
        hovertemplate="<b>%{x}</b><br>CPD: %{y:.2f}<extra></extra>"
    ))
    fig_trend.add_vline(
        x=3, line_dash="dash",
        line_color="#FF5200", line_width=1.5,
        annotation_text="Ticket Raised",
        annotation_position="top",
        annotation=dict(font_size=10, font_color="#FF5200")
    )

    style_fig(fig_trend, f"CPD Trend ({AGG_KEY}) — {month} {year} · {active_subcat_label}")
    fig_trend.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    ba_vals   = [before_agg, cpd_ticket_month, after_agg]
    ba_labels = [f"Before\n(M-3→M-1)", "Ticket\nMonth", f"After\n(M+1→M+3)"]
    ba_colors = ["#3B82F6", "#FF5200", "#10B981"]

    fig_ba = go.Figure(go.Bar(
        x=ba_labels, y=ba_vals,
        marker_color=ba_colors,
        text=[f"{v:.2f}" for v in ba_vals],
        textposition="outside",
        textfont=dict(size=13, color="#1E293B", weight="bold"),
        hovertemplate=f"<b>%{{x}}</b><br>{AGG_KEY} CPD: %{{y:.2f}}<extra></extra>",
        width=0.55
    ))
    style_fig(fig_ba, f"Before vs After ({AGG_KEY})")
    fig_ba.update_layout(height=340, showlegend=False)
    fig_ba.update_xaxes(tickfont=dict(size=11))
    st.plotly_chart(fig_ba, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 2: Bucket CPD Trend + Bucket Impact Bar
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🪣 Dealer Bucket Analysis</div>', unsafe_allow_html=True)

col_bl, col_br = st.columns([3, 2])

with col_bl:
    bucket_trend_df = (
        dealer_agg.groupby("BUCKET")[PERIOD_COLS]
        .agg(pandas_agg)
        .reset_index()
    )
    bucket_trend_df = bucket_trend_df.rename(columns={"CPD": "Ticket_Month"})

    period_order_renamed  = ["CM-3","CM-2","CM-1","Ticket_Month","CM+1","CM+2","CM+3"]
    period_label_map      = {
        "CM-3": "M-3", "CM-2": "M-2", "CM-1": "M-1",
        "Ticket_Month": "Ticket Month",
        "CM+1": "M+1", "CM+2": "M+2", "CM+3": "M+3"
    }

    bucket_trend_melt = bucket_trend_df.melt(
        id_vars="BUCKET",
        var_name="Period",
        value_name="CPD_Value"
    )
    bucket_trend_melt["Period"] = pd.Categorical(
        bucket_trend_melt["Period"], categories=period_order_renamed, ordered=True
    )
    bucket_trend_melt = bucket_trend_melt.sort_values("Period")
    bucket_trend_melt["Period_Label"] = bucket_trend_melt["Period"].map(period_label_map)

    fig_bt = px.line(
        bucket_trend_melt,
        x="Period_Label", y="CPD_Value",
        color="BUCKET",
        markers=True,
        color_discrete_sequence=COLORS,
        category_orders={"Period_Label": list(period_label_map.values())},
        labels={"CPD_Value": f"{AGG_KEY} CPD", "Period_Label": "Period"},
    )
    fig_bt.update_traces(line=dict(width=2), marker=dict(size=8))
    style_fig(fig_bt, f"Bucket-wise CPD Trend ({AGG_KEY}) Across Ticket Window")
    fig_bt.update_layout(height=360, legend_title_text="Bucket")
    st.plotly_chart(fig_bt, use_container_width=True)

with col_br:
    bucket_impact = (
        dealer_agg.groupby("BUCKET")["cpd_change"]
        .agg(pandas_agg)
        .reset_index()
        .sort_values("cpd_change", ascending=True)
    )
    bucket_impact["Color"] = bucket_impact["cpd_change"].apply(
        lambda x: "#10B981" if x >= 0 else "#EF4444"
    )
    bucket_impact["Label"] = bucket_impact["cpd_change"].apply(
        lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
    )

    fig_impact = go.Figure(go.Bar(
        x=bucket_impact["cpd_change"],
        y=bucket_impact["BUCKET"],
        orientation="h",
        marker_color=bucket_impact["Color"],
        text=bucket_impact["Label"],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{AGG_KEY} Change: %{{x:.2f}}<extra></extra>"
    ))
    fig_impact.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    style_fig(fig_impact, f"{AGG_KEY} CPD Change by Bucket (After − Before)")
    fig_impact.update_layout(height=360, showlegend=False)
    fig_impact.update_xaxes(title_text=f"{AGG_KEY} CPD Change")
    fig_impact.update_yaxes(title_text="")
    st.plotly_chart(fig_impact, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 3: Ticket Distribution + CPD Change Dist.
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🎫 Ticket & CPD Change Distribution</div>', unsafe_allow_html=True)

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    ticket_by_bucket = (
        filtered_data.groupby("BUCKET")["TICKET_ID"]
        .nunique().reset_index(name="TICKET_CNT")
    )
    fig_pie = px.pie(
        ticket_by_bucket, names="BUCKET", values="TICKET_CNT",
        color_discrete_sequence=COLORS, hole=0.45
    )
    fig_pie.update_traces(
        textposition="outside", textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Tickets: %{value:,}<br>%{percent}<extra></extra>"
    )
    style_fig(fig_pie, f"Ticket Share by Bucket")
    fig_pie.update_layout(height=320, showlegend=False, margin=dict(t=42,b=30,l=10,r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_t2:
    mean_change = aggregate(dealer_agg["cpd_change"])
    fig_hist = px.histogram(
        dealer_agg, x="cpd_change", nbins=30,
        color_discrete_sequence=[ORANGE],
        labels={"cpd_change": "CPD Change (After − Before)"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=1.5,
                       annotation_text="Break-even", annotation_position="top right",
                       annotation=dict(font_size=10, font_color="#EF4444"))
    fig_hist.add_vline(x=mean_change, line_dash="dot", line_color="#10B981", line_width=1.5,
                       annotation_text=f"{AGG_KEY}: {mean_change:.2f}",
                       annotation_position="top left",
                       annotation=dict(font_size=10, font_color="#10B981"))
    style_fig(fig_hist, "Distribution of CPD Change per Dealer")
    fig_hist.update_layout(height=320)
    fig_hist.update_yaxes(title_text="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_t3:
    fig_box = px.box(
        dealer_agg, x="BUCKET", y="TICKET_CNT",
        color="BUCKET", color_discrete_sequence=COLORS,
        points="outliers",
        labels={"TICKET_CNT": "Tickets per Dealer", "BUCKET": "Bucket"},
    )
    fig_box.update_traces(marker_size=4)
    style_fig(fig_box, "Ticket Count Distribution by Bucket")
    fig_box.update_layout(height=320, showlegend=False)
    fig_box.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 4: Monthly Ticket Trend (full dataset)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Monthly Ticket Volume Trend (All Periods)</div>', unsafe_allow_html=True)

# Apply sub-category filter to full dataset for trend consistency
df_trend_base = df.copy()
if "ALL CATEGORIES" not in sub_cat_selection and len(sub_cat_selection) > 0:
    df_trend_base = df_trend_base[df_trend_base["TICKET_SUB_CATEGORY"].isin(sub_cat_selection)]

monthly_trend = (
    df_trend_base.groupby(["YEAR","TICKET_RAISED_MONTH","MONTH_NUM"])["TICKET_ID"]
    .nunique().reset_index(name="TICKET_CNT")
    .sort_values(["YEAR","MONTH_NUM"])
)
monthly_trend["YearMonth"] = (
    monthly_trend["TICKET_RAISED_MONTH"] + " " + monthly_trend["YEAR"].astype(str)
)

fig_monthly = px.bar(
    monthly_trend, x="YearMonth", y="TICKET_CNT",
    color="YEAR", color_discrete_sequence=["#3B82F6","#FF5200"],
    barmode="group",
    labels={"TICKET_CNT": "Unique Tickets", "YearMonth": "Month-Year"},
    text="TICKET_CNT"
)
fig_monthly.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_size=10)

current_label = f"{month} {year}"
if current_label in monthly_trend["YearMonth"].values:
    x_idx = list(monthly_trend["YearMonth"]).index(current_label)
    fig_monthly.add_vline(
        x=x_idx,
        line_dash="dot", line_color=ORANGE, line_width=2,
        annotation_text="Selected", annotation_position="top",
        annotation=dict(font_color=ORANGE, font_size=10)
    )

style_fig(fig_monthly, f"Total Tickets per Month · {active_subcat_label}")
fig_monthly.update_layout(height=320, legend_title_text="Year")
fig_monthly.update_xaxes(tickangle=-35, tickfont=dict(size=10))
st.plotly_chart(fig_monthly, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  🏷️  TICKET SUB-CATEGORY INTELLIGENCE SECTION
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏷️ Ticket Sub-Category Intelligence</div>', unsafe_allow_html=True)

if "ALL CATEGORIES" not in sub_cat_selection and len(sub_cat_selection) > 0:
    st.markdown(
        f'<div class="info-banner">🔍 Sub-category filter active: <b>{active_subcat_label}</b> — all charts below reflect this selection.</div>',
        unsafe_allow_html=True
    )

# ── SC-1: Volume + CPD Impact side by side ──────────────────────
col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    # Ticket volume by sub-category (bar)
    subcat_vol = (
        filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
        .nunique().reset_index(name="TICKET_COUNT")
        .sort_values("TICKET_COUNT", ascending=True)
    )
    # Limit to top 15 for readability
    subcat_vol = subcat_vol.tail(15)
    subcat_vol["pct"] = subcat_vol["TICKET_COUNT"] / subcat_vol["TICKET_COUNT"].sum() * 100

    fig_scvol = go.Figure(go.Bar(
        x=subcat_vol["TICKET_COUNT"],
        y=subcat_vol["TICKET_SUB_CATEGORY"],
        orientation="h",
        marker=dict(
            color=subcat_vol["TICKET_COUNT"],
            colorscale=[[0,"#FFEDD5"],[0.5,"#FF7A3D"],[1,"#FF5200"]],
            showscale=False
        ),
        text=[f"{c:,} ({p:.1f}%)" for c,p in zip(subcat_vol["TICKET_COUNT"], subcat_vol["pct"])],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Tickets: %{x:,}<extra></extra>"
    ))
    style_fig(fig_scvol, "Top Sub-Categories by Ticket Volume")
    fig_scvol.update_layout(height=420, showlegend=False)
    fig_scvol.update_xaxes(title_text="Ticket Count")
    fig_scvol.update_yaxes(title_text="", tickfont=dict(size=10))
    st.plotly_chart(fig_scvol, use_container_width=True)

with col_sc2:
    # CPD change by sub-category — join ticket sub-cat back to dealer CPD
    # For each dealer, find their dominant sub-category by ticket count
    dealer_subcat_counts = (
        filtered_data.groupby(["DEALER_CODE","TICKET_SUB_CATEGORY"])
        .size().reset_index(name="cnt")
    )
    dealer_primary_subcat = (
        dealer_subcat_counts.sort_values("cnt", ascending=False)
        .drop_duplicates("DEALER_CODE")[["DEALER_CODE","TICKET_SUB_CATEGORY"]]
    )
    dealer_with_subcat = dealer_agg.merge(dealer_primary_subcat, on="DEALER_CODE", how="left")

    subcat_cpd = (
        dealer_with_subcat.groupby("TICKET_SUB_CATEGORY")["cpd_change"]
        .agg(pandas_agg)
        .reset_index()
        .sort_values("cpd_change", ascending=True)
        .tail(15)
    )
    subcat_cpd["Color"] = subcat_cpd["cpd_change"].apply(
        lambda x: "#10B981" if x >= 0 else "#EF4444"
    )
    subcat_cpd["Label"] = subcat_cpd["cpd_change"].apply(
        lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
    )

    fig_sccpd = go.Figure(go.Bar(
        x=subcat_cpd["cpd_change"],
        y=subcat_cpd["TICKET_SUB_CATEGORY"],
        orientation="h",
        marker_color=subcat_cpd["Color"],
        text=subcat_cpd["Label"],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate=f"<b>%{{y}}</b><br>{AGG_KEY} CPD Change: %{{x:.2f}}<extra></extra>"
    ))
    fig_sccpd.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    style_fig(fig_sccpd, f"{AGG_KEY} CPD Change by Sub-Category (After − Before)")
    fig_sccpd.update_layout(height=420, showlegend=False)
    fig_sccpd.update_xaxes(title_text=f"{AGG_KEY} CPD Change")
    fig_sccpd.update_yaxes(title_text="", tickfont=dict(size=10))
    st.plotly_chart(fig_sccpd, use_container_width=True)

# ── SC-2: Sub-Category CPD Trend (line) + Before/After bar ──────
st.markdown('<div class="section-title">📊 Sub-Category CPD Trend Across Ticket Window</div>', unsafe_allow_html=True)

col_sc3, col_sc4 = st.columns([3,2])

with col_sc3:
    # For CPD period metrics, join sub-category using dealer primary mapping
    dealer_with_subcat2 = dealer_agg.merge(dealer_primary_subcat, on="DEALER_CODE", how="left")
    dealer_with_subcat2["TICKET_SUB_CATEGORY"] = dealer_with_subcat2["TICKET_SUB_CATEGORY"].fillna("Unknown")

    # Pick top N sub-cats by ticket count for legibility
    top_subcats = (
        filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
        .nunique().nlargest(8).index.tolist()
    )
    sc_trend_base = dealer_with_subcat2[
        dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)
    ]

    sc_period_cols = ["CM-3","CM-2","CM-1","CM+1","CM+2","CM+3"]
    sc_period_display = ["M-3","M-2","M-1","M+1","M+2","M+3"]

    sc_trend_df = sc_trend_base.groupby("TICKET_SUB_CATEGORY")[sc_period_cols + ["CPD"]].agg(pandas_agg).reset_index()
    sc_trend_df = sc_trend_df.rename(columns={"CPD": "Ticket_Month"})
    all_sc_cols = ["CM-3","CM-2","CM-1","Ticket_Month","CM+1","CM+2","CM+3"]
    all_sc_labels = ["M-3","M-2","M-1","Ticket Month","M+1","M+2","M+3"]

    sc_melt = sc_trend_df.melt(id_vars="TICKET_SUB_CATEGORY", var_name="Period", value_name="CPD_Value")
    sc_melt["Period"] = pd.Categorical(sc_melt["Period"], categories=all_sc_cols, ordered=True)
    sc_melt = sc_melt.sort_values("Period")
    label_map2 = dict(zip(all_sc_cols, all_sc_labels))
    sc_melt["Period_Label"] = sc_melt["Period"].map(label_map2)

    fig_sctrend = px.line(
        sc_melt,
        x="Period_Label", y="CPD_Value",
        color="TICKET_SUB_CATEGORY",
        markers=True,
        color_discrete_sequence=COLORS,
        category_orders={"Period_Label": all_sc_labels},
        labels={"CPD_Value": f"{AGG_KEY} CPD", "Period_Label": "Period", "TICKET_SUB_CATEGORY": "Sub-Category"},
    )
    fig_sctrend.update_traces(line=dict(width=2), marker=dict(size=7))
    fig_sctrend.add_vline(x=3, line_dash="dash", line_color=ORANGE, line_width=1.5,
                          annotation_text="Ticket Raised",
                          annotation=dict(font_size=10, font_color=ORANGE))
    style_fig(fig_sctrend, f"CPD Trend by Sub-Category ({AGG_KEY}) — Top 8 by Volume")
    fig_sctrend.update_layout(height=380, legend_title_text="Sub-Category",
                               legend=dict(font=dict(size=9)))
    st.plotly_chart(fig_sctrend, use_container_width=True)

with col_sc4:
    # Before vs After grouped bar by sub-category
    sc_ba = dealer_with_subcat2[dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)]
    sc_ba_agg = sc_ba.groupby("TICKET_SUB_CATEGORY")[["cpd_before","cpd_after"]].agg(pandas_agg).reset_index()
    sc_ba_agg = sc_ba_agg.sort_values("cpd_after", ascending=False)

    fig_scba = go.Figure()
    fig_scba.add_trace(go.Bar(
        name="Before (M-3→M-1)",
        x=sc_ba_agg["TICKET_SUB_CATEGORY"],
        y=sc_ba_agg["cpd_before"],
        marker_color="#3B82F6",
        hovertemplate="<b>%{x}</b><br>Before: %{y:.2f}<extra></extra>"
    ))
    fig_scba.add_trace(go.Bar(
        name="After (M+1→M+3)",
        x=sc_ba_agg["TICKET_SUB_CATEGORY"],
        y=sc_ba_agg["cpd_after"],
        marker_color="#10B981",
        hovertemplate="<b>%{x}</b><br>After: %{y:.2f}<extra></extra>"
    ))
    fig_scba.update_layout(barmode="group")
    style_fig(fig_scba, f"{AGG_KEY} CPD Before vs After by Sub-Category")
    fig_scba.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_scba.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    st.plotly_chart(fig_scba, use_container_width=True)

# ── SC-3: Sub-Category × Bucket Heatmap ─────────────────────────
st.markdown('<div class="section-title">🌡️ Sub-Category × Bucket — CPD Change Heatmap</div>', unsafe_allow_html=True)

sc_bucket_df = dealer_with_subcat2[dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)]
sc_bucket_heat = (
    sc_bucket_df.groupby(["TICKET_SUB_CATEGORY","BUCKET"])["cpd_change"]
    .agg(pandas_agg).reset_index()
)
sc_bucket_pivot = sc_bucket_heat.pivot(
    index="TICKET_SUB_CATEGORY", columns="BUCKET", values="cpd_change"
)

fig_scheat = px.imshow(
    sc_bucket_pivot,
    color_continuous_scale="RdYlGn",
    aspect="auto",
    labels=dict(x="Bucket", y="Sub-Category", color=f"{AGG_KEY} CPD Δ"),
    text_auto=".2f"
)
style_fig(fig_scheat, f"{AGG_KEY} CPD Change: Sub-Category × Bucket — positive = dealers improved after ticket resolution")
fig_scheat.update_layout(height=380)
fig_scheat.update_xaxes(title_text="Dealer Bucket", tickangle=-20)
fig_scheat.update_yaxes(title_text="", tickfont=dict(size=10))
st.plotly_chart(fig_scheat, use_container_width=True)

# ── SC-4: Sub-Category Risk Table ───────────────────────────────
# st.markdown('<div class="section-title">⚠️ Sub-Category Risk & Opportunity Summary</div>', unsafe_allow_html=True)

# sc_summary = (
#     filtered_data.groupby("TICKET_SUB_CATEGORY")
#     .agg(
#         Unique_Tickets=("TICKET_ID","nunique"),
#         Affected_Dealers=("DEALER_CODE","nunique"),
#     ).reset_index()
# )
# # Merge CPD metrics via dealer primary subcat
# sc_cpd_metrics = (
#     dealer_with_subcat2.groupby("TICKET_SUB_CATEGORY")
#     .agg(
#         Avg_CPD_Before=("cpd_before", "mean"),
#         Avg_CPD_After=("cpd_after", "mean"),
#         Avg_CPD_Change=("cpd_change", "mean"),
#         Pct_Dealers_Improved=("cpd_change", lambda x: (x > 0).mean() * 100)
#     ).reset_index()
# )
# sc_summary = sc_summary.merge(sc_cpd_metrics, on="TICKET_SUB_CATEGORY", how="left")
# sc_summary = sc_summary.sort_values("Avg_CPD_Change", ascending=False)

# def color_change(val):
#     if pd.isna(val): return ""
#     if val > 0:   return "background-color:#D1FAE5;color:#065F46;font-weight:600"
#     if val < -1:  return "background-color:#FEE2E2;color:#991B1B;font-weight:600"
#     return "background-color:#FEF3C7;color:#92400E;font-weight:600"

# sc_display = sc_summary.rename(columns={
#     "TICKET_SUB_CATEGORY": "Sub-Category",
#     "Unique_Tickets": "Tickets",
#     "Affected_Dealers": "Dealers",
#     "Avg_CPD_Before": "Avg CPD Before",
#     "Avg_CPD_After": "Avg CPD After",
#     "Avg_CPD_Change": "Avg CPD Change",
#     "Pct_Dealers_Improved": "% Dealers Improved"
# }).round(2)

# st.dataframe(
#     sc_display.style
#         .applymap(color_change, subset=["Avg CPD Change"])
#         .format({
#             "Tickets": "{:,}",
#             "Dealers": "{:,}",
#             "Avg CPD Before": "{:.2f}",
#             "Avg CPD After": "{:.2f}",
#             "Avg CPD Change": "{:+.2f}",
#             "% Dealers Improved": "{:.1f}%"
#         }),
#     use_container_width=True,
#     hide_index=True,
#     height=min(40 + len(sc_display) * 35, 500)
# )

# ─────────────────────────────────────────────
# OPTIONAL: Heatmap — Month × Bucket
# ─────────────────────────────────────────────
if show_heatmap:
    st.markdown('<div class="section-title">🌡️ CPD Change Heatmap — Month × Bucket</div>', unsafe_allow_html=True)

    hmap_base = df_trend_base.copy()
    # Deduplicate by dealer per month for CPD metrics
    hmap_dedup = hmap_base.drop_duplicates(subset=["YEAR","TICKET_RAISED_MONTH","DEALER_CODE"])
    hmap_df = (
        hmap_dedup.groupby(["TICKET_RAISED_MONTH","BUCKET"])["cpd_change"]
        .agg(pandas_agg).reset_index()
    )
    hmap_pivot = hmap_df.pivot(
        index="BUCKET", columns="TICKET_RAISED_MONTH", values="cpd_change"
    )
    month_cols = [m for m in MONTH_ORDER if m in hmap_pivot.columns]
    hmap_pivot = hmap_pivot[month_cols]

    fig_hmap = px.imshow(
        hmap_pivot, color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(x="Month", y="Bucket", color=f"{AGG_KEY} CPD Δ"),
        text_auto=".2f"
    )
    style_fig(fig_hmap, f"{AGG_KEY} CPD Change (After − Before) by Month & Bucket · {active_subcat_label}")
    fig_hmap.update_layout(height=330)
    fig_hmap.update_xaxes(title_text="Month")
    fig_hmap.update_yaxes(title_text="")
    st.plotly_chart(fig_hmap, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL: Sub-Category Deep Dive
# ─────────────────────────────────────────────
if show_subcat_deep:
    st.markdown('<div class="section-title">🔬 Sub-Category Deep Dive — Month-over-Month Trend</div>', unsafe_allow_html=True)

    sc_mom = (
        df_trend_base.groupby(["YEAR","TICKET_RAISED_MONTH","MONTH_NUM","TICKET_SUB_CATEGORY"])["TICKET_ID"]
        .nunique().reset_index(name="TICKET_COUNT")
        .sort_values(["YEAR","MONTH_NUM"])
    )
    sc_mom["YearMonth"] = sc_mom["TICKET_RAISED_MONTH"] + " " + sc_mom["YEAR"].astype(str)

    # Pick top 6 sub-cats for MoM trend
    top6_sc = (
        filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
        .nunique().nlargest(6).index.tolist()
    )
    sc_mom_filtered = sc_mom[sc_mom["TICKET_SUB_CATEGORY"].isin(top6_sc)]

    fig_scmom = px.line(
        sc_mom_filtered,
        x="YearMonth", y="TICKET_COUNT",
        color="TICKET_SUB_CATEGORY",
        markers=True,
        color_discrete_sequence=COLORS,
        labels={"TICKET_COUNT": "Ticket Count", "YearMonth": "Month", "TICKET_SUB_CATEGORY": "Sub-Category"},
    )
    fig_scmom.update_traces(line=dict(width=2), marker=dict(size=7))
    style_fig(fig_scmom, "Month-over-Month Ticket Volume — Top 6 Sub-Categories")
    fig_scmom.update_layout(height=340, legend_title_text="Sub-Category", legend=dict(font=dict(size=10)))
    fig_scmom.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    st.plotly_chart(fig_scmom, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL: Dealer Drill-Down
# ─────────────────────────────────────────────
if show_dealer_drill:
    st.markdown('<div class="section-title">🔍 Dealer-Level Drill-Down</div>', unsafe_allow_html=True)

    # Attach primary sub-category to dealer view
    drill_df = dealer_agg.merge(dealer_primary_subcat, on="DEALER_CODE", how="left")
    drill_df = (
        drill_df[["DEALER_CODE","BUCKET","TICKET_SUB_CATEGORY","TICKET_CNT",
                  "cpd_before","CPD","cpd_after","cpd_change"]]
        .copy()
        .sort_values("cpd_change", ascending=False)
    )
    drill_df.columns = ["Dealer Code","Bucket","Primary Sub-Cat","Tickets",
                        "CPD Before","CPD Ticket Month","CPD After","CPD Change"]
    drill_df = drill_df.round(2)

    top_n = st.slider("Show Top / Bottom N Dealers", 5, 50, 15)
    tab_top, tab_bot, tab_search = st.tabs(["🏆 Top Improvers","⚠️ Bottom Performers","🔎 Search Dealer"])

    with tab_top:
        st.dataframe(
            drill_df.head(top_n).style
                .background_gradient(subset=["CPD Change"], cmap="Greens")
                .format({"Tickets": "{:,.0f}"}),
            use_container_width=True, hide_index=True
        )

    with tab_bot:
        st.dataframe(
            drill_df.tail(top_n).sort_values("CPD Change").style
                .background_gradient(subset=["CPD Change"], cmap="Reds_r")
                .format({"Tickets": "{:,.0f}"}),
            use_container_width=True, hide_index=True
        )

    with tab_search:
        search_code = st.text_input("Enter Dealer Code")
        if search_code:
            result = drill_df[drill_df["Dealer Code"].astype(str).str.contains(search_code)]
            if not result.empty:
                st.dataframe(result, use_container_width=True, hide_index=True)
                matched = dealer_agg[
                    dealer_agg["DEALER_CODE"].astype(str).str.contains(search_code)
                ]
                if not matched.empty:
                    spark_vals = matched[PERIOD_COLS].agg(pandas_agg)
                    fig_spark = go.Figure(go.Scatter(
                        x=PERIOD_LABELS, y=spark_vals.values,
                        mode="lines+markers",
                        line=dict(color=ORANGE, width=2),
                        marker=dict(size=9)
                    ))
                    style_fig(fig_spark, f"CPD Trend ({AGG_KEY}) for Dealer: {search_code}")
                    fig_spark.update_layout(height=260)
                    st.plotly_chart(fig_spark, use_container_width=True)

                    # Also show sub-categories for this dealer
                    dealer_sc = filtered_data[
                        filtered_data["DEALER_CODE"].astype(str).str.contains(search_code)
                    ][["TICKET_ID","TICKET_SUB_CATEGORY"]].drop_duplicates()
                    if not dealer_sc.empty:
                        st.caption(f"Tickets raised by dealer {search_code}:")
                        st.dataframe(dealer_sc, use_container_width=True, hide_index=True)
            else:
                st.info("No dealer found with that code.")

# ─────────────────────────────────────────────
# OPTIONAL: Raw Data Table
# ─────────────────────────────────────────────
if show_raw_table:
    st.markdown('<div class="section-title">📋 Raw Filtered Data</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered_data):,} records for **{month} {year}** · {active_subcat_label}")
    st.dataframe(
        filtered_data.drop(columns=["MONTH_NUM","MONTH_DATE"], errors="ignore"),
        use_container_width=True, height=320
    )

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">⬇️ Export</div>', unsafe_allow_html=True)

export_df = filtered_data.drop(columns=["MONTH_NUM","MONTH_DATE"], errors="ignore")
col_d1, col_d2 = st.columns([1, 4])
with col_d1:
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"CARS24_dealer_data_{month}_{year}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#94A3B8;font-size:0.8rem;margin-top:30px;
            border-top:1px solid #E2E8F0;padding-top:14px;">
  CARS24 Internal Analytics · Dealer Ticket Impact Dashboard · Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
