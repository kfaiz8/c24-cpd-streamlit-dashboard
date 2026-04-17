import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
  .info-banner { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 18px; color: #1D4ED8; font-size: 0.88rem; margin-bottom: 14px; }
  .warn-banner { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 12px 18px; color: #92400E; font-size: 0.88rem; margin-bottom: 14px; }
  .divider { border-top: 1px solid #E2E8F0; margin: 18px 0; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

ORANGE     = "#FF5200"
COLORS     = ["#FF5200","#3B82F6","#10B981","#F59E0B","#8B5CF6",
               "#EC4899","#06B6D4","#84CC16","#6366F1"]
CHART_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#374151"

WINDOW_COLS  = ["D_CM-3","D_CM-2","D_CM-1","D_CM+1","D_CM+2","D_CM+3"]
PERIOD_COLS  = ["D_CM-3","D_CM-2","D_CM-1","CPD","D_CM+1","D_CM+2","D_CM+3"]
PERIOD_LABELS = ["M-3","M-2","M-1","Ticket Month","M+1","M+2","M+3"]
PERIOD_LABEL_MAP = {
    "D_CM-3":"M-3","D_CM-2":"M-2","D_CM-1":"M-1",
    "CPD":"Ticket Month",
    "D_CM+1":"M+1","D_CM+2":"M+2","D_CM+3":"M+3"
}

def month_sort_key(m):
    try:
        return MONTH_ORDER.index(m)
    except ValueError:
        return 99

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("CPD_data.csv", low_memory=False)

    # ── Basic cleanup ──────────────────────────────────────────────
    df["BUCKET"] = df["BUCKET"].fillna("Unknown").astype(str)
    df["TICKET_SUB_CATEGORY"] = df["TICKET_SUB_CATEGORY"].fillna("Unknown").astype(str)
    df["MONTH_NUM"] = df["TICKET_RAISED_MONTH"].apply(month_sort_key)
    df["MONTH_DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" +
        df["MONTH_NUM"].apply(lambda x: str(x + 1).zfill(2)),
        format="%Y-%m", errors="coerce"
    )

    # ── Step 1: dealer-month CPD master (one row per dealer × month) ─
    dealer_month_cpd = (
        df.groupby(["DEALER_CODE", "MONTH_DATE"])["CPD"]
        .first()
        .reset_index()
        .sort_values(["DEALER_CODE", "MONTH_DATE"])
    )

    # ── Step 2: compute windows via shift on the FULL panel ─────────
    grp = dealer_month_cpd.groupby("DEALER_CODE")["CPD"]
    for i in range(1, 4):
        dealer_month_cpd[f"D_CM-{i}"] = grp.shift(i)
        dealer_month_cpd[f"D_CM+{i}"] = grp.shift(-i)

    # ── Step 3: derive before / after / change on master ────────────
    before_cols = ["D_CM-3","D_CM-2","D_CM-1"]
    after_cols  = ["D_CM+1","D_CM+2","D_CM+3"]
    dealer_month_cpd["cpd_before"] = dealer_month_cpd[before_cols].mean(axis=1)
    dealer_month_cpd["cpd_after"]  = dealer_month_cpd[after_cols].mean(axis=1)
    dealer_month_cpd["cpd_change"] = (
        dealer_month_cpd["cpd_after"] - dealer_month_cpd["cpd_before"]
    )

    # ── Step 4: preserve source CM columns as fallback, rename them ──
    # ROOT CAUSE FIX: The source CSV already contains correct CM-3…CM+3
    # values for every month including Aug 2025 (the earliest month).
    # The shift-based recomputation produces NaN for Aug 2025's pre-ticket
    # columns (D_CM-1, D_CM-2, D_CM-3) because there are no earlier rows
    # in the dataset to shift from — even though the data exists in the
    # source file. Previously, the code dropped CM-3…CM+3 before merging,
    # discarding this information entirely.
    #
    # Fix: rename the source columns to _src_ variants instead of dropping
    # them. After merging the shift-computed D_CM± columns, we fill any
    # NaN D_CM± values with the corresponding source column values.
    # This restores Aug 2025 (and any future earliest-month edge cases)
    # without affecting any other month where the shift works correctly.
    src_rename = {}
    for i in range(1, 4):
        if f"CM-{i}" in df.columns:
            src_rename[f"CM-{i}"] = f"_src_CM-{i}"
        if f"CM+{i}" in df.columns:
            src_rename[f"CM+{i}"] = f"_src_CM+{i}"
    df = df.rename(columns=src_rename)

    # ── Step 5: merge window columns onto ticket-level rows ──────────
    dynamic_cols = (
        ["DEALER_CODE","MONTH_DATE"] +
        [f"D_CM-{i}" for i in range(1, 4)] +
        [f"D_CM+{i}" for i in range(1, 4)] +
        ["cpd_before","cpd_after","cpd_change"]
    )
    df = df.merge(
        dealer_month_cpd[dynamic_cols],
        on=["DEALER_CODE","MONTH_DATE"],
        how="left"
    )

    # ── Fallback: fill NaN D_CM± with source CSV values ─────────────
    # For months where shift() couldn't look back/forward far enough
    # (e.g. Aug 2025 has no prior months in the dataset), the source
    # CSV's CM columns contain the real values — use them to fill gaps.
    for i in range(1, 4):
        src_pre  = f"_src_CM-{i}"
        src_post = f"_src_CM+{i}"
        if src_pre in df.columns:
            df[f"D_CM-{i}"] = df[f"D_CM-{i}"].fillna(df[src_pre])
        if src_post in df.columns:
            df[f"D_CM+{i}"] = df[f"D_CM+{i}"].fillna(df[src_post])

    # Drop temporary source columns — no longer needed downstream
    src_cols_to_drop = [c for c in df.columns if c.startswith("_src_CM")]
    df = df.drop(columns=src_cols_to_drop)

    return df, dealer_month_cpd


df, dealer_month_cpd = load_data()

# ─────────────────────────────────────────────
# PLOTLY STYLE HELPER
# ─────────────────────────────────────────────
def style_fig(fig, title=""):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        title=dict(text=title, y=0.99, x=0.5, xanchor="center", yanchor="top",
                   font=dict(size=14, color="#1E293B")),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0",
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0",
                        font=dict(size=12, color="#1E293B"))
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
        "🏪 Dealer Code", ["ALL DEALERS"] + dealer_list,
        default=["ALL DEALERS"], help="Select one or more dealers, or keep ALL DEALERS"
    )

    bucket_list = sorted(df["BUCKET"].unique())
    bucket_selection = st.multiselect(
        "🪣 Dealer Bucket", ["ALL BUCKETS"] + bucket_list,
        default=["ALL BUCKETS"], help="Dealer performance tier buckets"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    pre_filter = df[(df["YEAR"] == year) & (df["TICKET_RAISED_MONTH"] == month)].copy()
    if "ALL DEALERS" not in dealer_selection and dealer_selection:
        pre_filter = pre_filter[pre_filter["DEALER_CODE"].astype(str).isin(dealer_selection)]
    if "ALL BUCKETS" not in bucket_selection and bucket_selection:
        pre_filter = pre_filter[pre_filter["BUCKET"].isin(bucket_selection)]

    sub_cat_list = sorted(pre_filter["TICKET_SUB_CATEGORY"].unique())
    sub_cat_selection = st.multiselect(
        "🏷️ Ticket Sub-Category", ["ALL CATEGORIES"] + sub_cat_list,
        default=["ALL CATEGORIES"], help="Filter by specific ticket issue type(s)."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    agg_function = st.selectbox(
        "📐 CPD Aggregation",
        ["Mean (AVG)", "Sum", "Median", "Min", "Max"],
        help="Applied globally to every chart and KPI metric"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Display Options")
    show_raw_table    = st.toggle("Show Raw Data Table",    value=False)
    show_dealer_drill = st.toggle("Dealer Drill-Down",      value=False)
    show_heatmap      = st.toggle("Month × Bucket Heatmap", value=True)
    show_subcat_deep  = st.toggle("Sub-Category Deep Dive", value=True)

# ─────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────
AGG_KEY    = agg_function.split(" ")[0]
AGG_MAP    = {"Mean":"mean","Sum":"sum","Median":"median","Min":"min","Max":"max"}
pandas_agg = AGG_MAP[AGG_KEY]


def aggregate(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return {
        "Mean":   series.mean(),
        "Sum":    series.sum(min_count=1),
        "Median": series.median(),
        "Min":    series.min(),
        "Max":    series.max(),
    }[AGG_KEY]


def fmt_val(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:,.{decimals}f}"


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
filtered_data = df[
    (df["YEAR"] == year) & (df["TICKET_RAISED_MONTH"] == month)
].copy()

if "ALL DEALERS" not in dealer_selection and dealer_selection:
    filtered_data = filtered_data[
        filtered_data["DEALER_CODE"].astype(str).isin(dealer_selection)
    ]
if "ALL BUCKETS" not in bucket_selection and bucket_selection:
    filtered_data = filtered_data[filtered_data["BUCKET"].isin(bucket_selection)]
if "ALL CATEGORIES" not in sub_cat_selection and sub_cat_selection:
    filtered_data = filtered_data[
        filtered_data["TICKET_SUB_CATEGORY"].isin(sub_cat_selection)
    ]

if filtered_data.empty:
    st.warning("⚠️ No data matches the current filters. Adjust the sidebar filters and try again.")
    st.stop()

# ─────────────────────────────────────────────
# BUILD DEALER_AGG — one row per dealer
# ─────────────────────────────────────────────
dealer_cpd_agg = (
    filtered_data
    .groupby("DEALER_CODE")[WINDOW_COLS + ["CPD"]]
    .mean()
    .reset_index()
)
dealer_meta = (
    filtered_data.groupby("DEALER_CODE")
    .agg(BUCKET=("BUCKET","first"), TICKET_CNT=("TICKET_ID","nunique"))
    .reset_index()
)
dealer_agg = dealer_meta.merge(dealer_cpd_agg, on="DEALER_CODE", how="left")

dealer_agg["cpd_before"] = dealer_agg[["D_CM-3","D_CM-2","D_CM-1"]].mean(axis=1)
dealer_agg["cpd_after"]  = dealer_agg[["D_CM+1","D_CM+2","D_CM+3"]].mean(axis=1)
dealer_agg["cpd_change"] = dealer_agg["cpd_after"] - dealer_agg["cpd_before"]

n_pre_missing = (dealer_agg[["D_CM-3","D_CM-2","D_CM-1"]].isna().all(axis=1)).sum()
pre_data_missing = n_pre_missing == len(dealer_agg)

dealer_primary_subcat = (
    filtered_data.groupby(["DEALER_CODE","TICKET_SUB_CATEGORY"])
    .size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False)
    .drop_duplicates("DEALER_CODE")[["DEALER_CODE","TICKET_SUB_CATEGORY"]]
)

active_subcat_label = (
    "All Sub-Categories"
    if ("ALL CATEGORIES" in sub_cat_selection or not sub_cat_selection)
    else ", ".join(sub_cat_selection)
)

# ─────────────────────────────────────────────
# COHORT INFO BANNER
# ─────────────────────────────────────────────
cohort_size = dealer_agg["DEALER_CODE"].nunique()
with_pre  = int((dealer_agg["D_CM-1"].notna()).sum())
with_post = int((dealer_agg["D_CM+1"].notna()).sum())

if pre_data_missing:
    st.markdown(
        f'<div class="warn-banner">⚠️ <b>No pre-ticket history available</b> for '
        f'{month} {year}. This is the earliest month in the dataset — CM-3, CM-2, '
        f'CM-1 are all N/A for this cohort. Pre-ticket KPIs will show N/A.</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

total_tickets    = int(filtered_data["TICKET_ID"].nunique())
total_dealers    = cohort_size
unique_subcats   = filtered_data["TICKET_SUB_CATEGORY"].nunique()
before_agg       = aggregate(dealer_agg["cpd_before"])
after_agg        = aggregate(dealer_agg["cpd_after"])
cpd_ticket_month = aggregate(dealer_agg["CPD"])

if np.isnan(before_agg):
    improvement     = float("nan")
    pct_improvement = float("nan")
else:
    improvement     = (after_agg - before_agg) if not np.isnan(after_agg) else float("nan")
    pct_improvement = (improvement / before_agg * 100) if before_agg else float("nan")

valid_change     = dealer_agg["cpd_change"].dropna()
positive_dealers = int((valid_change > 0).sum())
pct_positive     = positive_dealers / len(valid_change) * 100 if len(valid_change) else 0


def delta_html(val, suffix=""):
    if val is None or np.isnan(val):
        return '<span class="kpi-delta-neu">— N/A</span>'
    icon = "▲" if val > 0 else "▼"
    cls  = "kpi-delta-pos" if val > 0 else ("kpi-delta-neg" if val < 0 else "kpi-delta-neu")
    return f'<span class="{cls}">{icon} {abs(val):.2f}{suffix}</span>'


c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-icon">🎫</div>
      <div class="kpi-label">Total Tickets Raised</div>
      <div class="kpi-value">{total_tickets:,}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">{total_dealers} unique dealers</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card" style="border-top-color:#8B5CF6;">
      <div class="kpi-icon">🏷️</div>
      <div class="kpi-label">Active Sub-Categories</div>
      <div class="kpi-value">{unique_subcats}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">issue types in view</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card" style="border-top-color:#3B82F6;">
      <div class="kpi-icon">📦</div>
      <div class="kpi-label">{AGG_KEY} CPD Before Ticket</div>
      <div class="kpi-value">{fmt_val(before_agg)}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">3-month avg (M-3 to M-1) · {with_pre} dealers w/ data</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card" style="border-top-color:#10B981;">
      <div class="kpi-icon">🚗</div>
      <div class="kpi-label">{AGG_KEY} CPD Ticket Month</div>
      <div class="kpi-value">{fmt_val(cpd_ticket_month)}</div>
      <div class="kpi-delta-neu" style="font-size:0.82rem;margin-top:4px;">month of ticket raise</div>
    </div>""", unsafe_allow_html=True)
with c5:
    sign = "+" if (not np.isnan(improvement) and improvement >= 0) else ""
    bc   = "#94A3B8" if np.isnan(improvement) else ("#10B981" if improvement >= 0 else "#EF4444")
    st.markdown(f"""<div class="kpi-card" style="border-top-color:{bc};">
      <div class="kpi-icon">📈</div>
      <div class="kpi-label">{AGG_KEY} CPD Improvement</div>
      <div class="kpi-value">{sign + fmt_val(improvement) if not np.isnan(improvement) else 'N/A'}</div>
      {delta_html(pct_improvement, suffix="%")}
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""<div class="kpi-card" style="border-top-color:#8B5CF6;">
      <div class="kpi-icon">✅</div>
      <div class="kpi-label">Dealers with +ve Impact</div>
      <div class="kpi-value">{positive_dealers}</div>
      <div class="kpi-delta-pos" style="font-size:0.82rem;margin-top:4px;">▲ {pct_positive:.1f}% of dealers w/ valid change</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1 — CPD Trend + Before/After Bar
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🚗 CPD Purchase Trend — Ticket Window</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="info-banner">📊 Showing CPD window for <b>{cohort_size} dealers</b> '
    f'who raised tickets in <b>{month} {year}</b>. '
    f'Window values (D_CM±) are drawn from each dealer\'s full time-series computed '
    f'before filtering — they are not re-aggregated from filtered data. '
    f'Cross-month asymmetry (e.g. Jan CM+1 ≠ Feb CM-1) is expected: different months '
    f'contain different dealer cohorts.</div>',
    unsafe_allow_html=True
)

col_left, col_right = st.columns([2, 1])

with col_left:
    values = [aggregate(dealer_agg[c]) for c in PERIOD_COLS]
    point_colors = ["#3B82F6","#3B82F6","#3B82F6","#FF5200","#10B981","#10B981","#10B981"]

    text_labels = [fmt_val(v, 1) for v in values]

    fig_trend = go.Figure()
    fig_trend.add_vrect(x0=-0.5, x1=2.5, fillcolor="#EFF6FF", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="Pre-Ticket", annotation_position="top left",
        annotation=dict(font_size=11, font_color="#3B82F6"))
    fig_trend.add_vrect(x0=3.5, x1=6.5, fillcolor="#F0FDF4", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="Post-Ticket", annotation_position="top right",
        annotation=dict(font_size=11, font_color="#10B981"))

    plot_y = [v if not np.isnan(v) else None for v in values]

    fig_trend.add_trace(go.Scatter(
        x=PERIOD_LABELS, y=plot_y,
        mode="lines+markers+text",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=11, color=point_colors, line=dict(color="white", width=2)),
        text=text_labels,
        textposition="top center",
        textfont=dict(size=11, color="#1E293B"),
        hovertemplate="<b>%{x}</b><br>CPD: %{y:.2f}<extra></extra>",
        connectgaps=False
    ))
    fig_trend.add_vline(x=3, line_dash="dash", line_color="#FF5200", line_width=1.5,
        annotation_text="Ticket Raised", annotation_position="top",
        annotation=dict(font_size=10, font_color="#FF5200"))
    style_fig(fig_trend, f"CPD Trend ({AGG_KEY}) — {month} {year} · {active_subcat_label}")
    fig_trend.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    ba_vals   = [before_agg, cpd_ticket_month, after_agg]
    ba_labels = ["Before\n(M-3→M-1)", "Ticket\nMonth", "After\n(M+1→M+3)"]
    ba_colors = ["#3B82F6", "#FF5200", "#10B981"]
    ba_text   = [fmt_val(v) for v in ba_vals]
    ba_plot   = [v if not np.isnan(v) else None for v in ba_vals]

    fig_ba = go.Figure(go.Bar(
        x=ba_labels, y=ba_plot,
        marker_color=ba_colors,
        text=ba_text, textposition="outside",
        textfont=dict(size=13, color="#1E293B"),
        hovertemplate=f"<b>%{{x}}</b><br>{AGG_KEY} CPD: %{{y:.2f}}<extra></extra>",
        width=0.55
    ))
    style_fig(fig_ba, f"Before vs After ({AGG_KEY})")
    fig_ba.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig_ba, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 2 — Bucket CPD Trend + Impact Bar
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🪣 Dealer Bucket Analysis</div>', unsafe_allow_html=True)
col_bl, col_br = st.columns([3, 2])

BUCKET_PERIOD_COLS = ["D_CM-3","D_CM-2","D_CM-1","CPD","D_CM+1","D_CM+2","D_CM+3"]

with col_bl:
    bucket_trend_rows = []
    for bucket, grp_df in dealer_agg.groupby("BUCKET"):
        row = {"BUCKET": bucket}
        for col in BUCKET_PERIOD_COLS:
            row[col] = aggregate(grp_df[col])
        bucket_trend_rows.append(row)
    bucket_trend_df = pd.DataFrame(bucket_trend_rows)

    bucket_trend_melt = bucket_trend_df.melt(
        id_vars="BUCKET", var_name="Period", value_name="CPD_Value"
    )
    bucket_trend_melt["Period"] = pd.Categorical(
        bucket_trend_melt["Period"], categories=BUCKET_PERIOD_COLS, ordered=True
    )
    bucket_trend_melt = bucket_trend_melt.sort_values("Period")
    bucket_trend_melt["Period_Label"] = bucket_trend_melt["Period"].map(PERIOD_LABEL_MAP)

    fig_bt = px.line(
        bucket_trend_melt, x="Period_Label", y="CPD_Value",
        color="BUCKET", markers=True,
        color_discrete_sequence=COLORS,
        category_orders={"Period_Label": list(PERIOD_LABEL_MAP.values())},
        labels={"CPD_Value": f"{AGG_KEY} CPD", "Period_Label": "Period"},
    )
    fig_bt.update_traces(line=dict(width=2), marker=dict(size=8),
                         connectgaps=False)
    style_fig(fig_bt, f"Bucket-wise CPD Trend ({AGG_KEY}) Across Ticket Window")
    fig_bt.update_layout(height=360, legend_title_text="Bucket")
    st.plotly_chart(fig_bt, use_container_width=True)

with col_br:
    bucket_impact_rows = []
    for bucket, grp_df in dealer_agg.groupby("BUCKET"):
        val = aggregate(grp_df["cpd_change"])
        bucket_impact_rows.append({"BUCKET": bucket, "cpd_change": val})
    bucket_impact = (
        pd.DataFrame(bucket_impact_rows)
        .dropna(subset=["cpd_change"])
        .sort_values("cpd_change", ascending=True)
    )
    bucket_impact["Color"] = bucket_impact["cpd_change"].apply(
        lambda x: "#10B981" if x >= 0 else "#EF4444"
    )
    bucket_impact["Label"] = bucket_impact["cpd_change"].apply(
        lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
    )
    fig_impact = go.Figure(go.Bar(
        x=bucket_impact["cpd_change"], y=bucket_impact["BUCKET"],
        orientation="h",
        marker_color=bucket_impact["Color"],
        text=bucket_impact["Label"], textposition="outside",
        textfont=dict(size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{AGG_KEY} Change: %{{x:.2f}}<extra></extra>"
    ))
    fig_impact.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    style_fig(fig_impact, f"{AGG_KEY} CPD Change by Bucket (After − Before)")
    fig_impact.update_layout(height=360, showlegend=False)
    fig_impact.update_xaxes(title_text=f"{AGG_KEY} CPD Change")
    st.plotly_chart(fig_impact, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 3 — Distributions
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🎫 Ticket & CPD Change Distribution</div>', unsafe_allow_html=True)
ct1, ct2, ct3 = st.columns(3)

with ct1:
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
    style_fig(fig_pie, "Ticket Share by Bucket")
    fig_pie.update_layout(height=320, showlegend=False, margin=dict(t=42,b=30,l=10,r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with ct2:
    valid_changes = dealer_agg["cpd_change"].dropna()
    mean_change   = aggregate(valid_changes)
    fig_hist = px.histogram(
        pd.DataFrame({"cpd_change": valid_changes}), x="cpd_change", nbins=30,
        color_discrete_sequence=[ORANGE],
        labels={"cpd_change": "CPD Change (After − Before)"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=1.5,
        annotation_text="Break-even", annotation_position="top right",
        annotation=dict(font_size=10, font_color="#EF4444"))
    if not np.isnan(mean_change):
        fig_hist.add_vline(x=mean_change, line_dash="dot", line_color="#10B981",
            line_width=1.5,
            annotation_text=f"{AGG_KEY}: {mean_change:.2f}",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#10B981"))
    style_fig(fig_hist, "Distribution of CPD Change per Dealer")
    fig_hist.update_layout(height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

with ct3:
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
# ROW 4 — Monthly Ticket Trend (full dataset)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Monthly Ticket Volume Trend (All Periods)</div>', unsafe_allow_html=True)

df_trend_base = df.copy()
if "ALL CATEGORIES" not in sub_cat_selection and sub_cat_selection:
    df_trend_base = df_trend_base[
        df_trend_base["TICKET_SUB_CATEGORY"].isin(sub_cat_selection)
    ]

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
fig_monthly.update_traces(
    texttemplate="%{text:,.0f}", textposition="outside", textfont_size=10
)
current_label = f"{month} {year}"
if current_label in monthly_trend["YearMonth"].values:
    x_idx = list(monthly_trend["YearMonth"]).index(current_label)
    fig_monthly.add_vline(x=x_idx, line_dash="dot", line_color=ORANGE, line_width=2,
        annotation_text="Selected", annotation_position="top",
        annotation=dict(font_color=ORANGE, font_size=10))
style_fig(fig_monthly, f"Total Tickets per Month · {active_subcat_label}")
fig_monthly.update_layout(height=320, legend_title_text="Year")
fig_monthly.update_xaxes(tickangle=-35, tickfont=dict(size=10))
st.plotly_chart(fig_monthly, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# SUB-CATEGORY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏷️ Ticket Sub-Category Intelligence</div>', unsafe_allow_html=True)

if "ALL CATEGORIES" not in sub_cat_selection and sub_cat_selection:
    st.markdown(
        f'<div class="info-banner">🔍 Sub-category filter active: <b>{active_subcat_label}</b></div>',
        unsafe_allow_html=True
    )

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    subcat_vol = (
        filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
        .nunique().reset_index(name="TICKET_COUNT")
        .sort_values("TICKET_COUNT", ascending=True).tail(15)
    )
    subcat_vol["pct"] = subcat_vol["TICKET_COUNT"] / subcat_vol["TICKET_COUNT"].sum() * 100
    fig_scvol = go.Figure(go.Bar(
        x=subcat_vol["TICKET_COUNT"], y=subcat_vol["TICKET_SUB_CATEGORY"],
        orientation="h",
        marker=dict(color=subcat_vol["TICKET_COUNT"],
                    colorscale=[[0,"#FFEDD5"],[0.5,"#FF7A3D"],[1,"#FF5200"]],
                    showscale=False),
        text=[f"{c:,} ({p:.1f}%)" for c,p in
              zip(subcat_vol["TICKET_COUNT"], subcat_vol["pct"])],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Tickets: %{x:,}<extra></extra>"
    ))
    style_fig(fig_scvol, "Top Sub-Categories by Ticket Volume")
    fig_scvol.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_scvol, use_container_width=True)

with col_sc2:
    dealer_with_subcat = dealer_agg.merge(dealer_primary_subcat, on="DEALER_CODE", how="left")
    subcat_cpd_rows = []
    for sc, grp_df in dealer_with_subcat.groupby("TICKET_SUB_CATEGORY"):
        val = aggregate(grp_df["cpd_change"].dropna())
        subcat_cpd_rows.append({"TICKET_SUB_CATEGORY": sc, "cpd_change": val})
    subcat_cpd = (
        pd.DataFrame(subcat_cpd_rows)
        .dropna(subset=["cpd_change"])
        .sort_values("cpd_change", ascending=True).tail(15)
    )
    subcat_cpd["Color"] = subcat_cpd["cpd_change"].apply(
        lambda x: "#10B981" if x >= 0 else "#EF4444"
    )
    subcat_cpd["Label"] = subcat_cpd["cpd_change"].apply(
        lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
    )
    fig_sccpd = go.Figure(go.Bar(
        x=subcat_cpd["cpd_change"], y=subcat_cpd["TICKET_SUB_CATEGORY"],
        orientation="h", marker_color=subcat_cpd["Color"],
        text=subcat_cpd["Label"], textposition="outside", textfont=dict(size=10),
        hovertemplate=f"<b>%{{y}}</b><br>{AGG_KEY} CPD Change: %{{x:.2f}}<extra></extra>"
    ))
    fig_sccpd.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    style_fig(fig_sccpd, f"{AGG_KEY} CPD Change by Sub-Category (After − Before)")
    fig_sccpd.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_sccpd, use_container_width=True)

st.markdown('<div class="section-title">📊 Sub-Category CPD Trend Across Ticket Window</div>',
            unsafe_allow_html=True)
col_sc3, col_sc4 = st.columns([3, 2])

dealer_with_subcat2 = dealer_with_subcat.copy()
dealer_with_subcat2["TICKET_SUB_CATEGORY"] = (
    dealer_with_subcat2["TICKET_SUB_CATEGORY"].fillna("Unknown")
)
top_subcats = (
    filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
    .nunique().nlargest(8).index.tolist()
)

with col_sc3:
    sc_trend_base = dealer_with_subcat2[
        dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)
    ]
    sc_trend_rows = []
    for sc, grp_df in sc_trend_base.groupby("TICKET_SUB_CATEGORY"):
        row = {"TICKET_SUB_CATEGORY": sc}
        for col in BUCKET_PERIOD_COLS:
            row[col] = aggregate(grp_df[col])
        sc_trend_rows.append(row)
    sc_trend_df = pd.DataFrame(sc_trend_rows)

    sc_melt = sc_trend_df.melt(
        id_vars="TICKET_SUB_CATEGORY", var_name="Period", value_name="CPD_Value"
    )
    sc_melt["Period"] = pd.Categorical(
        sc_melt["Period"], categories=BUCKET_PERIOD_COLS, ordered=True
    )
    sc_melt = sc_melt.sort_values("Period")
    sc_melt["Period_Label"] = sc_melt["Period"].map(PERIOD_LABEL_MAP)

    fig_sctrend = px.line(
        sc_melt, x="Period_Label", y="CPD_Value",
        color="TICKET_SUB_CATEGORY", markers=True,
        color_discrete_sequence=COLORS,
        category_orders={"Period_Label": list(PERIOD_LABEL_MAP.values())},
        labels={"CPD_Value": f"{AGG_KEY} CPD", "TICKET_SUB_CATEGORY": "Sub-Category"},
    )
    fig_sctrend.update_traces(line=dict(width=2), marker=dict(size=7),
                               connectgaps=False)
    fig_sctrend.add_vline(x=3, line_dash="dash", line_color=ORANGE, line_width=1.5,
        annotation_text="Ticket Raised",
        annotation=dict(font_size=10, font_color=ORANGE))
    style_fig(fig_sctrend, f"CPD Trend by Sub-Category ({AGG_KEY}) — Top 8 by Volume")
    fig_sctrend.update_layout(height=380, legend_title_text="Sub-Category",
                               legend=dict(font=dict(size=9)))
    st.plotly_chart(fig_sctrend, use_container_width=True)

with col_sc4:
    sc_ba = dealer_with_subcat2[dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)]
    sc_ba_rows = []
    for sc, grp_df in sc_ba.groupby("TICKET_SUB_CATEGORY"):
        sc_ba_rows.append({
            "TICKET_SUB_CATEGORY": sc,
            "cpd_before": aggregate(grp_df["cpd_before"]),
            "cpd_after":  aggregate(grp_df["cpd_after"]),
        })
    sc_ba_agg = (
        pd.DataFrame(sc_ba_rows)
        .sort_values("cpd_after", ascending=False)
    )
    fig_scba = go.Figure()
    fig_scba.add_trace(go.Bar(
        name="Before (M-3→M-1)", x=sc_ba_agg["TICKET_SUB_CATEGORY"],
        y=[v if not np.isnan(v) else None for v in sc_ba_agg["cpd_before"]],
        marker_color="#3B82F6",
        hovertemplate="<b>%{x}</b><br>Before: %{y:.2f}<extra></extra>"
    ))
    fig_scba.add_trace(go.Bar(
        name="After (M+1→M+3)", x=sc_ba_agg["TICKET_SUB_CATEGORY"],
        y=[v if not np.isnan(v) else None for v in sc_ba_agg["cpd_after"]],
        marker_color="#10B981",
        hovertemplate="<b>%{x}</b><br>After: %{y:.2f}<extra></extra>"
    ))
    fig_scba.update_layout(barmode="group")
    style_fig(fig_scba, f"{AGG_KEY} CPD Before vs After by Sub-Category")
    fig_scba.update_layout(height=380,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_scba.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    st.plotly_chart(fig_scba, use_container_width=True)

# Sub-Category × Bucket heatmap
st.markdown('<div class="section-title">🌡️ Sub-Category × Bucket — CPD Change Heatmap</div>',
            unsafe_allow_html=True)
sc_bucket_df = dealer_with_subcat2[dealer_with_subcat2["TICKET_SUB_CATEGORY"].isin(top_subcats)]
sc_bucket_rows = []
for (sc, bkt), grp_df in sc_bucket_df.groupby(["TICKET_SUB_CATEGORY","BUCKET"]):
    sc_bucket_rows.append({
        "TICKET_SUB_CATEGORY": sc,
        "BUCKET": bkt,
        "cpd_change": aggregate(grp_df["cpd_change"].dropna())
    })
sc_bucket_heat = pd.DataFrame(sc_bucket_rows).dropna(subset=["cpd_change"])
sc_bucket_pivot = sc_bucket_heat.pivot(
    index="TICKET_SUB_CATEGORY", columns="BUCKET", values="cpd_change"
)
fig_scheat = px.imshow(
    sc_bucket_pivot, color_continuous_scale="RdYlGn", aspect="auto",
    labels=dict(x="Bucket", y="Sub-Category", color=f"{AGG_KEY} CPD Δ"),
    text_auto=".2f"
)
style_fig(fig_scheat,
          f"{AGG_KEY} CPD Change: Sub-Category × Bucket — positive = improved after ticket")
fig_scheat.update_layout(height=380)
st.plotly_chart(fig_scheat, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL: Month × Bucket Heatmap
# ─────────────────────────────────────────────
if show_heatmap:
    st.markdown('<div class="section-title">🌡️ CPD Change Heatmap — Month × Bucket</div>',
                unsafe_allow_html=True)

    hmap_base = df_trend_base.copy()
    hmap_dealer = (
        hmap_base
        .groupby(["YEAR","TICKET_RAISED_MONTH","MONTH_NUM","DEALER_CODE","BUCKET"])
        [["D_CM-3","D_CM-2","D_CM-1","D_CM+1","D_CM+2","D_CM+3"]]
        .mean().reset_index()
    )
    hmap_dealer["cpd_before"] = hmap_dealer[["D_CM-3","D_CM-2","D_CM-1"]].mean(axis=1)
    hmap_dealer["cpd_after"]  = hmap_dealer[["D_CM+1","D_CM+2","D_CM+3"]].mean(axis=1)
    hmap_dealer["cpd_change"] = hmap_dealer["cpd_after"] - hmap_dealer["cpd_before"]

    hmap_rows = []
    for (mo, bkt), grp_df in hmap_dealer.groupby(["TICKET_RAISED_MONTH","BUCKET"]):
        hmap_rows.append({
            "TICKET_RAISED_MONTH": mo,
            "BUCKET": bkt,
            "cpd_change": aggregate(grp_df["cpd_change"].dropna())
        })
    hmap_df = pd.DataFrame(hmap_rows).dropna(subset=["cpd_change"])
    hmap_pivot = hmap_df.pivot(
        index="BUCKET", columns="TICKET_RAISED_MONTH", values="cpd_change"
    )
    month_cols = [m for m in MONTH_ORDER if m in hmap_pivot.columns]
    hmap_pivot = hmap_pivot[month_cols]

    fig_hmap = px.imshow(
        hmap_pivot, color_continuous_scale="RdYlGn", aspect="auto",
        labels=dict(x="Month", y="Bucket", color=f"{AGG_KEY} CPD Δ"),
        text_auto=".2f"
    )
    style_fig(fig_hmap,
              f"{AGG_KEY} CPD Change (After − Before) by Month & Bucket · {active_subcat_label}")
    fig_hmap.update_layout(height=330)
    st.plotly_chart(fig_hmap, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL: Sub-Category Deep Dive
# ─────────────────────────────────────────────
if show_subcat_deep:
    st.markdown('<div class="section-title">🔬 Sub-Category Deep Dive — Month-over-Month Trend</div>',
                unsafe_allow_html=True)
    sc_mom = (
        df_trend_base
        .groupby(["YEAR","TICKET_RAISED_MONTH","MONTH_NUM","TICKET_SUB_CATEGORY"])["TICKET_ID"]
        .nunique().reset_index(name="TICKET_COUNT")
        .sort_values(["YEAR","MONTH_NUM"])
    )
    sc_mom["YearMonth"] = sc_mom["TICKET_RAISED_MONTH"] + " " + sc_mom["YEAR"].astype(str)
    top6_sc = (
        filtered_data.groupby("TICKET_SUB_CATEGORY")["TICKET_ID"]
        .nunique().nlargest(6).index.tolist()
    )
    fig_scmom = px.line(
        sc_mom[sc_mom["TICKET_SUB_CATEGORY"].isin(top6_sc)],
        x="YearMonth", y="TICKET_COUNT",
        color="TICKET_SUB_CATEGORY", markers=True,
        color_discrete_sequence=COLORS,
        labels={"TICKET_COUNT":"Ticket Count","TICKET_SUB_CATEGORY":"Sub-Category"},
    )
    fig_scmom.update_traces(line=dict(width=2), marker=dict(size=7))
    style_fig(fig_scmom, "Month-over-Month Ticket Volume — Top 6 Sub-Categories")
    fig_scmom.update_layout(height=340, legend=dict(font=dict(size=10)))
    fig_scmom.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    st.plotly_chart(fig_scmom, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL: Dealer Drill-Down
# ─────────────────────────────────────────────
if show_dealer_drill:
    st.markdown('<div class="section-title">🔍 Dealer-Level Drill-Down</div>', unsafe_allow_html=True)

    drill_df = dealer_agg.merge(dealer_primary_subcat, on="DEALER_CODE", how="left")
    drill_df = (
        drill_df[["DEALER_CODE","BUCKET","TICKET_SUB_CATEGORY","TICKET_CNT",
                  "cpd_before","CPD","cpd_after","cpd_change"]]
        .copy().sort_values("cpd_change", ascending=False)
    )
    drill_df.columns = ["Dealer Code","Bucket","Primary Sub-Cat","Tickets",
                        "CPD Before","CPD Ticket Month","CPD After","CPD Change"]
    drill_df = drill_df.round(2)

    def color_cpd_change(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        max_abs = drill_df["CPD Change"].abs().max() + 1e-9
        if v > 0:
            intensity = min(int(abs(v) / max_abs * 180), 180)
            return (f"background-color: rgb({255-intensity},255,{255-intensity});"
                    f"color:#065F46;font-weight:600")
        elif v < 0:
            intensity = min(int(abs(v) / max_abs * 180), 180)
            return (f"background-color: rgb(255,{255-intensity},{255-intensity});"
                    f"color:#991B1B;font-weight:600")
        return ""

    top_n = st.slider("Show Top / Bottom N Dealers", 5, 50, 15)
    tab_top, tab_bot, tab_search = st.tabs(
        ["🏆 Top Improvers","⚠️ Bottom Performers","🔎 Search Dealer"]
    )
    with tab_top:
        st.dataframe(
            drill_df.head(top_n).style
                .map(color_cpd_change, subset=["CPD Change"])
                .format({"Tickets": "{:,.0f}"}),
            use_container_width=True, hide_index=True
        )
    with tab_bot:
        st.dataframe(
            drill_df.tail(top_n).sort_values("CPD Change").style
                .map(color_cpd_change, subset=["CPD Change"])
                .format({"Tickets": "{:,.0f}"}),
            use_container_width=True, hide_index=True
        )
    with tab_search:
        search_code = st.text_input("Enter Dealer Code")
        if search_code:
            result = drill_df[drill_df["Dealer Code"].astype(str).str.contains(search_code)]
            if not result.empty:
                st.dataframe(
                    result.style
                        .map(color_cpd_change, subset=["CPD Change"])
                        .format({"Tickets": "{:,.0f}"}),
                    use_container_width=True, hide_index=True
                )
                matched = dealer_agg[
                    dealer_agg["DEALER_CODE"].astype(str).str.contains(search_code)
                ]
                if not matched.empty:
                    spark_vals = [aggregate(matched[c]) for c in PERIOD_COLS]
                    spark_plot = [v if not np.isnan(v) else None for v in spark_vals]
                    fig_spark = go.Figure(go.Scatter(
                        x=PERIOD_LABELS, y=spark_plot,
                        mode="lines+markers",
                        line=dict(color=ORANGE, width=2),
                        marker=dict(size=9),
                        connectgaps=False
                    ))
                    style_fig(fig_spark, f"CPD Trend ({AGG_KEY}) for Dealer: {search_code}")
                    fig_spark.update_layout(height=260)
                    st.plotly_chart(fig_spark, use_container_width=True)

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
col_d1, _ = st.columns([1, 4])
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
