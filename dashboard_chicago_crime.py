"""
Dashboard Analisis Kriminalitas Chicago 2020-2025
Streamlit App — siap dijalankan dengan: streamlit run dashboard_chicago_crime.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Crime Dashboard 2020–2025",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    padding: 4px 0;
}

/* Main background */
.main { background-color: #0d1117; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
}
.metric-delta {
    font-size: 0.8rem;
    margin-top: 0.25rem;
    color: #8b949e;
}
.metric-delta.up { color: #3fb950; }
.metric-delta.down { color: #f85149; }

/* Section header */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.88rem;
    color: #8b949e;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    color: #c9d1d9;
    line-height: 1.7;
}
.insight-box strong { color: #58a6ff; }
.insight-box.warning { border-left-color: #f0883e; }
.insight-box.warning strong { color: #f0883e; }
.insight-box.success { border-left-color: #3fb950; }
.insight-box.success strong { color: #3fb950; }
.insight-box.danger { border-left-color: #f85149; }
.insight-box.danger strong { color: #f85149; }

/* Tab overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #8b949e !important;
    border-radius: 7px;
    padding: 0.45rem 1rem;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

/* Page title */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.03em;
    line-height: 1.4;
    padding-top: 20px;
    margin-top: 10px;
}
}
.page-title span { color: #58a6ff; }
.page-desc {
    font-size: 0.92rem;
    color: #8b949e;
    margin-top: 0.4rem;
    line-height: 1.6;
}

/* Divider */
.hline { border-top: 1px solid #21262d; margin: 1.2rem 0; }

/* Question badge */
.q-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    border-radius: 20px;
    padding: 2px 12px;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK THEME DEFAULTS
# ─────────────────────────────────────────────
PLOT_BG    = "#0d1117"
PAPER_BG   = "#161b22"
GRID_COLOR = "#21262d"
FONT_COLOR = "#c9d1d9"
ACCENT     = "#58a6ff"
PALETTE    = ["#58a6ff","#3fb950","#f0883e","#a371f7","#f85149","#56d364","#d2a8ff","#ffa657"]

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="DM Sans, sans-serif", color=FONT_COLOR, size=12),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
)

def apply_base(fig, title=""):
    fig.update_layout(**BASE_LAYOUT, title=dict(text=title, font=dict(family="Syne, sans-serif", size=15, color="#e6edf3"), x=0.02))
    return fig

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("chicago_crime.csv")
    df.columns = df.columns.str.lower().str.strip()
    # parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df['hour'] = df['date'].dt.hour
    # ensure year col
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    df['year'] = df['year'].astype(int)
    # bool → int
    for col in ['arrest','domestic']:
        if df[col].dtype == object:
            df[col] = df[col].map({'True':True,'False':False,True:True,False:False})
        df[col] = df[col].astype(bool)
    df['arrest_int']   = df['arrest'].astype(int)
    df['domestic_int'] = df['domestic'].astype(int)
    df = df.dropna(subset=['latitude','longitude','district'])
    df['district'] = df['district'].astype(int)
    return df

@st.cache_data
def make_district_features(df):
    feats = df.groupby('district').agg(
        jumlah_kejahatan=('arrest_int','count'),
        arrest_rate_pct=('arrest_int','mean'),
        domestic_rate_pct=('domestic_int','mean'),
    ).reset_index()
    feats['arrest_rate_pct']   *= 100
    feats['domestic_rate_pct'] *= 100
    return feats

@st.cache_data
def run_kmeans(_feats, k=6):
    X = _feats[['jumlah_kejahatan','arrest_rate_pct','domestic_rate_pct']].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    df_c = _feats.copy()
    df_c['cluster'] = labels + 1
    df_c['cluster_label'] = "Cluster " + df_c['cluster'].astype(str)
    return df_c, sil

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Chicago Crime\n**Analisis 2020–2025**")
    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
   
    df_raw = load_data()
    year_options = sorted(df_raw['year'].unique().tolist())
    year_filter = st.multiselect(
        "Filter Tahun",
        year_options,
        default=year_options
    )

    df = df_raw[df_raw['year'].isin(year_filter)].copy()

    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
    st.caption(f"📊 **{len(df):,}** baris aktif dari {df['year'].min()}–{df['year'].max()}")
    st.caption(f"🗂️ **{df['primary_type'].nunique()}** tipe kejahatan")
    st.caption(f"🏙️ **{df['district'].nunique()}** district")

    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
    st.caption("📌 Sumber: BigQuery Public Dataset\n`bigquery-public-data.chicago_crime.crime`")
    st.caption("🎓 Proyek Akhir — Analisis Data Berbasis Cloud (SST60202)")

# ─────────────────────────────────────────────
# HEADER + KPI
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-title">Kriminalitas Chicago <span>2020–2025</span></div>
<div class="page-desc">Dashboard analisis tren, pola, dan klasterisasi district berbasis data BigQuery Public Dataset.</div>
""", unsafe_allow_html=True)
st.markdown("<div class='hline'></div>", unsafe_allow_html=True)

# KPI row
total = len(df)
arrests = df['arrest'].sum()
arrest_pct = arrests / total * 100
domestic = df['domestic'].sum()
top_type = df['primary_type'].value_counts().idxmax()
top_type_cnt = df['primary_type'].value_counts().max()
top_dist = df['district'].value_counts().idxmax()

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Total Kejahatan", f"{total:,}", "📋", None),
    (k2, "Total Penangkapan", f"{arrests:,}", "👮", f"{arrest_pct:.1f}% dari total"),
    (k3, "Arrest Rate", f"{arrest_pct:.1f}%", "⚖️", None),
    (k4, "Tipe Terbanyak", top_type.title(), "🔑", f"{top_type_cnt} kasus"),
    (k5, "District Tertinggi", f"District {top_dist}", "🏘️", None),
]
for col, label, val, icon, sub in kpis:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{val}</div>
        {'<div class="metric-delta">' + sub + '</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='hline'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_eda, tab_q1, tab_q2, tab_q3, tab_q4, tab_q5 = st.tabs([
    "📊 EDA",
    "📈 Q1 · Tren Kejahatan",
    "🏘️ Q2 · Per District",
    "⚖️ Q3 · Arrest Rate",
    "🕐 Q4 · Pola Jam",
    "🤖 Q5 · K-Means Clustering",
])

# ════════════════════════════════════════════
# TAB EDA
# ════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Gambaran umum distribusi dan karakteristik dataset Chicago Crime 2020–2025.</div>', unsafe_allow_html=True)

    # ── EDA Page 1: Distribusi Jenis Kejahatan + Arrest Pie ──
    with st.expander("📌 EDA 1 — Distribusi Jenis Kejahatan & Proporsi Penangkapan", expanded=True):
        col_a, col_b = st.columns([3, 2])

        with col_a:
            top_types = df['primary_type'].value_counts().reset_index()
            top_types.columns = ['primary_type','count']
            fig_bar = go.Figure(go.Bar(
                y=top_types['primary_type'], x=top_types['count'],
                orientation='h',
                marker=dict(color=top_types['count'], colorscale='Blues', showscale=False),
                text=top_types['count'], textposition='outside',
                textfont=dict(color=FONT_COLOR, size=10),
            ))
            apply_base(fig_bar, "Distribusi Jenis Kejahatan (Semua Tipe)")
            fig_bar.update_layout(height=520, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            arrest_counts = df['arrest'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=['Tidak Ditangkap','Ditangkap'],
                values=[arrest_counts.get(False,0), arrest_counts.get(True,0)],
                hole=0.48,
                marker_colors=['#f85149','#3fb950'],
                textinfo='percent+label',
                textfont=dict(size=13),
            ))
            apply_base(fig_pie, "Proporsi Penangkapan")
            fig_pie.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

            domestic_counts = df['domestic'].value_counts()
            fig_pie2 = go.Figure(go.Pie(
                labels=['Non-Domestik','Domestik'],
                values=[domestic_counts.get(False,0), domestic_counts.get(True,0)],
                hole=0.48,
                marker_colors=['#58a6ff','#f0883e'],
                textinfo='percent+label',
                textfont=dict(size=13),
            ))
            apply_base(fig_pie2, "Proporsi Kejahatan Domestik")
            fig_pie2.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_pie2, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>Interpretasi:</strong> Jenis kejahatan terbanyak dalam dataset adalah <strong>Theft (Pencurian)</strong>, 
            yang mendominasi jauh di atas tipe lainnya. Ini mencerminkan pola umum kejahatan perkotaan besar di Amerika Serikat. 
            Dari sisi penangkapan, <strong>lebih dari 85% kasus tidak berujung pada penangkapan</strong>, mengindikasikan 
            bahwa penegakan hukum masih menghadapi tantangan besar — baik dari sisi bukti, kapasitas, maupun kecepatan respons. 
            Sementara itu, proporsi kejahatan domestik relatif kecil (~20%), namun tetap signifikan sebagai kategori tersendiri.
        </div>
        """, unsafe_allow_html=True)

    # ── EDA Page 2: Heatmap tipe per tahun ──
    with st.expander("📌 EDA 2 — Heatmap Tipe Kejahatan per Tahun (Top 8)"):
        top8 = df['primary_type'].value_counts().head(8).index.tolist()
        heat_df = (df[df['primary_type'].isin(top8)]
                   .groupby(['year','primary_type']).size()
                   .reset_index(name='count'))
        heat_pivot = heat_df.pivot(index='primary_type', columns='year', values='count').fillna(0)

        fig_heat = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(str).tolist(),
            y=heat_pivot.index.tolist(),
            colorscale='YlOrRd',
            text=heat_pivot.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Jumlah", tickfont=dict(color=FONT_COLOR)),
        ))
        apply_base(fig_heat, "Heatmap — Tipe Kejahatan per Tahun (Top 8)")
        fig_heat.update_layout(height=380)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="insight-box warning">
            <strong>Interpretasi:</strong> Heatmap menunjukkan pola fluktuasi kejahatan di setiap tipe dari tahun ke tahun. 
            <strong>Theft</strong> dan <strong>Battery</strong> secara konsisten berada di angka tinggi. 
            Terdapat lonjakan yang terlihat pada tahun 2023–2024, yang bisa mengindikasikan peningkatan aktivitas kriminal 
            pasca-pandemi COVID-19 seiring normalisasi mobilitas masyarakat. Beberapa tipe seperti 
            <strong>Criminal Damage</strong> dan <strong>Assault</strong> juga menunjukkan kehadiran stabil lintas tahun.
        </div>
        """, unsafe_allow_html=True)

    # ── EDA Page 3: Korelasi Spearman ──
    with st.expander("📌 EDA 3 — Heatmap Korelasi Spearman Antar Variabel"):
        num_df = df[['arrest_int','domestic_int','hour','year']].rename(columns={
            'arrest_int':'Arrest','domestic_int':'Domestik','hour':'Jam','year':'Tahun'
        })
        corr_matrix = num_df.corr(method='spearman')

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdYlGn',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="Korelasi", tickfont=dict(color=FONT_COLOR)),
        ))
        apply_base(fig_corr, "Heatmap Korelasi Spearman Antar Variabel")
        fig_corr.update_layout(height=360, width=480)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        <div class="insight-box success">
            <strong>Interpretasi:</strong> Korelasi Spearman digunakan karena variabel bersifat ordinal/biner. 
            Hasil menunjukkan <strong>korelasi antara variabel sangat lemah</strong> (mendekati 0), artinya 
            variabel seperti jam kejadian, tahun, status domestik, dan penangkapan bersifat relatif 
            <strong>independen satu sama lain</strong>. Hal ini mengindikasikan bahwa pola penangkapan tidak 
            semata-mata bergantung pada waktu atau kategori domestik, melainkan ada faktor kompleks lain 
            (jenis kejahatan, lokasi, kapasitas kepolisian) yang lebih menentukan.
        </div>
        """, unsafe_allow_html=True)

    # ── EDA Page 4: Peta Geospasial ──
    with st.expander("📌 EDA 4 — Peta Sebaran Kejahatan Chicago"):
        sample_geo = df.sample(min(600, len(df)), random_state=42)
        fig_map = px.scatter_mapbox(
            sample_geo,
            lat='latitude', lon='longitude',
            color='primary_type',
            hover_data=['year','district','arrest'],
            zoom=10, height=500,
            mapbox_style='carto-darkmatter',
            color_discrete_sequence=PALETTE,
        )
        fig_map.update_traces(marker_size=5, opacity=0.7)
        fig_map.update_layout(paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
                               title=dict(text="Peta Sebaran Kejahatan Chicago (sample 600 titik)", font=dict(family="Syne, sans-serif",size=14,color="#e6edf3"),x=0.02),
                               margin=dict(l=0,r=0,t=40,b=0),
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>Interpretasi:</strong> Peta sebaran menunjukkan bahwa kejahatan tersebar di <strong>seluruh wilayah Chicago</strong>, 
            namun dengan konsentrasi lebih tinggi di area tengah-selatan kota. Beberapa tipe kejahatan seperti 
            <strong>Theft</strong> terlihat lebih menyebar di pusat kota, sementara tipe seperti 
            <strong>Battery</strong> dan <strong>Narcotics</strong> cenderung mengelompok di area tertentu. 
            Ini mendukung pendekatan <em>hot-spot policing</em> yang menargetkan zona geografis spesifik.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB Q1: TREN KEJAHATAN
# ════════════════════════════════════════════
with tab_q1:
    st.markdown('<div class="q-badge">Pertanyaan 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Tren Jumlah Kejahatan di Chicago 2020–2025</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Bagaimana tren jumlah kejahatan di Chicago dari tahun 2020 hingga 2025?</div>', unsafe_allow_html=True)

    trend_year = df.groupby('year').size().reset_index(name='jumlah_kejahatan')
    trend_year['pct_change'] = trend_year['jumlah_kejahatan'].pct_change() * 100

    # ── Chart 1a: Bar + Line ──
    max_idx = trend_year['jumlah_kejahatan'].idxmax()
    colors = [ACCENT if i != max_idx else '#f85149' for i in range(len(trend_year))]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=trend_year['year'], y=trend_year['jumlah_kejahatan'],
        marker_color=colors, name='Jumlah Kejahatan',
        text=trend_year['jumlah_kejahatan'],
        textposition='outside', textfont=dict(color=FONT_COLOR, size=12, family='Syne'),
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_year['year'], y=trend_year['jumlah_kejahatan'],
        mode='lines+markers', name='Tren',
        line=dict(color='#f0883e', width=2.5),
        marker=dict(size=9, color='#f0883e', symbol='circle'),
    ))
    apply_base(fig_trend, "Tren Jumlah Kejahatan per Tahun (Bar merah = puncak tertinggi)")
    fig_trend.update_layout(height=380, xaxis=dict(tickmode='linear', dtick=1), barmode='overlay')
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Chart 1b: % Change ──
    col1, col2 = st.columns([1,1])
    with col1:
        pct_df = trend_year.dropna(subset=['pct_change'])
        fig_pct = go.Figure(go.Bar(
            x=pct_df['year'], y=pct_df['pct_change'],
            marker_color=['#3fb950' if v < 0 else '#f85149' for v in pct_df['pct_change']],
            text=[f"{v:+.1f}%" for v in pct_df['pct_change']],
            textposition='outside', textfont=dict(color=FONT_COLOR, size=11),
        ))
        apply_base(fig_pct, "Perubahan % YoY Jumlah Kejahatan")
        fig_pct.add_hline(y=0, line_color='#8b949e', line_dash='dash', line_width=1)
        fig_pct.update_layout(height=300, xaxis=dict(tickmode='linear',dtick=1))
        st.plotly_chart(fig_pct, use_container_width=True)

    with col2:
        # Tabel ringkasan
        st.markdown("<br>", unsafe_allow_html=True)
        display_df = trend_year.copy()
        display_df['pct_change'] = display_df['pct_change'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
        )
        display_df.columns = ['Tahun','Jumlah Kejahatan','Perubahan YoY']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box warning">
        <strong>Interpretasi:</strong> Tren jumlah kejahatan di Chicago bersifat <strong>fluktuatif</strong> — tidak menunjukkan 
        kenaikan atau penurunan yang konsisten. Tahun 2021 mencatat penurunan (−9.94%) yang kemungkinan dipengaruhi oleh 
        pembatasan mobilitas pandemi COVID-19. Namun, tahun 2022 dan 2023 menunjukkan lonjakan signifikan 
        (+15.3% dan +25%), yang berkorelasi dengan normalisasi aktivitas publik pasca-pandemi. 
        Puncak kejahatan terjadi pada <strong>2023</strong>, diikuti sedikit penurunan di 2024 dan 2025. 
        Pola ini menunjukkan bahwa kondisi sosial-ekonomi pasca-pandemi memiliki dampak nyata terhadap angka kriminalitas.
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1c: Distribusi per jenis per tahun (stacked area) ──
    st.markdown("<br>", unsafe_allow_html=True)
    top5_types = df['primary_type'].value_counts().head(5).index.tolist()
    trend_type = (df[df['primary_type'].isin(top5_types)]
                  .groupby(['year','primary_type']).size().reset_index(name='count'))

    fig_area = px.area(
        trend_type, x='year', y='count', color='primary_type',
        color_discrete_sequence=PALETTE,
        labels={'count':'Jumlah','year':'Tahun','primary_type':'Tipe Kejahatan'},
    )
    apply_base(fig_area, "Tren per Tipe Kejahatan (Top 5) — Area Chart")
    fig_area.update_layout(height=320, xaxis=dict(tickmode='linear',dtick=1))
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Insight Tambahan:</strong> Breakdown per tipe menunjukkan bahwa <strong>Theft</strong> adalah 
        penyumbang terbesar di setiap tahun dan ikut berkontribusi pada lonjakan 2023. 
        Tipe lain seperti <strong>Battery</strong> dan <strong>Criminal Damage</strong> juga berfluktuasi searah 
        dengan tren keseluruhan, mengindikasikan faktor sistemik yang memengaruhi semua jenis kejahatan secara bersamaan.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB Q2: PER DISTRICT
# ════════════════════════════════════════════
with tab_q2:
    st.markdown('<div class="q-badge">Pertanyaan 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Kejahatan per District di Chicago</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">District mana yang menyumbang jumlah kejahatan terbanyak, dan berapa rata-rata kejahatan per district?</div>', unsafe_allow_html=True)

    dist_df = df.groupby('district').agg(
        jumlah_kejahatan=('arrest_int','count'),
        arrest_rate=('arrest_int','mean'),
        domestic_rate=('domestic_int','mean'),
    ).reset_index().sort_values('jumlah_kejahatan', ascending=False)
    dist_df['arrest_rate_pct']   = dist_df['arrest_rate'] * 100
    dist_df['domestic_rate_pct'] = dist_df['domestic_rate'] * 100
    avg_crime = dist_df['jumlah_kejahatan'].mean()

    # ── Chart 2a: Bar horizontal ──
    fig_dist = go.Figure(go.Bar(
        y=dist_df['district'].astype(str),
        x=dist_df['jumlah_kejahatan'],
        orientation='h',
        marker=dict(
            color=dist_df['jumlah_kejahatan'],
            colorscale='Blues', showscale=True,
            colorbar=dict(title="Jumlah", tickfont=dict(color=FONT_COLOR)),
        ),
        text=dist_df['jumlah_kejahatan'],
        textposition='outside',
        textfont=dict(color=FONT_COLOR, size=10),
    ))
    apply_base(fig_dist, "Jumlah Kejahatan per District (diurutkan)")
    fig_dist.add_vline(x=avg_crime, line_dash='dash', line_color='#f0883e',
                       annotation_text=f" Rata-rata: {avg_crime:.0f}", annotation_font_color='#f0883e')
    fig_dist.update_layout(height=520, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_dist, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── Chart 2b: Scatter Kejahatan vs Arrest Rate ──
    with col_a:
        fig_scatter = px.scatter(
            dist_df, x='jumlah_kejahatan', y='arrest_rate_pct',
            text='district', color='jumlah_kejahatan',
            color_continuous_scale='Reds', size='jumlah_kejahatan',
            labels={'jumlah_kejahatan':'Jumlah Kejahatan','arrest_rate_pct':'Arrest Rate (%)'},
        )
        fig_scatter.update_traces(textposition='top center', textfont=dict(color='#c9d1d9', size=9))
        apply_base(fig_scatter, "Jumlah Kejahatan vs Arrest Rate per District")
        fig_scatter.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Chart 2c: Domestic rate bubble ──
    with col_b:
        fig_bub = px.scatter(
            dist_df, x='jumlah_kejahatan', y='domestic_rate_pct',
            size='jumlah_kejahatan', color='arrest_rate_pct',
            text='district', color_continuous_scale='Viridis',
            labels={'jumlah_kejahatan':'Jumlah Kejahatan','domestic_rate_pct':'Domestic Rate (%)','arrest_rate_pct':'Arrest Rate (%)'},
        )
        fig_bub.update_traces(textposition='top center', textfont=dict(color='#c9d1d9', size=9))
        apply_base(fig_bub, "Jumlah Kejahatan vs Domestic Rate (warna = Arrest Rate)")
        fig_bub.update_layout(height=350)
        st.plotly_chart(fig_bub, use_container_width=True)

    st.markdown("""
    <div class="insight-box danger">
        <strong>Interpretasi:</strong> Terdapat <strong>ketimpangan signifikan</strong> antar district dalam jumlah kejahatan. 
        District-district tertentu (seperti District 6, 11, dan 8) secara konsisten mendominasi dengan angka jauh di atas 
        rata-rata, sementara beberapa district lain berada jauh di bawah. Garis putus-putus oranye menandai rata-rata 
        sebagai referensi. Scatter plot menunjukkan bahwa <strong>district dengan kejahatan tinggi tidak selalu memiliki 
        arrest rate tinggi</strong> — ini mengindikasikan adanya kesenjangan antara volume kejahatan dan kapasitas penegakan hukum. 
        Intervensi berbasis district perlu mempertimbangkan kedua dimensi ini secara bersamaan.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB Q3: ARREST RATE TREND
# ════════════════════════════════════════════
with tab_q3:
    st.markdown('<div class="q-badge">Pertanyaan 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Tren Arrest Rate 2020–2025</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Apakah tingkat penangkapan (arrest rate) menunjukkan tren meningkat atau menurun dari 2020 ke 2025?</div>', unsafe_allow_html=True)

    arrest_trend = df.groupby('year').agg(
        total=('arrest_int','count'),
        arrested=('arrest_int','sum'),
    ).reset_index()
    arrest_trend['arrest_rate_pct'] = arrest_trend['arrested'] / arrest_trend['total'] * 100

    # Spearman test
    rho, pval = stats.spearmanr(arrest_trend['year'], arrest_trend['arrest_rate_pct'])

    # ── KPI row ──
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">🔬 Spearman ρ</div>
        <div class="metric-value">{rho:.3f}</div>
        <div class="metric-delta">Korelasi tren</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 p-value</div>
        <div class="metric-value">{pval:.4f}</div>
        <div class="metric-delta">{'Signifikan (p < 0.05)' if pval < 0.05 else 'Tidak Signifikan (p ≥ 0.05)'}</div>
    </div>""", unsafe_allow_html=True)
    trend_dir = "↑ Meningkat" if rho > 0 else "↓ Menurun"
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">📉 Arah Tren</div>
        <div class="metric-value">{trend_dir}</div>
        <div class="metric-delta">{'Positif' if rho > 0 else 'Negatif'}</div>
    </div>""", unsafe_allow_html=True)

    # ── Chart 3a: Line arrest rate ──
    fig_arr = go.Figure()
    fig_arr.add_trace(go.Bar(
        x=arrest_trend['year'], y=arrest_trend['arrested'],
        name='Jumlah Ditangkap', marker_color='#3fb950',
        yaxis='y2',
    ))
    fig_arr.add_trace(go.Scatter(
        x=arrest_trend['year'], y=arrest_trend['arrest_rate_pct'],
        mode='lines+markers+text',
        name='Arrest Rate (%)',
        line=dict(color='#3fb950', width=3),
        marker=dict(size=10, color='#3fb950'),
        text=[f"{v:.1f}%" for v in arrest_trend['arrest_rate_pct']],
        textposition='top center', textfont=dict(color='#3fb950', size=11),
    ))
    apply_base(fig_arr, "Tren Arrest Rate per Tahun (%) dengan Jumlah Penangkapan")
    fig_arr.update_layout(
    height=380,
    xaxis=dict(tickmode='linear', dtick=1),

    yaxis=dict(
        title=dict(
            text="Arrest Rate (%)",
            font=dict(color='#3fb950')
        )
    ),

    yaxis2=dict(
        title=dict(
            text="Jumlah Ditangkap",
            font=dict(color='#3fb950')
        ),
        overlaying='y',
        side='right',
        showgrid=False
    ),

    legend=dict(x=0.01, y=0.99),
)
    st.plotly_chart(fig_arr, use_container_width=True)

    # ── Chart 3b: Arrest rate by type ──
    arrest_by_type = (df.groupby('primary_type')
                      .agg(total=('arrest_int','count'), arrested=('arrest_int','sum'))
                      .assign(rate=lambda x: x['arrested']/x['total']*100)
                      .reset_index()
                      .sort_values('rate', ascending=False)
                      .head(15))

    fig_atype = go.Figure(go.Bar(
        y=arrest_by_type['primary_type'], x=arrest_by_type['rate'],
        orientation='h',
        marker=dict(color=arrest_by_type['rate'], colorscale='Greens', showscale=False),
        text=[f"{v:.1f}%" for v in arrest_by_type['rate']],
        textposition='outside', textfont=dict(color=FONT_COLOR, size=10),
    ))
    apply_base(fig_atype, "Arrest Rate per Tipe Kejahatan (Top 15 Tertinggi)")
    fig_atype.update_layout(height=400, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_atype, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box success">
        <strong>Interpretasi:</strong> Uji korelasi Spearman menghasilkan ρ = <strong>{rho:.3f}</strong> 
        dengan p-value = <strong>{pval:.4f}</strong> 
        ({'signifikan secara statistik' if pval < 0.05 else 'tidak signifikan secara statistik'}). 
        Arah tren arrest rate bersifat <strong>{"meningkat" if rho > 0 else "menurun"}</strong> secara keseluruhan. 
        Visualisasi per tipe kejahatan mengungkap bahwa tipe seperti <strong>Prostitution</strong>, 
        <strong>Narcotics</strong>, dan <strong>Weapons Violation</strong> memiliki arrest rate tertinggi — 
        ini karena jenis kejahatan tersebut sering kali tertangkap tangan (in-flagrante). 
        Sementara itu, tipe seperti <strong>Theft</strong> dan <strong>Criminal Damage</strong> 
        memiliki arrest rate yang jauh lebih rendah karena sulit melacak pelaku setelah kejadian.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB Q4: POLA JAM
# ════════════════════════════════════════════
with tab_q4:
    st.markdown('<div class="q-badge">Pertanyaan 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Pola Waktu Kejahatan per Jam</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pada jam berapa kejahatan paling sering terjadi dalam sehari?</div>', unsafe_allow_html=True)

    hour_df = df.groupby('hour').agg(
        jumlah=('arrest_int','count'),
        arrested=('arrest_int','sum'),
    ).reset_index()
    hour_df['arrest_rate_pct'] = hour_df['arrested'] / hour_df['jumlah'] * 100

    # ── Chart 4a: Bar per jam + line arrest rate ──
    max_hour = hour_df.loc[hour_df['jumlah'].idxmax(), 'hour']
    bar_colors = ['#f85149' if h == max_hour else ACCENT for h in hour_df['hour']]

    fig_hour = go.Figure()
    fig_hour.add_trace(go.Bar(
        x=hour_df['hour'], y=hour_df['jumlah'],
        name='Jumlah Kejahatan', marker_color=bar_colors,
    ))
    fig_hour.add_trace(go.Scatter(
        x=hour_df['hour'], y=hour_df['arrest_rate_pct'],
        mode='lines+markers', name='Arrest Rate (%)',
        line=dict(color='#f0883e', width=2.5),
        marker=dict(size=7), yaxis='y2',
    ))
    apply_base(fig_hour, f"Distribusi Kejahatan per Jam (bar merah = puncak jam {max_hour}:00)")
    fig_hour.update_layout(
        height=380,
        xaxis=dict(tickmode='linear', dtick=2, title="Jam (0–23)"),
        yaxis=dict(title="Jumlah Kejahatan"),
        yaxis2=dict(
    title=dict(
        text="Arrest Rate (%)",
        font=dict(color='#f0883e')
    ),
    overlaying='y',
    side='right',
    showgrid=False
),
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # ── Chart 4b: Polar / Radar chart ──
    col_a, col_b = st.columns(2)
    with col_a:
        fig_polar = go.Figure(go.Scatterpolar(
            r=hour_df['jumlah'].tolist() + [hour_df['jumlah'].iloc[0]],
            theta=[f"{h}:00" for h in hour_df['hour'].tolist()] + ["0:00"],
            mode='lines+markers',
            fill='toself', fillcolor='rgba(88,166,255,0.15)',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=5),
        ))
        apply_base(fig_polar, "Pola 24 Jam (Polar Chart)")
        fig_polar.update_layout(
            height=360, polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, gridcolor=GRID_COLOR, color=FONT_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR, color=FONT_COLOR),
            ),
        )
        st.plotly_chart(fig_polar, use_container_width=True)

    # ── Chart 4c: Heatmap jam x hari-minggu ──
    with col_b:
        if 'date' in df.columns:
            df['dayofweek'] = df['date'].dt.day_name()
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_hour = df.groupby(['dayofweek','hour']).size().reset_index(name='count')
            dow_pivot = dow_hour.pivot(index='dayofweek', columns='hour', values='count').fillna(0)
            dow_pivot = dow_pivot.reindex([d for d in day_order if d in dow_pivot.index])

            fig_dow = go.Figure(go.Heatmap(
                z=dow_pivot.values, x=dow_pivot.columns.tolist(),
                y=dow_pivot.index.tolist(), colorscale='Reds',
                colorbar=dict(title="Jumlah", tickfont=dict(color=FONT_COLOR)),
            ))
            apply_base(fig_dow, "Heatmap Jam × Hari (jumlah kejahatan)")
            fig_dow.update_layout(height=360)
            st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box warning">
        <strong>Interpretasi:</strong> Kejahatan tidak terdistribusi merata sepanjang hari. 
        Terdapat dua pola utama: <strong>lonjakan pada siang hari (sekitar jam 10–12)</strong> 
        yang berkaitan dengan aktivitas publik (theft di pusat perbelanjaan, kantor), 
        dan <strong>puncak di malam hari (jam 18–23)</strong> yang umumnya terkait dengan 
        kejahatan jalanan dan kejahatan di bawah pengaruh alkohol. 
        Jam-jam dini hari (01–05) mencatat angka paling rendah. 
        <strong>Polar chart</strong> memperlihatkan pola melingkar yang tidak simetris — 
        menunjukkan dominasi kejahatan di separuh waktu aktif (siang-malam). 
        Heatmap hari × jam memperlihatkan bahwa <strong>akhir pekan malam (Jumat–Sabtu)</strong> 
        adalah kombinasi waktu paling rawan, relevan untuk strategi patroli.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB Q5: K-MEANS
# ════════════════════════════════════════════
with tab_q5:
    st.markdown('<div class="q-badge">Pertanyaan 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Klasterisasi District dengan K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Berdasarkan jumlah kejahatan dan arrest rate, district mana yang memiliki karakteristik serupa?</div>', unsafe_allow_html=True)

    district_feats = make_district_features(df)
    cluster_df, sil_score = run_kmeans(district_feats)

    # Identifikasi cluster prioritas (high crime, low arrest)
    summary = cluster_df.groupby('cluster_label').agg(
        n_district=('district','count'),
        avg_crime=('jumlah_kejahatan','mean'),
        avg_arrest=('arrest_rate_pct','mean'),
        avg_domestic=('domestic_rate_pct','mean'),
        districts=('district', lambda x: sorted(x.tolist())),
    ).reset_index()
    mms = MinMaxScaler()
    summary[['cn','an']] = mms.fit_transform(summary[['avg_crime','avg_arrest']])
    summary['risk_score'] = summary['cn'] - summary['an']
    most_dangerous = summary.loc[summary['risk_score'].idxmax(), 'cluster_label']

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="metric-card">
        <div class="metric-label">🤖 Silhouette Score</div>
        <div class="metric-value">{sil_score:.3f}</div>
        <div class="metric-delta">Kualitas klaster (0–1)</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <div class="metric-label">🏘️ Jumlah Klaster</div>
        <div class="metric-value">6</div>
        <div class="metric-delta">K optimal</div>
    </div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="metric-card">
        <div class="metric-label">⚠️ Prioritas Intervensi</div>
        <div class="metric-value">{most_dangerous}</div>
        <div class="metric-delta">High crime, low arrest</div>
    </div>""", unsafe_allow_html=True)

    # ── Chart 5a: Scatter Cluster ──
    fig_cl = px.scatter(
        cluster_df, x='jumlah_kejahatan', y='arrest_rate_pct',
        color='cluster_label', text='district',
        size='domestic_rate_pct',
        color_discrete_sequence=PALETTE,
        labels={'jumlah_kejahatan':'Jumlah Kejahatan','arrest_rate_pct':'Arrest Rate (%)','cluster_label':'Cluster'},
    )
    fig_cl.update_traces(textposition='top center', textfont=dict(size=9, color='#c9d1d9'))
    apply_base(fig_cl, "Klasterisasi District — Jumlah Kejahatan vs Arrest Rate (ukuran = Domestic Rate)")
    fig_cl.update_layout(height=420)
    st.plotly_chart(fig_cl, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── Chart 5b: Radar/spider per cluster ──
    with col_a:
        categories = ['avg_crime','avg_arrest','avg_domestic']
        labels_radar = ['Avg Kejahatan','Avg Arrest (%)','Avg Domestic (%)']
        mms2 = MinMaxScaler()
        radar_norm = summary.copy()
        radar_norm[categories] = mms2.fit_transform(summary[categories])

        fig_radar = go.Figure()
        for _, row in radar_norm.iterrows():
            vals = [row[c] for c in categories] + [row[categories[0]]]
            is_prio = row['cluster_label'] == most_dangerous
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=labels_radar + [labels_radar[0]],
                name=row['cluster_label'] + (' ⚠️' if is_prio else ''),
                line=dict(width=3 if is_prio else 1.5,
                          color='#f85149' if is_prio else None),
                fill='toself', opacity=0.6 if is_prio else 0.2,
            ))
        apply_base(fig_radar, "Profil Tiap Cluster (Normalized Radar)")
        fig_radar.update_layout(
            height=380, polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, gridcolor=GRID_COLOR, color=FONT_COLOR, range=[0,1]),
                angularaxis=dict(gridcolor=GRID_COLOR, color=FONT_COLOR),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Chart 5c: Bar avg crime per cluster ──
    with col_b:
        fig_bar_cl = go.Figure()
        fig_bar_cl.add_trace(go.Bar(
            name='Avg Kejahatan',
            x=summary['cluster_label'], y=summary['avg_crime'],
            marker_color=['#f85149' if cl == most_dangerous else ACCENT for cl in summary['cluster_label']],
            text=summary['avg_crime'].round(1), textposition='outside', textfont=dict(color=FONT_COLOR),
        ))
        fig_bar_cl.add_trace(go.Scatter(
            name='Avg Arrest Rate (%)',
            x=summary['cluster_label'], y=summary['avg_arrest'],
            mode='lines+markers', yaxis='y2',
            line=dict(color='#3fb950', width=2.5), marker=dict(size=8),
        ))
        apply_base(fig_bar_cl, "Avg Kejahatan & Arrest Rate per Cluster")
        fig_bar_cl.update_layout(
            height=380,
            yaxis=dict(title="Rata-rata Kejahatan"),
            yaxis2=dict(
    title=dict(
        text="Arrest Rate (%)",
        font=dict(color='#3fb950')
    ),
    overlaying='y',
    side='right',
    showgrid=False
),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_bar_cl, use_container_width=True)

    # ── Tabel profil cluster ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Profil Detail Tiap Cluster**")
    display_summary = summary[['cluster_label','districts','n_district','avg_crime','avg_arrest','avg_domestic']].copy()
    display_summary.columns = ['Cluster','District','Jml District','Avg Kejahatan','Avg Arrest (%)','Avg Domestic (%)']
    display_summary['Avg Kejahatan']  = display_summary['Avg Kejahatan'].round(1)
    display_summary['Avg Arrest (%)']  = display_summary['Avg Arrest (%)'].round(1)
    display_summary['Avg Domestic (%)'] = display_summary['Avg Domestic (%)'].round(1)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="insight-box danger">
        <strong>Interpretasi:</strong> K-Means membagi {district_feats['district'].nunique()} district Chicago 
        menjadi <strong>6 klaster</strong> berdasarkan kemiripan profil keamanan (jumlah kejahatan, arrest rate, domestic rate). 
        Silhouette score = <strong>{sil_score:.3f}</strong> mengindikasikan 
        {'kualitas klaster yang cukup baik' if sil_score > 0.3 else 'pemisahan klaster yang moderat'}. 
        <strong>{most_dangerous}</strong> diidentifikasi sebagai prioritas intervensi karena memiliki kombinasi 
        <em>jumlah kejahatan tinggi namun arrest rate rendah</em> — pola paling berbahaya karena 
        mencerminkan ketidakmampuan penegakan hukum mengejar volume kejahatan. 
        Radar chart memperlihatkan perbedaan profil yang jelas antar klaster, mendukung strategi 
        penanganan yang <strong>berbeda-beda per klaster</strong> alih-alih kebijakan seragam.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.78rem; padding: 0.8rem 0">
    Analisis Tren dan Pola Kriminalitas di Chicago 2020–2025 · Proyek Akhir SST60202 — Analisis Data Berbasis Cloud<br>
    Sumber data: BigQuery Public Dataset <code>bigquery-public-data.chicago_crime.crime</code>
</div>
""", unsafe_allow_html=True)
