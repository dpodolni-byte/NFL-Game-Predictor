import streamlit as st
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@400;500;600&display=swap');

:root {
  --bg:      #080f1e;
  --card:    #0d1a2d;
  --card2:   #111f33;
  --border:  #1a2e47;
  --navy:    #013369;
  --red:     #D50A0A;
  --gold:    #F5A623;
  --text:    #e8edf5;
  --muted:   #5a7a9a;
  --green:   #22c55e;
  --amber:   #f59e0b;
}

/* Full page background */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
  background: var(--bg) !important;
  font-family: 'Barlow', sans-serif;
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
section.main > div { padding: 0 !important; }
.block-container { padding: 0 0 60px !important; max-width: 860px !important; }

/* ── Header ── */
.site-header {
  background: linear-gradient(160deg, #011c40 0%, #010e22 100%);
  border-bottom: 3px solid var(--red);
  padding: 36px 32px 28px;
  text-align: center;
  margin-bottom: 0;
}
.site-header h1 {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2.8rem; font-weight: 900;
  color: #fff; margin: 0;
  letter-spacing: 1px; line-height: 1;
  text-transform: uppercase;
}
.site-header .subtitle {
  color: var(--muted); font-size: .82rem;
  margin: 8px 0 0; letter-spacing: 1px; text-transform: uppercase;
}
.accuracy-pill {
  display: inline-block; margin-top: 12px;
  background: rgba(245,166,35,.12); border: 1px solid rgba(245,166,35,.35);
  color: var(--gold); border-radius: 999px;
  padding: 4px 16px; font-size: .75rem; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase;
  font-family: 'Barlow Condensed', sans-serif;
}

/* ── Picker section ── */
.picker-wrap {
  background: var(--card);
  border-bottom: 1px solid var(--border);
  padding: 28px 32px 24px;
}

/* ── Team logos ── */
.matchup-logos {
  display: flex; align-items: center;
  justify-content: center; gap: 0; margin-bottom: 22px;
}
.team-side { flex: 1; text-align: center; }
.team-side .role {
  font-size: .6rem; font-weight: 700; letter-spacing: 3px;
  color: var(--muted); text-transform: uppercase; margin-bottom: 10px;
}
.logo-ring {
  width: 100px; height: 100px; border-radius: 50%;
  border: 2px solid var(--border); background: var(--card2);
  display: inline-flex; align-items: center; justify-content: center;
  overflow: hidden; transition: border-color .25s;
}
.logo-ring img { width: 80px; height: 80px; object-fit: contain; }
.logo-ring.active { border-color: var(--gold); }
.team-name-lbl {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem; font-weight: 700; margin-top: 8px; color: #fff;
}
.vs-sep {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.8rem; font-weight: 900; color: var(--muted);
  padding: 0 24px; padding-top: 20px; flex-shrink: 0;
}

/* Streamlit select overrides */
[data-testid="stSelectbox"] > div > div {
  background: var(--card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
[data-testid="stSelectbox"] label {
  color: var(--muted) !important;
  font-size: .65rem !important;
  font-weight: 700 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
  width: 100%; padding: 14px;
  background: linear-gradient(135deg, var(--red), #a00808) !important;
  color: #fff !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 1rem !important; font-weight: 800 !important;
  letter-spacing: 2px !important; text-transform: uppercase !important;
  border: none !important; border-radius: 10px !important;
  transition: opacity .2s !important;
}
[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ── Result: Winner Banner ── */
.winner-banner {
  background: linear-gradient(160deg, #012248 0%, #010d1e 100%);
  border-top: 1px solid var(--border);
  border-bottom: 3px solid var(--gold);
  padding: 36px 32px 30px;
  text-align: center;
}
.winner-banner .predicted-lbl {
  font-size: .6rem; font-weight: 700; letter-spacing: 4px;
  color: var(--gold); text-transform: uppercase;
  font-family: 'Barlow Condensed', sans-serif;
}
.winner-banner .winner-logo-wrap {
  width: 110px; height: 110px; border-radius: 50%;
  border: 3px solid var(--gold); background: rgba(255,255,255,.04);
  display: inline-flex; align-items: center; justify-content: center;
  margin: 14px auto 10px; overflow: hidden;
}
.winner-banner .winner-logo-wrap img { width: 88px; height: 88px; object-fit: contain; }
.winner-name {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2.6rem; font-weight: 900; color: #fff;
  text-transform: uppercase; letter-spacing: 1px; line-height: 1;
}
.conf-badge {
  display: inline-block; margin-top: 8px;
  background: rgba(245,166,35,.12); border: 1px solid rgba(245,166,35,.4);
  color: var(--gold); border-radius: 999px;
  padding: 3px 14px; font-size: .7rem; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase;
  font-family: 'Barlow Condensed', sans-serif;
}

/* ── Probability bar ── */
.prob-section { padding: 0 32px; margin: 20px 0 0; }
.prob-labels {
  display: flex; justify-content: space-between;
  font-size: .82rem; font-weight: 600; margin-bottom: 5px;
}
.prob-track {
  height: 12px; background: var(--card2);
  border-radius: 999px; overflow: hidden;
  border: 1px solid var(--border);
}
.prob-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--red) 0%, var(--navy) 100%);
  transition: width .5s ease;
}
.prob-pcts {
  display: flex; justify-content: space-between;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: .95rem; font-weight: 700; margin-top: 4px;
  color: var(--muted);
}

/* ── H2H notice ── */
.h2h-box {
  margin: 14px 32px 0;
  background: rgba(245,166,35,.06);
  border: 1px solid rgba(245,166,35,.2);
  border-radius: 8px; padding: 10px 14px;
  font-size: .8rem; color: #c8a64a; text-align: center;
}

/* ── Component breakdown section ── */
.breakdown-wrap {
  background: var(--card); padding: 28px 32px;
  border-top: 1px solid var(--border);
}
.breakdown-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: .6rem; font-weight: 700; letter-spacing: 3px;
  color: var(--muted); text-transform: uppercase; margin-bottom: 18px;
}
.comp-row { margin-bottom: 14px; }
.comp-label {
  font-size: .6rem; font-weight: 700; letter-spacing: 2px;
  color: var(--muted); text-transform: uppercase; margin-bottom: 4px;
}
.comp-teams {
  display: flex; align-items: center; gap: 8px;
}
.comp-team-lbl { font-size: .75rem; font-weight: 600; min-width: 86px; }
.comp-bar-track {
  flex: 1; height: 7px; background: var(--card2);
  border-radius: 999px; overflow: hidden;
}
.comp-bar-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--red), #1a4a8a);
}
.comp-team-lbl.right { text-align: right; }

/* ── Stat cards grid ── */
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 12px; margin-top: 20px;
}
.stat-card {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px;
}
.stat-card-title {
  font-size: .55rem; font-weight: 700; letter-spacing: 2.5px;
  color: var(--muted); text-transform: uppercase; margin-bottom: 10px;
  font-family: 'Barlow Condensed', sans-serif;
}
.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,.04);
  font-size: .8rem;
}
.stat-row:last-child { border-bottom: none; }
.sl { color: var(--muted); }
.sv { font-weight: 700; color: var(--text); }
.sv.hot  { color: var(--green); }
.sv.cold { color: var(--red);   }
.sv.neut { color: var(--amber); }
.sv.win  { color: var(--gold);  }
.boost-note { color: var(--amber); font-size: .68rem; }

/* ── Blended score callout ── */
.score-callout {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 20px; margin-top: 12px;
  text-align: center;
}
.score-callout .sc-lbl {
  font-size: .55rem; font-weight: 700; letter-spacing: 2.5px;
  color: var(--muted); text-transform: uppercase;
  font-family: 'Barlow Condensed', sans-serif;
}
.score-numbers {
  display: flex; justify-content: center; gap: 40px; margin-top: 8px;
}
.score-team { text-align: center; }
.score-team .stn {
  font-size: .72rem; color: var(--muted); font-weight: 600;
}
.score-team .stv {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.7rem; font-weight: 900;
}
.stv.winner-clr { color: var(--gold); }
.stv.loser-clr  { color: var(--muted); }

/* ── Weight pills ── */
.weight-pills {
  display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px;
}
.wp {
  font-size: .62rem; font-weight: 700; padding: 3px 10px;
  border-radius: 6px; letter-spacing: 1px;
  font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase;
}
.wp-elo    { background: rgba(1,51,105,.5);   color: #7bb3f0; border:1px solid rgba(123,179,240,.2); }
.wp-avg    { background: rgba(34,197,94,.1);  color: #4ade80; border:1px solid rgba(34,197,94,.2); }
.wp-streak { background: rgba(245,158,11,.1); color: #fbbf24; border:1px solid rgba(245,158,11,.2); }
.wp-str    { background: rgba(213,10,10,.1);  color: #f87171; border:1px solid rgba(213,10,10,.2); }

/* ── Footer ── */
.site-footer {
  text-align: center; padding: 20px;
  font-size: .72rem; color: var(--muted);
  border-top: 1px solid var(--border);
}

/* hide streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), 'NFL_Project_Test_6.xlsx')

@st.cache_data(show_spinner="Loading NFL data…")
def load_data():
    df_raw  = pd.read_excel(DATA_FILE, sheet_name='Raw_Games')
    df_avg  = pd.read_excel(DATA_FILE, sheet_name='Rolling_3Yr_Avg')
    df_elo  = pd.read_excel(DATA_FILE, sheet_name='ELO_Ratings',  header=2)
    df_str  = pd.read_excel(DATA_FILE, sheet_name='Win_Streak',   header=2)
    df_ts   = pd.read_excel(DATA_FILE, sheet_name='Team_Strength', header=2)
    df_cfg  = pd.read_excel(DATA_FILE, sheet_name='Config',        header=None)

    # Clean Team_Strength column names (have newlines from Excel)
    df_ts.columns = [str(c).split('\n')[0].strip() for c in df_ts.columns]

    # ── Config weights ────────────────────────────────────────────────────────
    def cfg(label, default):
        for _, r in df_cfg.iterrows():
            if str(r.iloc[0]).strip() == label:
                try: return float(r.iloc[1])
                except: pass
        return default

    weights = {
        'elo':      cfg('Weight — ELO',            0.71),
        '3yr':      cfg('Weight — 3yr Avg',         0.108),
        'streak':   cfg('Weight — Win Streak',       0.029),
        'strength': cfg('Weight — Team Strength',    0.153),
        'boost':    cfg('Home Advantage Boost',      0.06),
    }

    # ── Name → abbreviation ───────────────────────────────────────────────────
    n2a = dict(zip(df_raw['Away Team'], df_raw['Away Abbr']))
    n2a.update(dict(zip(df_raw['Home Team'], df_raw['Home Abbr'])))

    # ── 3yr avg lookup ────────────────────────────────────────────────────────
    avg_lkp = {
        (int(r['Season']), r['Team']): r['AvgWinPct']
        for _, r in df_avg.iterrows() if pd.notna(r.get('AvgWinPct'))
    }

    # ── ELO: end-of-season post-ELO per team ─────────────────────────────────
    elo_lkp = {}
    for _, r in df_elo.iterrows():
        s = int(r['Season'])
        elo_lkp[(s, r['Away Team'])] = float(r['Post Away ELO'])
        elo_lkp[(s, r['Home Team'])] = float(r['Post Home ELO'])

    # ── Streak: end-of-season per team ───────────────────────────────────────
    str_lkp = {}
    for _, r in df_str.iterrows():
        s = int(r['Season'])
        str_lkp[(s, r['Away Team'])] = (r['Away Streak'], float(r['Away Streak Score']))
        str_lkp[(s, r['Home Team'])] = (r['Home Streak'], float(r['Home Streak Score']))

    # ── Team Strength lookup ──────────────────────────────────────────────────
    ts_lkp = {}
    for _, r in df_ts.iterrows():
        ts_val = r.get('Team Strength')
        if pd.notna(ts_val):
            ts_lkp[(int(r['Season']), r['Team Name'])] = float(ts_val)

    teams   = sorted(set(df_raw['Away Team']) | set(df_raw['Home Team']))
    seasons = sorted(df_raw['Season'].unique().tolist(), reverse=True)

    return {
        'df_raw': df_raw, 'avg_lkp': avg_lkp, 'elo_lkp': elo_lkp,
        'str_lkp': str_lkp, 'ts_lkp': ts_lkp, 'n2a': n2a,
        'weights': weights, 'teams': teams, 'seasons': seasons,
    }

# ── Prediction engine ─────────────────────────────────────────────────────────
def predict(season, away, home, d):
    s  = int(season)
    w  = d['weights']
    we, w3, ws, wt, boost = w['elo'], w['3yr'], w['streak'], w['strength'], w['boost']

    # 3yr avg
    a_avg = d['avg_lkp'].get((s, away))
    h_avg = d['avg_lkp'].get((s, home))
    avg_note = None
    if a_avg is None or h_avg is None:
        for dy in range(1, 8):
            if a_avg is None: a_avg = d['avg_lkp'].get((s+dy, away))
            if h_avg is None: h_avg = d['avg_lkp'].get((s+dy, home))
        a_avg = a_avg or 0.5; h_avg = h_avg or 0.5
        avg_note = "3yr avg not in data for this season — using nearest available"

    # ELO (1500 default)
    a_elo = d['elo_lkp'].get((s, away), 1500.0)
    h_elo = d['elo_lkp'].get((s, home), 1500.0)
    etot  = a_elo + h_elo
    a_en  = a_elo / etot; h_en = h_elo / etot

    # Streak
    a_sl, a_ss = d['str_lkp'].get((s, away), ('Neutral', 0.5))
    h_sl, h_ss = d['str_lkp'].get((s, home), ('Neutral', 0.5))

    # Team strength (normalised)
    a_ts = d['ts_lkp'].get((s, away), 50.0)
    h_ts = d['ts_lkp'].get((s, home), 50.0)
    ts_t = a_ts + h_ts
    a_tn = a_ts / ts_t if ts_t > 0 else 0.5
    h_tn = h_ts / ts_t if ts_t > 0 else 0.5

    # Blended scores
    a_sc = we*a_en + w3*a_avg               + ws*a_ss + wt*a_tn
    h_sc = we*h_en + w3*(h_avg + boost)     + ws*h_ss + wt*h_tn

    winner = home if h_sc >= a_sc else away
    gap    = h_sc - a_sc
    h_prob = 1 / (1 + np.exp(-gap * 22)); a_prob = 1 - h_prob
    mg     = abs(gap)
    conf   = ('Toss-Up'             if mg < 0.015 else
              'Slight Edge'         if mg < 0.04  else
              'Moderate Confidence' if mg < 0.08  else 'Strong Favourite')

    # H2H
    df = d['df_raw']
    h2h = df[(df['Season']==s) & (
        ((df['Away Team']==away) & (df['Home Team']==home)) |
        ((df['Away Team']==home) & (df['Home Team']==away))
    )]
    h2h_str = None
    if len(h2h):
        g = h2h.iloc[0]
        h2h_str = (f"They played in Week {int(g['Week'])}: "
                   f"{g['Away Team']} {int(g['Away Score'])}–{int(g['Home Score'])} "
                   f"{g['Home Team']} → {g['Actual Winner']} won")

    return dict(
        winner=winner, loser=(away if winner==home else home), conf=conf,
        a_sc=round(a_sc,4), h_sc=round(h_sc,4),
        a_prob=round(a_prob*100,1), h_prob=round(h_prob*100,1),
        a_avg=a_avg, h_avg=h_avg, boost=boost,
        a_elo=int(a_elo), h_elo=int(h_elo),
        a_en=round(a_en,4), h_en=round(h_en,4),
        a_sl=a_sl, h_sl=h_sl,
        a_ts=round(a_ts,1), h_ts=round(h_ts,1),
        a_tn=round(a_tn,4), h_tn=round(h_tn,4),
        a_ss=a_ss, h_ss=h_ss,
        h2h=h2h_str, avg_note=avg_note,
        away=away, home=home, season=s,
        weights=w,
    )

def logo(abbr):
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"

def streak_cls(lbl):
    return {'Hot':'hot','Cold':'cold'}.get(lbl,'neut')

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
  <h1>🏈 NFL Game Predictor</h1>
  <div class="subtitle">ELO · 3-Year Rolling Avg · Momentum Streak · Team Strength</div>
  <div class="accuracy-pill">Model Accuracy: 65.5%</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Game Predictor", "📅  2026 Season Preview"])

with tab1:
    # ── Team + Season pickers ─────────────────────────────────────────────────────
    st.markdown('<div class="picker-wrap">', unsafe_allow_html=True)

    c1, cv, c2 = st.columns([5, 1, 5])

    with c1:
        away = st.selectbox("🛫 Away Team", data['teams'],
                            index=data['teams'].index('Eagles') if 'Eagles' in data['teams'] else 0,
                            key='sel_away')
    with cv:
        st.markdown('<div class="vs-sep">@</div>', unsafe_allow_html=True)
    with c2:
        home = st.selectbox("🏠 Home Team", data['teams'],
                            index=data['teams'].index('Chiefs') if 'Chiefs' in data['teams'] else 1,
                            key='sel_home')

    a_abbr = data['n2a'].get(away, '')
    h_abbr = data['n2a'].get(home, '')

    # Logo display
    st.markdown(f"""
    <div class="matchup-logos">
      <div class="team-side">
        <div class="role">Away</div>
        <div class="logo-ring {'active' if a_abbr else ''}">
          <img src="{logo(a_abbr)}" alt="{away}" onerror="this.style.display='none'">
        </div>
        <div class="team-name-lbl">{away}</div>
      </div>
      <div class="vs-sep">VS</div>
      <div class="team-side">
        <div class="role">Home</div>
        <div class="logo-ring {'active' if h_abbr else ''}">
          <img src="{logo(h_abbr)}" alt="{home}" onerror="this.style.display='none'">
        </div>
        <div class="team-name-lbl">{home}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns([3, 3, 2])
    with sc1:
        season = st.selectbox("📅 Season", data['seasons'], key='sel_season')
    with sc2:
        st.write("")
    with sc3:
        st.write("")

    predict_btn = st.button("⚡  Predict Winner", disabled=(away == home), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Validation ────────────────────────────────────────────────────────────────
    if away == home:
        st.markdown("""
        <div style="background:rgba(213,10,10,.08);border:1px solid rgba(213,10,10,.25);
        border-radius:8px;padding:10px 16px;margin:0;text-align:center;
        font-size:.82rem;color:#f87171">
        Please select two different teams.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Run prediction only on button click ──────────────────────────────────────
    if predict_btn:
        st.session_state.result = predict(season, away, home, data)

    if 'result' not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:48px 32px;color:#3a5a7a;
        font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
        letter-spacing:1px;text-transform:uppercase">
          Select two teams and a season, then click Predict Winner
        </div>""", unsafe_allow_html=True)
        st.stop()

    res = st.session_state.result

    # ── Winner Banner ─────────────────────────────────────────────────────────────
    w_abbr = data['n2a'].get(res['winner'], '')
    st.markdown(f"""
    <div class="winner-banner">
      <div class="predicted-lbl">Predicted Winner</div>
      <div>
        <div class="winner-logo-wrap">
          <img src="{logo(w_abbr)}" alt="{res['winner']}">
        </div>
      </div>
      <div class="winner-name">{res['winner']}</div>
      <div class="conf-badge">{res['conf']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="prob-section">
      <div class="prob-labels">
        <span style="color:{'#fff' if res['a_prob'] > res['h_prob'] else '#5a7a9a'};font-weight:{'800' if res['a_prob']>res['h_prob'] else '500'}">
          {away}
        </span>
        <span style="color:{'#fff' if res['h_prob'] > res['a_prob'] else '#5a7a9a'};font-weight:{'800' if res['h_prob']>res['a_prob'] else '500'}">
          {home}
        </span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:{res['a_prob']}%"></div>
      </div>
      <div class="prob-pcts">
        <span style="color:{'#F5A623' if res['a_prob']>res['h_prob'] else '#5a7a9a'}">{res['a_prob']}%</span>
        <span style="color:{'#F5A623' if res['h_prob']>res['a_prob'] else '#5a7a9a'}">{res['h_prob']}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if res['h2h']:
        st.markdown(f'<div class="h2h-box">📋 {res["h2h"]}</div>', unsafe_allow_html=True)
    if res['avg_note']:
        st.markdown(f'<div class="h2h-box" style="color:#7bb3f0">ℹ️ {res["avg_note"]}</div>',
                    unsafe_allow_html=True)

    # ── Component Breakdown ───────────────────────────────────────────────────────
    st.markdown('<div class="breakdown-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="breakdown-title">Score Component Breakdown</div>',
                unsafe_allow_html=True)

    def comp_bar_html(label, av, hv, al, hl, fmt='.3f', pct=False):
        tot = av + hv
        aw  = av / tot * 100 if tot > 0 else 50
        ac  = '#22c55e' if av >= hv else '#5a7a9a'
        hc  = '#22c55e' if hv >= av else '#5a7a9a'
        avf = f"{av*100:.1f}%" if pct else format(av, fmt)
        hvf = f"{hv*100:.1f}%" if pct else format(hv, fmt)
        return f"""
        <div class="comp-row">
          <div class="comp-label">{label}</div>
          <div class="comp-teams">
            <span class="comp-team-lbl" style="color:{ac}">{al}: {avf}</span>
            <div class="comp-bar-track">
              <div class="comp-bar-fill" style="width:{aw:.1f}%"></div>
            </div>
            <span class="comp-team-lbl right" style="color:{hc}">{hl}: {hvf}</span>
          </div>
        </div>"""

    st.markdown(
        comp_bar_html("ELO Rating (normalised)", res['a_en'], res['h_en'], away, home, '.4f') +
        comp_bar_html("3-Year Rolling Avg Win %", res['a_avg'], res['h_avg'], away, home, pct=True) +
        comp_bar_html("Momentum Streak Score", res['a_ss'], res['h_ss'], away, home, '.1f') +
        comp_bar_html("Team Strength (normalised)", res['a_tn'], res['h_tn'], away, home, '.4f'),
        unsafe_allow_html=True
    )

    # ── Stat Cards ────────────────────────────────────────────────────────────────
    sc_a = streak_cls(res['a_sl']); sc_h = streak_cls(res['h_sl'])
    win_is_away = res['winner'] == away

    st.markdown(f"""
    <div class="stats-grid">

      <div class="stat-card">
        <div class="stat-card-title">3yr Rolling Avg Win %</div>
        <div class="stat-row">
          <span class="sl">{away}</span>
          <span class="sv {'win' if win_is_away else ''}">{res['a_avg']*100:.1f}%</span>
        </div>
        <div class="stat-row">
          <span class="sl">{home}</span>
          <span class="sv {'win' if not win_is_away else ''}">{res['h_avg']*100:.1f}%</span>
        </div>
        <div class="stat-row">
          <span class="sl boost-note">🏠 Home boost</span>
          <span class="sv boost-note">+{res['boost']*100:.1f}%</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-card-title">End-of-Season ELO</div>
        <div class="stat-row">
          <span class="sl">{away}</span>
          <span class="sv {'win' if win_is_away else ''}">{res['a_elo']:,}</span>
        </div>
        <div class="stat-row">
          <span class="sl">{home}</span>
          <span class="sv {'win' if not win_is_away else ''}">{res['h_elo']:,}</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-card-title">Momentum Streak</div>
        <div class="stat-row">
          <span class="sl">{away}</span>
          <span class="sv {sc_a}">{res['a_sl']}</span>
        </div>
        <div class="stat-row">
          <span class="sl">{home}</span>
          <span class="sv {sc_h}">{res['h_sl']}</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-card-title">Team Strength (0–100)</div>
        <div class="stat-row">
          <span class="sl">{away}</span>
          <span class="sv {'win' if win_is_away else ''}">{res['a_ts']:.1f}</span>
        </div>
        <div class="stat-row">
          <span class="sl">{home}</span>
          <span class="sv {'win' if not win_is_away else ''}">{res['h_ts']:.1f}</span>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── Final blended scores ──────────────────────────────────────────────────────
    a_winner = res['a_sc'] > res['h_sc']
    st.markdown(f"""
    <div class="score-callout">
      <div class="sc-lbl">Final Blended Score</div>
      <div class="score-numbers">
        <div class="score-team">
          <div class="stn">{away}</div>
          <div class="stv {'winner-clr' if a_winner else 'loser-clr'}">{res['a_sc']:.4f}</div>
        </div>
        <div class="score-team">
          <div class="stn">{home}</div>
          <div class="stv {'winner-clr' if not a_winner else 'loser-clr'}">{res['h_sc']:.4f}</div>
        </div>
      </div>
      <div class="weight-pills">
        <span class="wp wp-elo">ELO {res['weights']['elo']*100:.0f}%</span>
        <span class="wp wp-avg">3yr Avg {res['weights']['3yr']*100:.1f}%</span>
        <span class="wp wp-streak">Streak {res['weights']['streak']*100:.1f}%</span>
        <span class="wp wp-str">Team Strength {res['weights']['strength']*100:.1f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



with tab2:
    import math

    # ── 2026 hardcoded data ────────────────────────────────────────────────────
    # 2025 season records (source: 2026 NFL Draft order, Wikipedia)
    RECORDS_2025 = {
        'Raiders':    (3,  14, 0), 'Jets':       (3,  14, 0),
        'Cardinals':  (3,  14, 0), 'Titans':     (3,  14, 0),
        'Giants':     (4,  13, 0), 'Browns':     (5,  12, 0),
        'Commanders': (5,  12, 0), 'Saints':     (6,  11, 0),
        'Chiefs':     (6,  11, 0), 'Bengals':    (6,  11, 0),
        'Dolphins':   (7,  10, 0), 'Cowboys':    (7,   9, 1),
        'Ravens':     (8,   9, 0), 'Buccaneers': (8,   9, 0),
        'Panthers':   (8,   9, 0), 'Falcons':    (8,   9, 0),
        'Lions':      (9,   8, 0), 'Vikings':    (9,   8, 0),
        'Packers':    (9,   7, 1), 'Jaguars':    (9,   8, 0),
        'Colts':      (8,   9, 0), 'Steelers':   (10,  7, 0),
        'Chargers':   (11,  6, 0), 'Eagles':     (11,  6, 0),
        'Bears':      (11,  6, 0), 'Bills':      (12,  5, 0),
        'SF49ers':    (12,  5, 0), 'Texans':     (12,  5, 0),
        'Rams':       (12,  5, 0), 'Broncos':    (14,  3, 0),
        'Patriots':   (14,  3, 0), 'Seahawks':   (14,  3, 0),
    }

    # NFL divisions
    DIVISIONS = {
        'AFC North': ['Ravens', 'Browns', 'Bengals', 'Steelers'],
        'AFC South': ['Texans', 'Colts', 'Jaguars', 'Titans'],
        'AFC East':  ['Bills', 'Patriots', 'Jets', 'Dolphins'],
        'AFC West':  ['Broncos', 'Chiefs', 'Raiders', 'Chargers'],
        'NFC North': ['Lions', 'Packers', 'Vikings', 'Bears'],
        'NFC South': ['Buccaneers', 'Saints', 'Falcons', 'Panthers'],
        'NFC East':  ['Eagles', 'Cowboys', 'Commanders', 'Giants'],
        'NFC West':  ['Seahawks', 'Rams', 'SF49ers', 'Cardinals'],
    }

    # Draft & offseason adjustments (applied to end-of-2025 ELO as point boost)
    # Scale: +1 pt = roughly equivalent to ~half a win improvement
    ADJUSTMENTS = {
        # (label, position, round_or_type, elo_boost, emoji)
        'Raiders':    [('Fernando Mendoza', 'QB', 'R1 #1', +6.0, '🔴'),
                       ('Franchise rebuild', 'Org', 'Offseason', -1.0, '🔄')],
        'Cardinals':  [('Jeremiyah Love', 'RB', 'R1 #3', +2.0, '🏃'),
                       ('Carson Beck', 'QB', 'R3', +1.0, '🎯')],
        'Titans':     [('Carnell Tate', 'WR', 'R1 #4', +2.5, '⚡')],
        'Browns':     [('Spencer Fano', 'OT', 'R1', +2.0, '💪'),
                       ('2nd R1 pick (via JAX)', 'Draft', 'R1', +1.5, '🎯')],
        'Cowboys':    [('Caleb Downs', 'S', 'R1 #11', +2.0, '🛡'),
                       ('Quinnen Williams', 'DT', 'Trade', +1.5, '🔄'),
                       ('George Pickens', 'WR', 'Trade', +1.5, '🔄'),
                       ('Micah Parsons', 'EDGE', 'Traded away', -4.0, '📤')],
        'Texans':     [('Rueben Bain Jr.', 'EDGE', 'R1 #15', +2.5, '💥'),
                       ('Laremy Tunsil', 'OT', 'Trade', +1.5, '🔄')],
        'Lions':      [('Blake Miller', 'OT', 'R1 #17', +2.0, '🦁')],
        'Giants':     [('Olaivavega Ioane', 'IOL', 'R1', +2.0, '🔵')],
        'Eagles':     [('Makai Lemon', 'WR', 'R1 #23', +1.5, '🦅'),
                       ('Jaelan Phillips', 'EDGE', 'Trade', +1.5, '🔄')],
        'Broncos':    [('Omar Cooper Jr.', 'WR', 'R1 #30', +1.5, '🐴')],
        'Seahawks':   [('Jadarian Price', 'RB', 'R1 #32', +1.5, '🦅'),
                       ('Super Bowl champs', 'Momentum', '2025', +2.0, '🏆')],
        'Packers':    [('Micah Parsons', 'EDGE', 'Trade (from DAL)', +4.0, '🔄')],
        'Jets':       [('Sauce Gardner', 'CB', 'Trade (from IND)', +2.5, '🔄'),
                       ('T.J. Parker', 'EDGE', 'R2', +1.0, '🎯')],
        'Patriots':   [('Lost SB — retool', 'Org', 'Offseason', -1.0, '🔄')],
        'Chiefs':     [('Traded up in draft', 'Draft', 'R1', +2.0, '🏹')],
        'Bills':      [('Draft capital deployed', 'Draft', 'R1-2', +1.0, '🎯')],
        'Bears':      [('Strong draft class', 'Draft', 'R1', +2.0, '🐻')],
        'Colts':      [('Lost Sauce Gardner', 'CB', 'Trade', -2.0, '📤')],
    }

    # ── Compute 2026 projected strength ───────────────────────────────────────
    def win_pct_2025(team):
        w, l, t = RECORDS_2025.get(team, (8, 9, 0))
        total = w + l + t
        return (w + 0.5*t) / total if total > 0 else 0.5

    elo_end_2025 = {}
    for team in data['teams']:
        elo_end_2025[team] = data['elo_lkp'].get((2025, team), 1500.0)

    team_adjustments = {}
    for team in data['teams']:
        adj = sum(x[2+1] for x in ADJUSTMENTS.get(team, []))
        team_adjustments[team] = adj

    # Composite 2026 score = ELO + adjustment + 2025 win rate signal
    composites = {}
    for team in data['teams']:
        base_elo = elo_end_2025[team]
        adj      = team_adjustments.get(team, 0)
        wpc      = win_pct_2025(team)
        # Blend: ELO dominates (as per our model), win rate adds signal, adj on top
        composites[team] = (base_elo + adj * 8) * 0.75 + wpc * 1000 * 0.25

    vals = list(composites.values())
    vmin, vmax = min(vals), max(vals)

    def proj_wins(team):
        v = composites[team]
        # Map to 3–14 wins
        norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        return round((3 + norm * 11) * 2) / 2  # round to nearest 0.5

    # ── UI ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#011c40,#020a18);
    border-bottom:3px solid #D50A0A;padding:22px 24px 18px;margin-bottom:0">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
      font-weight:900;color:#fff;text-transform:uppercase;letter-spacing:1px">
        📅 2026 NFL Season Preview
      </div>
      <div style="color:#5a7a9a;font-size:.78rem;margin-top:4px;letter-spacing:1px;
      text-transform:uppercase">
        Projected standings · 2026 draft class · Offseason moves
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="background:rgba(245,166,35,.12);border:1px solid rgba(245,166,35,.3);
        color:#F5A623;border-radius:999px;padding:3px 12px;font-size:.7rem;font-weight:700;
        letter-spacing:1px;font-family:'Barlow Condensed',sans-serif;text-transform:uppercase">
          Draft Complete: April 23–25
        </span>
        <span style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);
        color:#4ade80;border-radius:999px;padding:3px 12px;font-size:.7rem;font-weight:700;
        letter-spacing:1px;font-family:'Barlow Condensed',sans-serif;text-transform:uppercase">
          SEA Defending Champions
        </span>
        <span style="background:rgba(90,122,154,.1);border:1px solid rgba(90,122,154,.25);
        color:#7aa8c8;border-radius:999px;padding:3px 12px;font-size:.7rem;font-weight:700;
        letter-spacing:1px;font-family:'Barlow Condensed',sans-serif;text-transform:uppercase">
          Full Schedule: Mid-May
        </span>
      </div>
    </div>
    <div style="background:rgba(245,166,35,.04);border:1px solid rgba(245,166,35,.15);
    border-radius:0 0 8px 8px;padding:8px 16px;margin-bottom:16px;
    font-size:.72rem;color:#8a7040">
      ⚠️ Projected wins are based on end-of-2025 ELO ratings + 2026 draft/trade adjustments.
      The official schedule releases mid-May 2026. Game-by-game predictions will be available
      in the Predictor tab once the schedule drops.
    </div>
    """, unsafe_allow_html=True)

    # ── Divisional Standings ───────────────────────────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:.65rem;
    font-weight:700;letter-spacing:3px;color:#5a7a9a;text-transform:uppercase;
    margin:4px 0 12px">2026 Projected Divisional Standings</div>""",
    unsafe_allow_html=True)

    afc_divs = ['AFC East', 'AFC North', 'AFC South', 'AFC West']
    nfc_divs = ['NFC East', 'NFC North', 'NFC South', 'NFC West']

    def division_table(div_name, teams_in_div):
        ranked = sorted(teams_in_div,
                        key=lambda t: (proj_wins(t), win_pct_2025(t)), reverse=True)
        rows = ""
        for i, team in enumerate(ranked):
            abbr   = data["n2a"].get(team, "")
            wins_2025 = RECORDS_2025.get(team, (0,0,0))
            rec_str = f"{wins_2025[0]}-{wins_2025[1]}" + (f"-{wins_2025[2]}" if wins_2025[2] else "")
            pw     = proj_wins(team)
            pl     = round(17 - pw, 1)
            champ_icon = " 🏆" if team == "Seahawks" else ""
            bg = "rgba(245,166,35,.06)" if i == 0 else "transparent"
            border = "border-left:3px solid #F5A623;" if i == 0 else "border-left:3px solid transparent;"
            rows += f"""<tr style="background:{bg}">
              <td style="padding:6px 8px;{border}">
                <div style="display:flex;align-items:center;gap:8px">
                  <img src="https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"
                       width="24" height="24"
                       style="border-radius:50%;background:#1a2235;padding:2px">
                  <span style="font-weight:{'800' if i==0 else '500'};
                  color:{'#fff' if i==0 else '#b0c4d8'}">{team}{champ_icon}</span>
                </div>
              </td>
              <td style="padding:6px 8px;text-align:center;color:#5a7a9a;font-size:.8rem">{rec_str}</td>
              <td style="padding:6px 8px;text-align:center;font-weight:700;
              color:{'#F5A623' if i==0 else '#e8edf5'}">{pw}</td>
              <td style="padding:6px 8px;text-align:center;color:#5a7a9a;font-size:.8rem">{pl}</td>
            </tr>"""
        return f"""
        <div style="background:#0d1a2d;border:1px solid #1a2e47;border-radius:10px;
        overflow:hidden;margin-bottom:12px">
          <div style="background:#011c40;padding:8px 12px;
          font-family:'Barlow Condensed',sans-serif;font-size:.75rem;font-weight:800;
          letter-spacing:2px;color:#7aa8c8;text-transform:uppercase">{div_name}</div>
          <table style="width:100%;border-collapse:collapse;font-size:.82rem">
            <thead><tr style="border-bottom:1px solid #1a2e47">
              <th style="padding:5px 8px;text-align:left;color:#3a5a7a;font-size:.65rem;
              letter-spacing:1px;font-weight:700;text-transform:uppercase">Team</th>
              <th style="padding:5px 8px;text-align:center;color:#3a5a7a;font-size:.65rem;
              letter-spacing:1px;font-weight:700;text-transform:uppercase">2025</th>
              <th style="padding:5px 8px;text-align:center;color:#3a5a7a;font-size:.65rem;
              letter-spacing:1px;font-weight:700;text-transform:uppercase">Proj W</th>
              <th style="padding:5px 8px;text-align:center;color:#3a5a7a;font-size:.65rem;
              letter-spacing:1px;font-weight:700;text-transform:uppercase">Proj L</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    col_afc, col_nfc = st.columns(2)
    with col_afc:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:.7rem;
        font-weight:700;letter-spacing:2px;color:#D50A0A;text-transform:uppercase;
        margin-bottom:8px">🏈 AFC</div>""", unsafe_allow_html=True)
        for d_name in afc_divs:
            st.markdown(division_table(d_name, DIVISIONS[d_name]), unsafe_allow_html=True)

    with col_nfc:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:.7rem;
        font-weight:700;letter-spacing:2px;color:#013369;text-transform:uppercase;
        margin-bottom:8px">🏈 NFC</div>""", unsafe_allow_html=True)
        for d_name in nfc_divs:
            st.markdown(division_table(d_name, DIVISIONS[d_name]), unsafe_allow_html=True)

    # ── Projected Playoffs ────────────────────────────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:.65rem;
    font-weight:700;letter-spacing:3px;color:#5a7a9a;text-transform:uppercase;
    margin:20px 0 12px">Projected Playoff Picture</div>""", unsafe_allow_html=True)

    def playoff_teams(conference_divs):
        div_winners = []
        wildcards_pool = []
        for dv in conference_divs:
            teams_sorted = sorted(DIVISIONS[dv],
                                  key=lambda t: (proj_wins(t), win_pct_2025(t)), reverse=True)
            div_winners.append(teams_sorted[0])
            wildcards_pool.extend(teams_sorted[1:])
        wildcards_pool.sort(key=lambda t: (proj_wins(t), win_pct_2025(t)), reverse=True)
        return div_winners, wildcards_pool[:3]

    afc_winners, afc_wc = playoff_teams(afc_divs)
    nfc_winners, nfc_wc = playoff_teams(nfc_divs)

    def playoff_html(conf_label, div_winners, wildcards, color):
        html = f"""<div style="background:#0d1a2d;border:1px solid #1a2e47;
        border-radius:10px;overflow:hidden;margin-bottom:12px">
          <div style="background:{color};padding:8px 12px;
          font-family:'Barlow Condensed',sans-serif;font-size:.75rem;font-weight:800;
          letter-spacing:2px;color:#fff;text-transform:uppercase">{conf_label}</div>"""
        for i, (team, label) in enumerate(
                [(t, f"Div {i+1}") for i, t in enumerate(div_winners)] +
                [(t, f"WC {i+1}") for i, t in enumerate(wildcards)]):
            abbr = data["n2a"].get(team, "")
            pw   = proj_wins(team)
            seed_col = "#F5A623" if i == 0 else ("#22c55e" if i < 4 else "#7aa8c8")
            bg   = "rgba(245,166,35,.05)" if i == 0 else "transparent"
            sep  = "border-top:1px solid rgba(255,255,255,.06);" if i == 4 else ""
            html += f"""<div style="display:flex;align-items:center;gap:10px;
            padding:8px 12px;background:{bg};{sep}">
              <span style="font-family:'Barlow Condensed',sans-serif;font-size:.75rem;
              font-weight:800;color:{seed_col};min-width:24px">#{i+1}</span>
              <img src="https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"
                   width="26" height="26" style="border-radius:50%;background:#1a2235;padding:2px">
              <span style="flex:1;font-weight:{'700' if i<4 else '400'};
              color:{'#fff' if i<4 else '#b0c4d8'};font-size:.84rem">{team}</span>
              <span style="font-size:.72rem;color:#5a7a9a">{label}</span>
              <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
              color:{'#F5A623' if i==0 else '#e8edf5'}">{pw}W</span>
            </div>"""
        html += "</div>"
        return html

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(playoff_html("AFC Projected Playoffs", afc_winners, afc_wc, "#8B0000"),
                    unsafe_allow_html=True)
    with pc2:
        st.markdown(playoff_html("NFC Projected Playoffs", nfc_winners, nfc_wc, "#013369"),
                    unsafe_allow_html=True)

    # ── Notable Draft Picks & Moves ───────────────────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:.65rem;
    font-weight:700;letter-spacing:3px;color:#5a7a9a;text-transform:uppercase;
    margin:20px 0 12px">2026 Key Draft Picks & Offseason Moves</div>""",
    unsafe_allow_html=True)

    NOTABLE = [
        ('Raiders',   'Fernando Mendoza',   'QB', 'R1 #1', +6.0,
         'First overall pick — franchise QB. Indiana product, 1st Hoosier in R1 since 1994.'),
        ('Packers',   'Micah Parsons',       'EDGE', 'Trade (from DAL)', +4.0,
         'Green Bay traded Kenny Clark + 2027 R1 to Dallas. Elite pass rusher arrives.'),
        ('Seahawks',  'Jadarian Price',      'RB', 'R1 #32', +1.5,
         'Defending Super Bowl champs add depth. Notre Dame RB closes out Round 1.'),
        ('Cowboys',   'Caleb Downs',         'S',  'R1 (trade up)', +2.0,
         'Cowboys traded up for the Ohio State safety. Also added Quinnen Williams + George Pickens. Lost Parsons.'),
        ('Cardinals', 'Jeremiyah Love',      'RB', 'R1 #3', +2.0,
         'Highest-drafted RB since Saquon Barkley. Also grabbed QB Carson Beck in Round 3.'),
        ('Texans',    'Rueben Bain Jr.',     'EDGE', 'R1 #15', +2.5,
         'Miami EDGE rusher adds elite pass rush to an already strong Texans D.'),
        ('Jets',      'Sauce Gardner',       'CB', 'Trade (from IND)', +2.5,
         'Jets acquired All-Pro corner from Colts via 2026+2027 R1 + Adonai Mitchell.'),
        ('Lions',     'Blake Miller',        'OT', 'R1 #17', +2.0,
         'Clemson tackle fills need. Allows Penei Sewell to shift left. Detroit goes OL-heavy.'),
        ('Chiefs',    '(Traded up)',         '—',  'R1', +2.0,
         'Kansas City moved up in Round 1 despite 6-11 season. Retooling quickly.'),
        ('Broncos',   'Omar Cooper Jr.',     'WR', 'R1 #30', +1.5,
         'Indiana WR adds to Denver offensive weapons after their 14-3 season.'),
        ('Titans',    'Carnell Tate',        'WR', 'R1 #4', +2.5,
         'Top 5 pick adds a true WR1 target to help a Titans rebuild from 3-14.'),
        ('Browns',    'Spencer Fano',        'OT', 'R1 + extra pick', +3.5,
         'Cleveland used two first-round picks (own + from JAX trade). Major OL investment.'),
    ]

    NOTABLE.sort(key=lambda x: abs(x[4]), reverse=True)

    cards_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
    for team, player, pos, slot, boost, note in NOTABLE:
        abbr     = data["n2a"].get(team, "")
        sign     = "+" if boost > 0 else ""
        b_color  = "#22c55e" if boost > 0 else "#ef4444"
        pos_clr  = {"QB":"#F5A623","EDGE":"#f87171","OT":"#7bb3f0","RB":"#a3e635",
                     "WR":"#34d399","CB":"#c084fc","S":"#60a5fa","—":"#9ca3af"}.get(pos,"#9ca3af")
        cards_html += f"""
        <div style="background:#0d1a2d;border:1px solid #1a2e47;border-radius:10px;padding:12px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <img src="https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"
                 width="28" height="28" style="border-radius:50%;background:#1a2235;padding:2px">
            <div>
              <div style="font-weight:700;font-size:.84rem;color:#fff">{player}</div>
              <div style="font-size:.65rem;color:#5a7a9a">{team} · {slot}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:.78rem;
              font-weight:800;color:{b_color}">{sign}{boost:.1f}</div>
              <div style="font-size:.6rem;color:#3a5a7a">ELO adj</div>
            </div>
          </div>
          <span style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.08);
          border-radius:4px;padding:1px 6px;font-size:.62rem;font-weight:700;
          color:{pos_clr};letter-spacing:1px">{pos}</span>
          <div style="font-size:.73rem;color:#6b8aaa;margin-top:7px;line-height:1.4">{note}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Footer note ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:20px;padding:12px 16px;
    background:rgba(1,51,105,.1);border:1px solid rgba(1,51,105,.25);
    border-radius:8px;font-size:.72rem;color:#5a7a9a;text-align:center">
      Projections use end-of-2025 ELO ratings from our model + manual adjustments
      for 2026 draft picks and notable trades. 2025 records sourced from the
      official 2026 NFL Draft order (Wikipedia). Full schedule-based predictions
      available once the NFL releases the 2026 schedule in May.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  NFL Game Predictor · 4-Component Blended Model · Data: 1999–2025
</div>
""", unsafe_allow_html=True)
