"""
NFL Game Predictor — Streamlit App
4-component blended model: ELO + 3yr Avg + Win Streak + Team Strength
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.nfl-header {
    background: linear-gradient(135deg, #013369 0%, #0a1628 100%);
    border-bottom: 3px solid #D50A0A; border-radius: 12px;
    padding: 22px 32px; text-align: center; margin-bottom: 20px;
}
.nfl-header h1 { font-size:1.9rem; font-weight:900; color:#fff; margin:0; }
.nfl-header p  { color:#7a9cc9; font-size:.82rem; margin:4px 0 0; }
.winner-box {
    background: linear-gradient(135deg, #013369, #0d2554);
    border: 2px solid #F5A623; border-radius: 16px;
    padding: 26px; text-align: center; margin-bottom: 14px;
}
.winner-box .wlabel { font-size:.62rem; letter-spacing:3px; color:#F5A623;
    text-transform:uppercase; font-weight:700; }
.winner-box .wname  { font-size:1.9rem; font-weight:900; color:#fff; margin:5px 0 3px; }
.conf-pill {
    display:inline-block; padding:3px 16px; border-radius:999px;
    background:rgba(245,166,35,.15); border:1px solid #F5A623;
    color:#F5A623; font-size:.7rem; font-weight:700;
    letter-spacing:1px; text-transform:uppercase;
}
.stat-box {
    background:#111827; border:1px solid #1e2d45;
    border-radius:12px; padding:14px; height:100%;
}
.stat-box h4 {
    font-size:.6rem; font-weight:700; color:#6b7fa3;
    letter-spacing:2px; text-transform:uppercase; margin:0 0 8px;
}
.stat-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:3px 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:.8rem;
}
.stat-row:last-child { border-bottom:none; }
.stat-lbl { color:#6b7fa3; }
.stat-val  { font-weight:700; color:#e8eaf0; }
.hot  { color:#22c55e !important; }
.cold { color:#ef4444 !important; }
.neut { color:#f59e0b !important; }
.prob-wrap { background:#1a2235; border-radius:999px; height:13px;
    overflow:hidden; border:1px solid #1e2d45; margin:6px 0; }
.prob-fill { height:100%; border-radius:999px;
    background:linear-gradient(90deg, #D50A0A, #013369); }
section[data-testid="stSidebar"] { background:#0d1525 !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────
CSV_TO_NAME = {
    'ARI':'Cardinals','ATL':'Falcons','BAL':'Ravens','BUF':'Bills',
    'CAR':'Panthers','CHI':'Bears','CIN':'Bengals','CLE':'Browns',
    'DAL':'Cowboys','DEN':'Broncos','DET':'Lions','GNB':'Packers',
    'HOU':'Texans','IND':'Colts','JAX':'Jaguars','KAN':'Chiefs',
    'LAC':'Chargers','LAR':'Rams','MIA':'Dolphins','MIN':'Vikings',
    'NOR':'Saints','NWE':'Patriots','NYG':'Giants','NYJ':'Jets',
    'OAK':'Raiders','PHI':'Eagles','PIT':'Steelers','SDG':'Chargers',
    'SEA':'Seahawks','SFO':'SF49ers','STL':'Rams','TAM':'Buccaneers',
    'TEN':'Titans','WAS':'Commanders',
}

# Position weights — offense-heavy, QB most critical
POSITION_WEIGHTS = {
    'QB':0.25, 'WR1':0.08,'WR2':0.05,'RB':0.07,
    'LT':0.025,'LG':0.015,'C':0.03,'RG':0.015,'RT':0.025, 'TE':0.05,
    'EDGE1':0.04,'EDGE2':0.03,'CB1':0.05,'CB2':0.04,
    'SS':0.03,'FS':0.03,'MLB':0.04,'OLB':0.03,'DT1':0.02,'DT2':0.02,
    'K':0.01,
}
BACKUP_WEIGHTS = {'QB_BK':0.0,'WR3':0.0,'RB2':0.0}
POS_DISPLAY = {
    'QB':'QB','WR1':'WR 1','WR2':'WR 2','RB':'RB',
    'LT':'LT','LG':'LG','C':'C','RG':'RG','RT':'RT','TE':'TE',
    'EDGE1':'EDGE 1','EDGE2':'EDGE 2','CB1':'CB 1','CB2':'CB 2',
    'SS':'SS','FS':'FS','MLB':'MLB','OLB':'OLB',
    'DT1':'DT 1','DT2':'DT 2','K':'K',
    'QB_BK':'QB BK','WR3':'WR 3','RB2':'RB 2',
}
OFF_POS  = {'QB','WR1','WR2','RB','LT','LG','C','RG','RT','TE'}
DEF_POS  = {'EDGE1','EDGE2','CB1','CB2','SS','FS','MLB','OLB','DT1','DT2'}
BACKUP_STARTERS = {'QB':'QB_BK','WR1':'WR3','RB':'RB2'}

def logo_url(abbr): return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"
def rating_color(r):
    if r>=90: return "#22c55e"
    if r>=80: return "#84cc16"
    if r>=70: return "#f59e0b"
    if r>=60: return "#ef4444"
    return "#6b7fa3"

# ─── Data loading ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📊 Loading NFL data…")
def load_all_data(excel_path, csv_path):
    df_raw = pd.read_excel(excel_path, sheet_name='Raw_Games')
    df_avg = pd.read_excel(excel_path, sheet_name='Rolling_3Yr_Avg')
    df_elo = pd.read_excel(excel_path, sheet_name='ELO_Ratings', header=2)
    df_str = pd.read_excel(excel_path, sheet_name='Win_Streak',  header=2)
    df_cfg = pd.read_excel(excel_path, sheet_name='Config',      header=None)

    def get_cfg(label, default):
        for _, row in df_cfg.iterrows():
            if str(row.iloc[0]).strip() == label:
                try: return float(row.iloc[1])
                except: pass
        return default

    default_cfg = {
        'w_elo':0.25,'w_3yr':0.25,'w_streak':0.25,'w_strength':0.25,
        'home_boost': get_cfg('Home Advantage Boost', 0.085),
    }
    NAME_TO_ABBR = dict(zip(df_raw['Away Team'], df_raw['Away Abbr']))
    NAME_TO_ABBR.update(dict(zip(df_raw['Home Team'], df_raw['Home Abbr'])))

    avg_lkp = {(int(r['Season']), r['Team']): r['AvgWinPct']
               for _, r in df_avg.iterrows() if pd.notna(r.get('AvgWinPct'))}
    elo_lkp = {}
    for _, r in df_elo.iterrows():
        s = int(r['Season'])
        elo_lkp[(s, r['Away Team'])] = float(r['Post Away ELO'])
        elo_lkp[(s, r['Home Team'])] = float(r['Post Home ELO'])
    streak_lkp = {}
    for _, r in df_str.iterrows():
        s = int(r['Season'])
        streak_lkp[(s, r['Away Team'])] = (r['Away Streak'], float(r['Away Streak Score']))
        streak_lkp[(s, r['Home Team'])] = (r['Home Streak'], float(r['Home Streak Score']))

    # ── Team Strength from CSV performance data ────────────────────────────
    df_perf = pd.read_csv(csv_path, low_memory=False)
    df_perf = df_perf[df_perf['Year'] >= 1999].copy()

    def parse_res(r):
        s = str(r)
        if s.startswith('W'): return 1.0
        if s.startswith('L'): return 0.0
        return 0.5

    df_perf['_win'] = df_perf['Result'].apply(parse_res)
    for c in ['Rate','Yds','Yds.2','DY/P','TO']:
        df_perf[c] = pd.to_numeric(df_perf[c], errors='coerce')

    agg = df_perf.groupby(['Tm','Year']).agg(
        AvgQBR=('Rate','mean'), AvgPassYds=('Yds','mean'),
        AvgRushYds=('Yds.2','mean'), AvgDYP=('DY/P','mean'), AvgTO=('TO','mean'),
    ).reset_index().dropna()

    def norm(s, invert=False):
        mn,mx = s.min(), s.max()
        v = (s-mn)/(mx-mn)*100
        return 100-v if invert else v

    agg['qbr_s'] = norm(agg['AvgQBR'])
    agg['off_s'] = norm(agg['AvgPassYds'] + agg['AvgRushYds'])
    agg['def_s'] = norm(agg['AvgDYP'], invert=True)
    agg['to_s']  = norm(agg['AvgTO'],  invert=True)
    agg['TS_raw']= 0.35*agg['qbr_s'] + 0.30*agg['off_s'] + 0.25*agg['def_s'] + 0.10*agg['to_s']
    mn,mx = agg['TS_raw'].min(), agg['TS_raw'].max()
    agg['TeamStrength'] = (agg['TS_raw']-mn)/(mx-mn)*100
    agg['TeamName'] = agg['Tm'].map(CSV_TO_NAME)
    agg = agg.dropna(subset=['TeamName'])

    strength_lkp, profile_lkp = {}, {}
    for _, r in agg.iterrows():
        k = (int(r['Year']), r['TeamName'])
        strength_lkp[k] = r['TeamStrength']
        profile_lkp[k]  = {'qbr_s':r['qbr_s'],'off_s':r['off_s'],'def_s':r['def_s'],'to_s':r['to_s']}

    # Carry forward for 2020-2025 (no CSV data)
    all_teams = sorted(set(df_raw['Away Team']) | set(df_raw['Home Team']))
    for team in all_teams:
        for season in range(2020, 2026):
            if (season, team) not in strength_lkp:
                for y in range(season-1, 1998, -1):
                    if (y, team) in strength_lkp:
                        strength_lkp[(season,team)] = strength_lkp[(y,team)]
                        profile_lkp[(season,team)]  = profile_lkp[(y,team)]
                        break

    return {
        'df_raw': df_raw, 'avg_lkp': avg_lkp,
        'elo_lkp': elo_lkp, 'streak_lkp': streak_lkp,
        'strength_lkp': strength_lkp, 'profile_lkp': profile_lkp,
        'NAME_TO_ABBR': NAME_TO_ABBR, 'default_cfg': default_cfg,
        'teams': all_teams,
        'seasons': sorted(df_raw['Season'].unique().tolist(), reverse=True),
    }

# ─── Roster generation ───────────────────────────────────────────────────────
def generate_roster(team, season, data):
    k  = (int(season), team)
    ts = data['strength_lkp'].get(k, 50.0)
    pr = data['profile_lkp'].get(k, {'qbr_s':50,'off_s':50,'def_s':50,'to_s':50})
    qbr_s,off_s,def_s = pr['qbr_s'], pr['off_s'], pr['def_s']
    base = 58 + (ts/100)*30

    def r(v, boost=0): return int(max(40, min(99, round(v+boost))))
    pos_ratings = {
        'QB':    r(base+(qbr_s-50)*0.30+5), 'WR1':  r(base+(off_s-50)*0.15+2),
        'WR2':   r(base+(off_s-50)*0.10-4), 'RB':   r(base+(off_s-50)*0.10),
        'LT':    r(base-1),  'LG':    r(base-4), 'C':r(base-2),
        'RG':    r(base-4),  'RT':    r(base-2), 'TE':r(base+(off_s-50)*0.08-2),
        'EDGE1': r(base+(def_s-50)*0.20+3), 'EDGE2':r(base+(def_s-50)*0.15-3),
        'CB1':   r(base+(def_s-50)*0.15+1), 'CB2':  r(base+(def_s-50)*0.10-4),
        'SS':    r(base+(def_s-50)*0.10-2), 'FS':   r(base+(def_s-50)*0.10-2),
        'MLB':   r(base+(def_s-50)*0.15),   'OLB':  r(base+(def_s-50)*0.10-5),
        'DT1':   r(base+(def_s-50)*0.12),   'DT2':  r(base+(def_s-50)*0.08-5),
        'K':     r(base-5),
        'QB_BK': r(base+(qbr_s-50)*0.30-15),
        'WR3':   r(base+(off_s-50)*0.08-10),
        'RB2':   r(base+(off_s-50)*0.08-8),
    }
    all_w = {**POSITION_WEIGHTS, **BACKUP_WEIGHTS}
    return {pos: {'label':POS_DISPLAY.get(pos,pos), 'rating':rat, 'weight':all_w.get(pos,0.0)}
            for pos, rat in pos_ratings.items()}

def calc_str_pct(roster, disabled):
    full = sum(v['rating']*v['weight'] for v in roster.values() if v['weight']>0)
    if full==0: return 1.0
    eff = 0.0
    for pos, v in roster.items():
        if v['weight']==0: continue
        if pos in disabled:
            bk = BACKUP_STARTERS.get(pos)
            if bk and bk not in disabled:
                eff += roster[bk]['rating'] * v['weight']
        else:
            eff += v['rating'] * v['weight']
    return eff/full

# ─── Prediction ──────────────────────────────────────────────────────────────
def predict(season, away, home, data, cfg, away_dis, home_dis):
    s = int(season)
    we,w3,ws,wt,boost = cfg['w_elo'],cfg['w_3yr'],cfg['w_streak'],cfg['w_strength'],cfg['home_boost']

    # 3yr avg
    a_avg = data['avg_lkp'].get((s,away))
    h_avg = data['avg_lkp'].get((s,home))
    note  = None
    if a_avg is None or h_avg is None:
        for dy in range(1,8):
            if a_avg is None: a_avg = data['avg_lkp'].get((s+dy,away))
            if h_avg is None: h_avg = data['avg_lkp'].get((s+dy,home))
        a_avg = a_avg or 0.5; h_avg = h_avg or 0.5
        note = "3yr avg not available for this season — using nearest available data."

    # ELO
    a_elo = data['elo_lkp'].get((s,away),1500.0)
    h_elo = data['elo_lkp'].get((s,home),1500.0)
    etot  = a_elo+h_elo
    a_en  = a_elo/etot; h_en = h_elo/etot

    # Streak
    a_sl,a_ss = data['streak_lkp'].get((s,away),('Neutral',0.5))
    h_sl,h_ss = data['streak_lkp'].get((s,home),('Neutral',0.5))

    # Team strength
    a_r = generate_roster(away,s,data); h_r = generate_roster(home,s,data)
    a_ts_full = data['strength_lkp'].get((s,away),50.0)
    h_ts_full = data['strength_lkp'].get((s,home),50.0)
    a_pct = calc_str_pct(a_r,away_dis); h_pct = calc_str_pct(h_r,home_dis)
    a_ts_eff = (a_ts_full/100)*a_pct; h_ts_eff = (h_ts_full/100)*h_pct
    st_sum = a_ts_eff+h_ts_eff
    a_tn = a_ts_eff/st_sum if st_sum>0 else 0.5
    h_tn = h_ts_eff/st_sum if st_sum>0 else 0.5

    # Blended scores
    a_sc = we*a_en + w3*a_avg               + ws*a_ss + wt*a_tn
    h_sc = we*h_en + w3*(h_avg+boost)        + ws*h_ss + wt*h_tn
    winner = home if h_sc>=a_sc else away
    gap    = h_sc-a_sc
    h_prob = 1/(1+np.exp(-gap*22)); a_prob = 1-h_prob
    mg     = abs(gap)
    conf   = ('Toss-Up' if mg<0.015 else 'Slight Edge' if mg<0.04
              else 'Moderate Confidence' if mg<0.08 else 'Strong Favourite')

    # H2H
    df = data['df_raw']
    h2h = df[(df['Season']==s) & (
        ((df['Away Team']==away)&(df['Home Team']==home)) |
        ((df['Away Team']==home)&(df['Home Team']==away))
    )]
    h2h_str = None
    if len(h2h):
        g = h2h.iloc[0]
        h2h_str = (f"Wk {int(g['Week'])}: {g['Away Team']} {int(g['Away Score'])}–"
                   f"{int(g['Home Score'])} {g['Home Team']} → {g['Actual Winner']} won")

    return dict(
        winner=winner, loser=(away if winner==home else home), conf=conf,
        a_sc=round(a_sc,4), h_sc=round(h_sc,4),
        a_prob=round(a_prob*100,1), h_prob=round(h_prob*100,1),
        a_avg=a_avg, h_avg=h_avg, boost=boost,
        a_elo=int(a_elo), h_elo=int(h_elo), a_en=round(a_en,4), h_en=round(h_en,4),
        a_sl=a_sl, h_sl=h_sl, a_ss=a_ss, h_ss=h_ss,
        a_ts=round(a_ts_full,1), h_ts=round(h_ts_full,1),
        a_pct=round(a_pct*100,1), h_pct=round(h_pct*100,1),
        a_tn=round(a_tn,4), h_tn=round(h_tn,4),
        a_r=a_r, h_r=h_r, h2h=h2h_str, note=note,
        away=away, home=home, season=s,
        cfg={'elo':we,'3yr':w3,'streak':ws,'strength':wt},
    )

def comp_bar(label, av, hv, al, hl, fmt='.3f'):
    aw = av/(av+hv)*100 if (av+hv)>0 else 50
    ac = '#22c55e' if av>=hv else '#6b7fa3'
    hc = '#22c55e' if hv>=av else '#6b7fa3'
    st.markdown(f"""
    <div style="margin-bottom:9px">
      <div style="font-size:.58rem;font-weight:700;color:#6b7fa3;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:3px">{label}</div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:100px;text-align:right;font-size:.78rem;font-weight:700;color:{ac}">
          {al}: {format(av,fmt)}</div>
        <div style="flex:1;background:#1a2235;border-radius:4px;height:7px;overflow:hidden">
          <div style="height:100%;width:{aw:.1f}%;
            background:linear-gradient(90deg,#D50A0A,#013369);border-radius:4px"></div>
        </div>
        <div style="width:100px;font-size:.78rem;font-weight:700;color:{hc}">
          {hl}: {format(hv,fmt)}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def roster_grid(side, team, roster, key):
    disabled = st.session_state.get(key, set())
    groups = [
        ('Offense', [p for p in OFF_POS if p in roster]),
        ('Backups', ['QB_BK','WR3','RB2']),
        ('Defense', [p for p in DEF_POS if p in roster]),
        ('Special', ['K']),
    ]
    changed = False
    for grp, positions in groups:
        st.markdown(f'<div style="font-size:.58rem;font-weight:700;color:#F5A623;'
                    f'letter-spacing:2px;text-transform:uppercase;margin:6px 0 3px">{grp}</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for i, pos in enumerate(positions):
            if pos not in roster: continue
            p     = roster[pos]
            avail = pos not in disabled
            rc    = rating_color(p['rating'])
            w_lbl = f"Wt: {p['weight']*100:.1f}%" if p['weight']>0 else "Backup"
            with cols[i%4]:
                checked = st.checkbox(
                    f"{p['label']}",
                    value=avail, key=f"{key}_{pos}",
                    help=f"Rating: {p['rating']} | {w_lbl}"
                )
                st.markdown(f'<div style="text-align:center;margin-top:-8px;'
                            f'font-size:.75rem;font-weight:800;color:{rc}">{p["rating"]}</div>',
                            unsafe_allow_html=True)
                if checked and pos in disabled:
                    disabled.discard(pos); changed=True
                elif not checked and pos not in disabled:
                    disabled.add(pos); changed=True
    if changed:
        st.session_state[key] = disabled
    return disabled

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""<div class="nfl-header">
      <h1>🏈 NFL Game Predictor</h1>
      <p>Blended model: ELO Rating · 3-Year Rolling Average · Momentum Streak · Team Strength</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Data")
        EXCEL = "NFL_Project_Test_5.xlsx"
        CSV   = "NFLData3.csv"
        excel_path = EXCEL if os.path.exists(EXCEL) else None
        csv_path   = CSV   if os.path.exists(CSV)   else None

        if not excel_path:
            up = st.file_uploader("NFL Excel file (.xlsx)", type=['xlsx'])
            if up:
                with open(EXCEL,"wb") as f: f.write(up.read())
                excel_path = EXCEL; st.rerun()
        else:
            st.success(f"✅ {EXCEL}")
        if not csv_path:
            up2 = st.file_uploader("Performance CSV (.csv)", type=['csv'])
            if up2:
                with open(CSV,"wb") as f: f.write(up2.read())
                csv_path = CSV; st.rerun()
        else:
            st.success(f"✅ {CSV}")

        if not excel_path or not csv_path:
            st.warning("Upload both files to continue.")
            st.stop()

        st.markdown("---")
        st.markdown("### ⚙️ Model Weights")
        st.caption("Adjust sliders · must sum to 100")
        w_elo  = st.slider("ELO Rating",      0,100,25,5)
        w_3yr  = st.slider("3yr Rolling Avg", 0,100,25,5)
        w_str  = st.slider("Win Streak",      0,100,25,5)
        w_ts   = st.slider("Team Strength",   0,100,25,5)
        total  = w_elo+w_3yr+w_str+w_ts
        col_dot = "🟢" if total==100 else "🔴"
        st.markdown(f"{col_dot} **{total}%** {'✓ Ready' if total==100 else '← must = 100'}")
        boost = st.slider("🏠 Home Boost %", 0, 20, 9, 1) / 100

        st.markdown("---")
        st.markdown("### 📈 Accuracy Benchmarks")
        st.markdown("""
| Model | Accuracy |
|---|---|
| Base 3yr avg | 58.5% |
| + Home boost | 59.6% |
| **4-component** | **~61.7%+** |
""")
        st.markdown("---")
        st.caption("Team Strength data: 1999–2019 (CSV). "
                   "2020–2025 carry-forward last known season.")

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        data = load_all_data(excel_path, csv_path)
    except Exception as e:
        st.error(f"Data load error: {e}"); st.stop()

    cfg = {'w_elo':w_elo/100,'w_3yr':w_3yr/100,'w_streak':w_str/100,
           'w_strength':w_ts/100,'home_boost':boost}

    # ── Team + Season Picker ────────────────────────────────────────────────
    c1, cv, c2 = st.columns([5,1,5])
    with c1:
        away = st.selectbox("🛫 Away Team", data['teams'],
                            index=data['teams'].index('Eagles') if 'Eagles' in data['teams'] else 0,
                            key='sel_away')
        a_abbr = data['NAME_TO_ABBR'].get(away,'')
        st.markdown(f"""<div style="text-align:center;padding:8px 0">
          <img src="{logo_url(a_abbr)}" width="80"
               style="border-radius:50%;border:2px solid #F5A623;padding:4px;background:#1a2235">
          <div style="font-weight:800;font-size:1rem;margin-top:5px">{away}</div>
        </div>""", unsafe_allow_html=True)
    with cv:
        st.markdown('<div style="text-align:center;padding-top:70px;'
                    'font-size:1.5rem;font-weight:900;color:#6b7fa3">@</div>',
                    unsafe_allow_html=True)
    with c2:
        home = st.selectbox("🏠 Home Team", data['teams'],
                            index=data['teams'].index('Chiefs') if 'Chiefs' in data['teams'] else 1,
                            key='sel_home')
        h_abbr = data['NAME_TO_ABBR'].get(home,'')
        st.markdown(f"""<div style="text-align:center;padding:8px 0">
          <img src="{logo_url(h_abbr)}" width="80"
               style="border-radius:50%;border:2px solid #F5A623;padding:4px;background:#1a2235">
          <div style="font-weight:800;font-size:1rem;margin-top:5px">{home}</div>
        </div>""", unsafe_allow_html=True)

    sa, sb = st.columns([2,5])
    with sa:
        season = st.selectbox("Season", data['seasons'], key='sel_season')
    with sb:
        st.write(""); st.write("")
        predict_btn = st.button("⚡  Predict Winner", use_container_width=True,
                                type="primary", disabled=(away==home or total!=100))

    if away==home:
        st.warning("Select two different teams."); return
    if total!=100:
        st.info(f"Sidebar weights = {total}% — adjust to 100% to enable prediction.")

    # ── Roster Toggles ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏥 Roster & Injury Simulation")
    st.caption("Uncheck a player to simulate them being out — prediction recalculates live")

    if 'away_dis' not in st.session_state: st.session_state['away_dis'] = set()
    if 'home_dis' not in st.session_state: st.session_state['home_dis'] = set()

    a_r = generate_roster(away, int(season), data)
    h_r = generate_roster(home, int(season), data)

    t1, t2 = st.tabs([f"🛫 {away}", f"🏠 {home}"])
    with t1:
        a_dis = roster_grid('away', away, a_r, 'away_dis')
        a_hp  = calc_str_pct(a_r, a_dis)
        bar_color = '#ef4444' if a_hp<0.9 else '#22c55e'
        st.markdown(f'<div style="font-size:.8rem;font-weight:700;color:{bar_color};margin-top:4px">'
                    f'Roster Health: {a_hp*100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(a_hp)
    with t2:
        h_dis = roster_grid('home', home, h_r, 'home_dis')
        h_hp  = calc_str_pct(h_r, h_dis)
        bar_color2 = '#ef4444' if h_hp<0.9 else '#22c55e'
        st.markdown(f'<div style="font-size:.8rem;font-weight:700;color:{bar_color2};margin-top:4px">'
                    f'Roster Health: {h_hp*100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(h_hp)

    # ── Prediction ────────────────────────────────────────────────────────
    if total == 100 and away != home:
        res = predict(season, away, home, data, cfg,
                      st.session_state.get('away_dis',set()),
                      st.session_state.get('home_dis',set()))

        st.markdown("---")
        st.markdown("### 🏆 Prediction")

        # Winner banner
        w_abbr = data['NAME_TO_ABBR'].get(res['winner'],'')
        st.markdown(f"""<div class="winner-box">
          <div class="wlabel">Predicted Winner</div>
          <div style="margin:10px 0">
            <img src="{logo_url(w_abbr)}" width="96"
                 style="border-radius:50%;border:3px solid #F5A623;
                        padding:5px;background:#1a2235">
          </div>
          <div class="wname">{res['winner']}</div>
          <span class="conf-pill">{res['conf']}</span>
        </div>""", unsafe_allow_html=True)

        # Probability
        st.markdown(f"""
        <div style="margin:10px 0 16px">
          <div style="display:flex;justify-content:space-between;
                      font-size:.82rem;font-weight:700;margin-bottom:4px">
            <span style="color:{'#22c55e' if res['a_prob']>res['h_prob'] else '#6b7fa3'}">
              {away} — {res['a_prob']}%</span>
            <span style="color:{'#22c55e' if res['h_prob']>res['a_prob'] else '#6b7fa3'}">
              {home} — {res['h_prob']}%</span>
          </div>
          <div class="prob-wrap">
            <div class="prob-fill" style="width:{res['a_prob']}%"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        if res['h2h']:  st.info(f"📋 Historical match-up: {res['h2h']}")
        if res['note']: st.caption(f"ℹ️ {res['note']}")

        # Component breakdown bars
        st.markdown("#### 📊 Score Component Breakdown")
        st.caption(f"Weights — ELO: {res['cfg']['elo']*100:.0f}%  ·  "
                   f"3yr Avg: {res['cfg']['3yr']*100:.0f}%  ·  "
                   f"Streak: {res['cfg']['streak']*100:.0f}%  ·  "
                   f"Team Strength: {res['cfg']['strength']*100:.0f}%")

        comp_bar("ELO (normalised)", res['a_en'], res['h_en'], away, home, '.4f')
        comp_bar("3yr Rolling Avg Win %", res['a_avg'], res['h_avg'], away, home, '.3f')
        comp_bar("Momentum Streak Score", res['a_ss'], res['h_ss'], away, home, '.1f')
        comp_bar("Team Strength (normalised)", res['a_tn'], res['h_tn'], away, home, '.4f')

        # Stat detail cards
        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        def sc(label): return f'<div class="stat-lbl">{label}</div>'
        def sv(val, extra=""): return f'<div class="stat-val">{val}{extra}</div>'

        with sc1:
            st.markdown(f"""<div class="stat-box"><h4>3yr Rolling Avg</h4>
            <div class="stat-row">{sc(away)}{sv(f'{res["a_avg"]*100:.1f}%')}</div>
            <div class="stat-row">{sc(home)}{sv(f'{res["h_avg"]*100:.1f}%',
              f' <span style="color:#fbbf24;font-size:.68rem">(+{res["boost"]*100:.1f}%)</span>')}</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="stat-box"><h4>End-of-Season ELO</h4>
            <div class="stat-row">{sc(away)}{sv(f'{res["a_elo"]:,}')}</div>
            <div class="stat-row">{sc(home)}{sv(f'{res["h_elo"]:,}')}</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            ac = {'Hot':'hot','Cold':'cold'}.get(res['a_sl'],'neut')
            hc = {'Hot':'hot','Cold':'cold'}.get(res['h_sl'],'neut')
            st.markdown(f"""<div class="stat-box"><h4>Momentum Streak</h4>
            <div class="stat-row">{sc(away)}<span class="stat-val {ac}">{res["a_sl"]}</span></div>
            <div class="stat-row">{sc(home)}<span class="stat-val {hc}">{res["h_sl"]}</span></div>
            </div>""", unsafe_allow_html=True)
        with sc4:
            ai = (f' <span style="color:#ef4444;font-size:.68rem">({res["a_pct"]:.0f}%✓)</span>'
                  if res['a_pct']<100 else '')
            hi = (f' <span style="color:#ef4444;font-size:.68rem">({res["h_pct"]:.0f}%✓)</span>'
                  if res['h_pct']<100 else '')
            st.markdown(f"""<div class="stat-box"><h4>Team Strength (CSV-derived)</h4>
            <div class="stat-row">{sc(away)}{sv(f'{res["a_ts"]:.1f}/100',ai)}</div>
            <div class="stat-row">{sc(home)}{sv(f'{res["h_ts"]:.1f}/100',hi)}</div>
            </div>""", unsafe_allow_html=True)

        # Final blended scores
        ac = '#22c55e' if res['a_sc']>res['h_sc'] else '#6b7fa3'
        hc = '#22c55e' if res['h_sc']>res['a_sc'] else '#6b7fa3'
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;
                    padding:14px 24px;margin-top:6px;
                    display:flex;justify-content:space-around;text-align:center">
          <div>
            <div style="font-size:.6rem;color:#6b7fa3;letter-spacing:2px;
                        font-weight:700;text-transform:uppercase">Blended Score</div>
            <div style="font-size:1.25rem;font-weight:900;color:{ac};margin-top:4px">
              {away}: {res['a_sc']:.4f}</div>
            <div style="font-size:1.25rem;font-weight:900;color:{hc}">
              {home}: {res['h_sc']:.4f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
