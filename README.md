# 🏈 NFL Game Predictor

4-component blended model with **65.5% historical accuracy** (2005–2025).

> **ELO Rating · 3-Year Rolling Average · Momentum Streak · Team Strength**

---

## 🚀 Deploy to Streamlit Cloud (Free, Permanent URL)

### Step 1 — GitHub
1. Go to [github.com/new](https://github.com/new) → create a **public** repo named `nfl-predictor`
2. Upload these files (drag-and-drop onto the repo page):
   - `app.py`
   - `requirements.txt`
   - `NFL_Project_Test_6.xlsx`
   - `README.md`

### Step 2 — Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app** → select your repo → Main file: `app.py` → **Deploy**
4. Takes ~2 minutes. You get a permanent public URL.

### Step 3 — Share
Send the URL to your professor. No login, no uploads, no installs.

---

## 🏃 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Place `NFL_Project_Test_6.xlsx` in the same folder as `app.py`.

---

## 📊 Model Components

| Component | Weight | Description |
|---|---|---|
| ELO Rating | 71% | Per-game rating updated by margin of victory. Resets to 1500 each season. |
| 3yr Rolling Avg | 10.8% | Average win % over the prior 3 seasons. |
| Momentum Streak | 2.9% | Hot (3+ wins) = 1.0 · Cold (3+ losses) = 0.0 · Neutral = 0.5 |
| Team Strength | 15.3% | QB Rating (35%) + Offensive Yards (30%) + Defensive Efficiency (25%) + Turnover Rate (10%) |
| Home Boost | +6% | Applied to the home team's 3yr avg component. |

**Overall backtest accuracy: 65.5%** (vs 58.5% base, 59.6% with home boost only)
