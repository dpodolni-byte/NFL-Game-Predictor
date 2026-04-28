# 🏈 NFL Game Predictor — Streamlit App

A 4-component blended prediction model: **ELO · 3-Year Rolling Average · Momentum Streak · Team Strength**

---

## 🚀 Deploy to Streamlit Cloud (Free, Permanent URL)

Follow these steps once and your professor (or anyone) can access it forever.

### Step 1 — Create a GitHub Account
If you don't have one, sign up free at [github.com](https://github.com)

### Step 2 — Create a New GitHub Repository
1. Go to [github.com/new](https://github.com/new)
2. Name it: `nfl-predictor` (or anything you like)
3. Set it to **Public**
4. Click **Create repository**

### Step 3 — Upload Files to GitHub
Upload the following 4 files to your repo (drag-and-drop on the GitHub page):

| File | Description |
|------|-------------|
| `app.py` | The Streamlit app code |
| `requirements.txt` | Python dependencies |
| `NFL_Project_Test_5.xlsx` | Your NFL model data |
| `NFLData3.csv` | Team performance stats |

> **GitHub file size limit:** Files under 25MB upload fine. If your Excel file is larger, use [Git LFS](https://git-lfs.com/) or the GitHub Desktop app.

### Step 4 — Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**
3. Click **New app**
4. Select your repository → Branch: `main` → Main file: `app.py`
5. Click **Deploy**

⏳ Takes 2–3 minutes to build the first time.

### Step 5 — Share Your URL
Streamlit gives you a permanent URL like:
```
https://your-username-nfl-predictor-app-xxxxxxx.streamlit.app
```
Share this link with your professor — it works in any browser, no login needed.

---

## 🔄 Updating the App

Any time you push a change to GitHub, Streamlit Cloud auto-redeploys. Just edit `app.py` on GitHub and save.

---

## 📁 Local Development (Optional)

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Place `NFL_Project_Test_5.xlsx` and `NFLData3.csv` in the same folder as `app.py`.

---

## 🧠 Model Components

| Component | Description | Source |
|-----------|-------------|--------|
| **ELO Rating** | Per-game rating updated by margin of victory. Resets to 1500 each season. Close (<7pts): ±10, Medium (8–15pts): ±30, Blowout (>15pts): ±60 | Excel model |
| **3yr Rolling Avg** | Average win % over prior 3 seasons | Excel model |
| **Momentum Streak** | Last-5-games: Hot (3+ wins) = 1.0, Cold (3+ losses) = 0.0, Neutral = 0.5 | Excel model |
| **Team Strength** | Derived from QB Rating, Offensive Yards, Defensive Efficiency, Turnover Rate. Formula: 35% QBR + 30% Offense + 25% Defense + 10% Turnovers. Normalised 0–100. | NFLData3.csv (1999–2019, carry-forward after) |

### Position Weights (for Roster Strength)
| Position | Weight |
|----------|--------|
| QB | 25% |
| WR1 | 8% |
| RB | 7% |
| WR2 | 5% |
| TE | 5% |
| OL (5 players) | 10% total |
| EDGE1/2, CB1/2 | 4–5% each |
| LB, S, DT | 2–4% each |
| K | 1% |
