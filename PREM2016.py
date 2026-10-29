"""
Basic exploratory analysis: Premier League 2015/16
(competition_id=2, season_id=27)

Uses only the match list (scores, dates, teams), so it's ONE download,
not 380 event files. Saves charts to PNG and prints summary tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsbomb_browser import get_matches   # from the earlier script

COMPETITION_ID, SEASON_ID = 2, 27

# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
m = get_matches(COMPETITION_ID, SEASON_ID)
m["date"] = pd.to_datetime(m["date"])
m["total_goals"] = m["home_score"] + m["away_score"]
m["result"] = np.select(
    [m["home_score"] > m["away_score"], m["home_score"] < m["away_score"]],
    ["Home win", "Away win"], default="Draw",
)

print(f"Games: {len(m)} | Teams: {m['home_team'].nunique()} | "
      f"{m['date'].min().date()} to {m['date'].max().date()}\n")

# ----------------------------------------------------------------------------
# 2. HEADLINE NUMBERS
# ----------------------------------------------------------------------------
print("=== Headline numbers ===")
print(f"Total goals:           {m['total_goals'].sum()}")
print(f"Goals per game:        {m['total_goals'].mean():.2f}")
print(f"Home goals per game:   {m['home_score'].mean():.2f}")
print(f"Away goals per game:   {m['away_score'].mean():.2f}")
print("\nResult split (%):")
print((m["result"].value_counts(normalize=True) * 100).round(1).to_string())

# ----------------------------------------------------------------------------
# 3. LEAGUE TABLE (built from results)
# ----------------------------------------------------------------------------
home = m.rename(columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})
away = m.rename(columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})
home["venue"], away["venue"] = "Home", "Away"
long = pd.concat([home[["team", "gf", "ga", "venue", "date"]],
                  away[["team", "gf", "ga", "venue", "date"]]], ignore_index=True)
long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
long["w"] = (long["gf"] > long["ga"]).astype(int)
long["d"] = (long["gf"] == long["ga"]).astype(int)
long["l"] = (long["gf"] < long["ga"]).astype(int)

table = (long.groupby("team")
         .agg(P=("pts", "size"), W=("w", "sum"), D=("d", "sum"), L=("l", "sum"),
              GF=("gf", "sum"), GA=("ga", "sum"), Pts=("pts", "sum"))
         .assign(GD=lambda d: d.GF - d.GA)
         .sort_values(["Pts", "GD", "GF"], ascending=False))
table.insert(0, "Pos", range(1, len(table) + 1))
print("\n=== League table ===")
print(table.to_string())

# ----------------------------------------------------------------------------
# 4. HOME VS AWAY, BEST ATTACKS / DEFENCES
# ----------------------------------------------------------------------------
ha = long.pivot_table(index="team", columns="venue", values="pts", aggfunc="sum")
ha["Home share of pts %"] = (ha["Home"] / (ha["Home"] + ha["Away"]) * 100).round(1)
print("\n=== Points home vs away (top 5 by home reliance) ===")
print(ha.sort_values("Home share of pts %", ascending=False).head(5).to_string())

print("\n=== Top 5 attacks ===")
print(table.sort_values("GF", ascending=False)[["GF"]].head(5).to_string())
print("\n=== Top 5 defences (fewest conceded) ===")
print(table.sort_values("GA")[["GA"]].head(5).to_string())

# ----------------------------------------------------------------------------
# 5. NOTABLE GAMES
# ----------------------------------------------------------------------------
m["margin"] = (m["home_score"] - m["away_score"]).abs()
cols = ["match_id", "date", "home_team", "score", "away_team"]
print("\n=== Highest-scoring games ===")
print(m.sort_values("total_goals", ascending=False)[cols + ["total_goals"]].head(5)
      .to_string(index=False))
print("\n=== Biggest wins ===")
print(m.sort_values("margin", ascending=False)[cols + ["margin"]].head(5)
      .to_string(index=False))

# ----------------------------------------------------------------------------
# 6. CHARTS
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) Points by team
t = table.sort_values("Pts")
axes[0, 0].barh(t.index, t["Pts"], color="#1f78b4")
axes[0, 0].set_title("Final points")
axes[0, 0].set_xlabel("Points")

# (b) Goals scored vs conceded
ax = axes[0, 1]
ax.scatter(table["GF"], table["GA"], s=60, color="#e31a1c")
for team, r in table.iterrows():
    ax.annotate(team, (r["GF"], r["GA"]), fontsize=7, xytext=(3, 3),
                textcoords="offset points")
ax.invert_yaxis()
ax.set_xlabel("Goals scored")
ax.set_ylabel("Goals conceded (axis flipped: up = better)")
ax.set_title("Attack vs defence")

# (c) Distribution of total goals per game
counts = m["total_goals"].value_counts().sort_index()
axes[1, 0].bar(counts.index, counts.values, color="#33a02c")
axes[1, 0].set_title("Total goals per game")
axes[1, 0].set_xlabel("Goals in match")
axes[1, 0].set_ylabel("Number of games")

# (d) Goals per game by month
monthly = m.groupby(m["date"].dt.to_period("M"))["total_goals"].mean()
axes[1, 1].plot(monthly.index.astype(str), monthly.values, marker="o", color="#ff7f00")
axes[1, 1].axhline(m["total_goals"].mean(), ls="--", color="gray", label="Season avg")
axes[1, 1].set_title("Average goals per game by month")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].legend()

fig.suptitle("Premier League 2015/16 - Basic Exploration", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig("pl_2015_16_eda.png", dpi=200, bbox_inches="tight")
print("\nSaved pl_2015_16_eda.png")