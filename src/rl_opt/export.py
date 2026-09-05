"""Write the canonical C1 trace, figures, and a self-contained HTML stepper."""

from __future__ import annotations

import json
from pathlib import Path

from rl_opt.config import DEFAULT_WEIGHTS, ENGAGEMENT_WEIGHTS, WorldParams
from rl_opt.figures import plot_horizon, plot_score_terms
from rl_opt.trace import SCHEMA, pick_canonical_user, trace_scenario
from rl_opt.world import TrueWorld

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "analysis" / "traces"
FIG_DIR = ROOT / "analysis" / "figures"


def _viewer_html(trace: dict) -> str:
    payload = json.dumps(trace)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>rl_opt trace stepper</title>
<style>
  :root {{ --bg:#070A14; --surf:#12182A; --text:#EEF1F8; --dim:#8494B4; --violet:#7C5CFF; --rose:#FF5470; --line:#2A3350; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font: 15px/1.45 ui-sans-serif, system-ui; background:var(--bg); color:var(--text); }}
  header {{ padding:20px 28px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; }}
  header span {{ color:var(--dim); font-size:13px; }}
  main {{ display:grid; grid-template-columns: 280px 1fr 1fr; min-height: calc(100vh - 64px); }}
  .list, .pane {{ padding:20px; overflow:auto; }}
  .list {{ border-right:1px solid var(--line); }}
  .pane + .pane {{ border-left:1px solid var(--line); }}
  h2 {{ font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); margin:0 0 12px; }}
  .card {{ background:var(--surf); border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:0 0 8px; cursor:pointer; }}
  .card.active {{ outline:1px solid var(--violet); }}
  .card b {{ display:block; }}
  .meta {{ color:var(--dim); font-size:12px; }}
  .bar {{ height:8px; background:#1b2238; border-radius:4px; margin-top:6px; display:flex; overflow:hidden; }}
  .bar i {{ display:block; height:100%; }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align:left; padding:6px 4px; border-bottom:1px solid var(--line); font-size:13px; }}
  th {{ color:var(--dim); font-weight:500; }}
  .pos {{ color:#A48BFF; }} .neg {{ color:var(--rose); }}
  .feed {{ margin-top:16px; }}
</style>
</head>
<body>
<header>
  <div>rl_opt trace · <code id="schema"></code></div>
  <span id="who"></span>
</header>
<main>
  <section class="list">
    <h2>Candidates (day 0)</h2>
    <div id="cards"></div>
  </section>
  <section class="pane" id="def"></section>
  <section class="pane" id="eng"></section>
</main>
<script>
const T = {payload};
const ACTIONS = ["like","reply","dwell","share","hide","report"];
document.getElementById("schema").textContent = T.schema;
document.getElementById("who").textContent =
  "world " + T.world_seed + " · user " + T.user_id + " · s0=" + T.user.satisfaction_before.toFixed(2);

function termsTable(terms) {{
  let rows = ACTIONS.map(a => {{
    const v = terms[a];
    const cls = v < 0 ? "neg" : "pos";
    return `<tr><td>${{a}}</td><td class="${{cls}}">${{v.toFixed(4)}}</td></tr>`;
  }}).join("");
  rows += `<tr><th>total</th><th>${{terms.total.toFixed(4)}}</th></tr>`;
  return `<table>${{rows}}</table>`;
}}

function pane(name, color) {{
  const f = T.feed[name];
  const c = window._sel;
  const terms = c ? c.score_terms[name] : null;
  return `
    <h2>${{name}}  w</h2>
    <div class="meta">mean bait in feed ${{f.mean_bait.toFixed(2)}} · Δs ${{f.delta_s.toFixed(3)}} · s1 ${{f.satisfaction_after.toFixed(3)}}</div>
    <div class="feed meta">slots ${{f.slots.join(", ")}}</div>
    ${{c ? `<h2 style="margin-top:22px">post ${{c.post_id}} · rank ${{c.rank[name]}} ${{c.shown[name] ? "(shown)" : "(cut)"}}</h2>
      <div class="meta">quality ${{c.quality.toFixed(2)}} · bait ${{c.bait.toFixed(2)}} · rel ${{c.relevance.toFixed(2)}}</div>
      ${{termsTable(terms)}}` : `<p class="meta">select a candidate</p>`}}
  `;
}}

function render() {{
  document.getElementById("def").innerHTML = pane("default", "var(--violet)");
  document.getElementById("eng").innerHTML = pane("engagement", "var(--rose)");
}}

const cards = document.getElementById("cards");
T.candidates
  .filter(c => c.shown.default || c.shown.engagement)
  .sort((a,b) => Math.abs(a.rank.default-a.rank.engagement) - Math.abs(b.rank.default-b.rank.engagement))
  .reverse()
  .forEach((c, i) => {{
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<b>post ${{c.post_id}} ${{c.bait>0.5 ? "· bait" : ""}}</b>
      <span class="meta">rank ${{c.rank.default}} vs ${{c.rank.engagement}} · q ${{c.quality.toFixed(2)}}</span>
      <div class="bar"><i style="width:${{Math.min(100, Math.abs(c.score_terms.default.total)*40)}}%;background:var(--violet)"></i></div>
      <div class="bar"><i style="width:${{Math.min(100, Math.abs(c.score_terms.engagement.total)*40)}}%;background:var(--rose)"></i></div>`;
    el.onclick = () => {{
      window._sel = c;
      [...cards.children].forEach(x => x.classList.remove("active"));
      el.classList.add("active");
      render();
    }};
    cards.appendChild(el);
    if (i === 0) el.click();
  }});
</script>
</body>
</html>
"""


def export(world_seed: int = 0, rollout_seed: int = 7, scenario_seed: int = 11) -> dict:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    params = WorldParams()
    world = TrueWorld.generate(params, seed=world_seed)
    pick = pick_canonical_user(world, scenario_seed=scenario_seed)
    trace = trace_scenario(world, pick.user_id, pick.candidate_ids)
    trace["scenario_seed"] = pick.scenario_seed
    trace["rollout_seed"] = rollout_seed

    default = world.rollout(DEFAULT_WEIGHTS, seed=rollout_seed)
    engagement = world.rollout(ENGAGEMENT_WEIGHTS, seed=rollout_seed)
    horizon = {
        "schema": SCHEMA,
        "world_seed": world_seed,
        "rollout_seed": rollout_seed,
        "default": {
            "proxy": default.proxy,
            "value": default.value,
            "daily_return": list(default.daily_return),
            "daily_bait_share": list(default.daily_bait_share),
            "daily_engagement": list(default.daily_engagement),
            "daily_satisfaction": list(default.daily_satisfaction),
        },
        "engagement": {
            "proxy": engagement.proxy,
            "value": engagement.value,
            "daily_return": list(engagement.daily_return),
            "daily_bait_share": list(engagement.daily_bait_share),
            "daily_engagement": list(engagement.daily_engagement),
            "daily_satisfaction": list(engagement.daily_satisfaction),
        },
    }

    (TRACE_DIR / "c1_user.json").write_text(json.dumps(trace, indent=2) + "\n")
    (TRACE_DIR / "c1_horizon.json").write_text(json.dumps(horizon, indent=2) + "\n")
    (TRACE_DIR / "viewer.html").write_text(_viewer_html(trace))
    plot_score_terms(trace, FIG_DIR / "c1_score_terms.png")
    plot_horizon(default, engagement, FIG_DIR / "c1_horizon.png")
    return {"user_id": pick.user_id, "n_candidates": len(pick.candidate_ids), "proxy_lift": engagement.proxy_lift(default)}


def main() -> int:
    info = export()
    print(f"canonical user {info['user_id']}  ({info['n_candidates']} candidates)")
    print(f"  trace    {TRACE_DIR / 'c1_user.json'}")
    print(f"  horizon  {TRACE_DIR / 'c1_horizon.json'}")
    print(f"  viewer   {TRACE_DIR / 'viewer.html'}")
    print(f"  figures  {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
