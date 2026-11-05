"""Run synthetic scenarios A-L headless and write a summary (SYNTHETIC / NOT PLANT-VALIDATED).

usage: python scripts/run_demo.py [hours] [A,B,...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.scenario_service import run_scenario  # noqa: E402

hours = float(sys.argv[1]) if len(sys.argv) > 1 else None
ids = sys.argv[2].split(",") if len(sys.argv) > 2 else list("ABCDEFGHIJKL")
out = {}
for sid in ids:
    t = time.time()
    r = run_scenario(sid, hours)
    out[sid] = {"title": r["title"], "hours": r["hours"], "summary": r["summary"], "wall_s": time.time() - t,
                "series": r["series"]}
    s = r["summary"]
    print(f"{sid}: {r['title'][:55]:55s} metal {s['metal_t']:.1f} t  SEC {s['sec_kwh_per_t'] or 0:.0f}  CrRec {s['cr_recovery_pct'] or 0:.1f}  "
          f"|Eres| {s['max_abs_energy_residual_mw']:.1e} MW  ({time.time() - t:.0f} s)", flush=True)
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data", "demo", "scenario_results.json")
with open(path, "w") as fh:
    json.dump(out, fh)
print("wrote", path)
