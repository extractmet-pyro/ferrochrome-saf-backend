"""Generate the SYNTHETIC per-tap training campaign used by the ML demo (writes app/data/demo/campaign_taps.csv)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.train import generate_campaign  # noqa: E402

days = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
t0 = time.time()
df = generate_campaign(days=days, progress=lambda f: print(f"{100 * f:5.1f} %", flush=True))
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data", "demo", "campaign_taps.csv")
df.to_csv(out, index=False)
print(f"wrote {len(df)} taps to {out} in {time.time() - t0:.0f} s")
