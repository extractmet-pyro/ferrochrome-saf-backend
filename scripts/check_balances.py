"""CI gate: reaction atom balances + 6 h nominal mass/energy closure."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.domain.reactions import check_all_balanced  # noqa: E402
from app.twin.simulator import FeCrSAFSimulator  # noqa: E402

bad = check_all_balanced()
assert not bad, f"unbalanced reactions: {bad}"
sim = FeCrSAFSimulator(spinup_h=2)
sim.reset_ledger()
s = sim.run(6)[-1]
mb, eb = s["mass_balance"], s["energy_balance"]
worst = max(abs(v["residual_pct_of_input"]) for k, v in mb.items())
print(f"reactions balanced; worst element residual {worst:.2e} % of input; energy residual {eb['residual_MW']:.2e} MW")
assert worst < 1e-8 and abs(eb["residual_MW"]) < 1e-6
