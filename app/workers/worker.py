"""Stand-alone worker entry point (optional). In the default single-service Render deployment
jobs run in the API process thread pool (app.workers.tasks). This entry runs a smoke job so
operators can verify the worker image: `python -m app.workers.worker`."""
from __future__ import annotations

import json

from ..twin.simulator import FeCrSAFSimulator


def main():
    sim = FeCrSAFSimulator(spinup_h=2)
    s = sim.run(1.0)[-1]
    print(json.dumps({"worker": "ok", "t_h": s["t_h"], "power_mw": s["power_mw"], "status": s["status"]}))


if __name__ == "__main__":
    main()
