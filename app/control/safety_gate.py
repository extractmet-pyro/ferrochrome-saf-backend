"""Safety / authority gate: every recommendation passes here before it is shown for approval.

The gate can only BLOCK or DOWNGRADE a recommendation; it can never send a command.
ADVISORY_ONLY is enforced in code: there is no command endpoint in this application.
"""
from __future__ import annotations

from ..twin.events import LIFECYCLE_STATES

REQUIRED_TAGS = ["EL_P_TOTAL", "EL_I1", "EL_I2", "EL_I3", "EL_R1", "EL_R2", "EL_R3", "GAS_CO", "GAS_FLOW_DRY", "TC_SHELL_SLAG"]


def check(snapshot: dict, tags: dict, est_state: dict | None, rec: dict, cfg, dist: dict) -> dict:
    reasons, warn = [], []
    state = snapshot.get("operating_state")
    if not cfg.advisory_only:
        reasons.append("configuration error: advisory_only must be true in this deployment")
    if not LIFECYCLE_STATES.get(state, {}).get("mpc", False):
        reasons.append(f"operating state '{state}' does not permit supervisory recommendations")
    bad = [t for t in REQUIRED_TAGS if tags.get(t, {}).get("quality") not in ("GOOD",)]
    if bad:
        reasons.append(f"data quality: required tags not GOOD: {bad}")
    if any(v.get("quality") == "SUSPECT_STUCK" for v in tags.values()):
        reasons.append("data quality: gas analyzer suspected stuck")
    if snapshot.get("status") != "ok":
        reasons.append("twin solver in frozen/fallback state")
    if est_state is not None and est_state.get("mode") != "EKF":
        reasons.append("state estimator in fallback mode")
    if dist.get("ood_flag"):
        warn.append("operation flagged out-of-distribution: hybrid ML correction disabled")
    sol = rec.get("solver", {})
    if not sol.get("success", False):
        reasons.append(f"optimiser did not converge: {sol.get('message')}")
    if sol.get("timed_out"):
        reasons.append("optimiser exceeded time budget")
    for m in rec.get("moves", []):
        if m["new"] < m["lo"] - 1e-9 or m["new"] > m["hi"] + 1e-9:
            reasons.append(f"{m['name']} outside bounds")
        if abs(m["new"] - m["current"]) > m["max_move"] + 1e-9:
            reasons.append(f"{m['name']} exceeds move limit")
    ver = rec.get("verification", {})
    for v in ver.get("hard_violations", []):
        reasons.append(f"verification on nonlinear twin violates hard limit: {v}")
    status = "ISSUED_FOR_REVIEW" if not reasons else "BLOCKED_SAFE_FALLBACK"
    return {"status": status, "blocking_reasons": reasons, "warnings": warn, "advisory_only": True,
            "fallback_action": None if not reasons else "hold current operator-approved setpoints (no change)"}
