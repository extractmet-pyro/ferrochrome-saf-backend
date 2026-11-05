"""KPI reports and exports (CSV / JSON / PDF)."""
from __future__ import annotations

import csv
import io
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def history_csv(history) -> str:
    rows = list(history)
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def kpi_report(s) -> dict:
    snap = s.snap
    taps = s.sim.taps
    return {"session": s.id, "t_h": snap["t_h"], "synthetic": True, "label": "SYNTHETIC / NOT PLANT-VALIDATED",
            "kpi_24h": snap["kpi_24h"], "metal_wt_pct": snap["fecr_wt_pct"], "slag_cr2o3_equiv_wt_pct": snap["slag_cr2o3_equiv_wt_pct"],
            "power_mw": snap["power_mw"], "energy_balance_residual_mw": snap["energy_balance_residual_mw"],
            "mass_balance_cr_residual_kg_per_h": snap["mass_balance_cr_residual_kg_per_h"],
            "taps": taps[-24:], "n_taps": len(taps), "alarms": snap["alarms"]}


def pdf_report(rep: dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, "FeCr-TWIN KPI report - Extractmet Private Limited")
    y -= 7 * mm
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.7, 0.1, 0.1)
    c.drawString(20 * mm, y, "SYNTHETIC / NOT PLANT-VALIDATED - advisory software, not a control system")
    c.setFillColorRGB(0, 0, 0)
    y -= 8 * mm
    for line in json.dumps({k: v for k, v in rep.items() if k != "taps"}, indent=1, default=str).splitlines()[:70]:
        c.drawString(20 * mm, y, line[:110])
        y -= 4 * mm
        if y < 20 * mm:
            c.showPage()
            y = H - 20 * mm
            c.setFont("Helvetica", 9)
    c.showPage()
    c.save()
    return buf.getvalue()
