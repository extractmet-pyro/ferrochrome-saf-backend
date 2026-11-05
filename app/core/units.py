"""Minimal unit registry with dimension checking (SI internal). Explicit t vs kg, kWh vs MJ,
Nm3 (273.15 K, 101.325 kPa) vs actual m3, dry vs wet gas basis are separate units/flags."""
from __future__ import annotations

UNITS = {  # unit: (dimension, factor to SI)
    "kg": ("mass", 1.0), "t": ("mass", 1000.0), "g": ("mass", 1e-3),
    "J": ("energy", 1.0), "kJ": ("energy", 1e3), "MJ": ("energy", 1e6), "GJ": ("energy", 1e9),
    "kWh": ("energy", 3.6e6), "MWh": ("energy", 3.6e9),
    "W": ("power", 1.0), "kW": ("power", 1e3), "MW": ("power", 1e6),
    "K": ("temperature", 1.0), "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "m": ("length", 1.0), "mm": ("length", 1e-3), "Pa": ("pressure", 1.0), "kPa": ("pressure", 1e3),
    "Nm3": ("normal_volume", 1.0), "mol": ("amount", 1.0), "kmol": ("amount", 1e3),
    "ohm": ("resistance", 1.0), "mOhm": ("resistance", 1e-3), "A": ("current", 1.0), "kA": ("current", 1e3),
    "V": ("voltage", 1.0), "wt%": ("fraction_pct", 1.0), "vol%": ("vfraction_pct", 1.0),
}
NM3_PER_KMOL = 22.414


class UnitError(ValueError):
    pass


def convert(value: float, frm: str, to: str) -> float:
    if frm == to:
        return value
    if frm in ("C", "degC") and to == "K":
        return value + 273.15
    if frm == "K" and to in ("C", "degC"):
        return value - 273.15
    if frm not in UNITS or to not in UNITS:
        raise UnitError(f"unknown unit {frm} or {to}")
    d1, f1 = UNITS[frm]
    d2, f2 = UNITS[to]
    if d1 != d2:
        raise UnitError(f"dimension mismatch: {frm} ({d1}) -> {to} ({d2})")
    return value * f1 / f2


def actual_m3_from_nm3(nm3: float, T_K: float, p_kPa: float) -> float:
    return nm3 * (T_K / 273.15) * (101.325 / p_kPa)


def dry_from_wet(wet: dict[str, float]) -> dict[str, float]:
    if "H2O" not in wet:
        raise UnitError("wet-basis composition must state H2O explicitly")
    d = {k: v for k, v in wet.items() if k != "H2O"}
    s = sum(d.values())
    return {k: 100 * v / s for k, v in d.items()}
