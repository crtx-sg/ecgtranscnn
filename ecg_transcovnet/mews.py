"""Modified Early Warning Score (MEWS) with SpO2 replacing AVPU."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median


_MK_ALPHA = 0.05


@dataclass
class MKResult:
    """Result of a Mann-Kendall trend test."""
    trend: str  # "increasing", "decreasing", "no trend"
    p_value: float
    slope: float  # Sen's slope
    s_statistic: int


@dataclass
class MEWSComponent:
    """A single MEWS scoring component."""
    name: str
    value: float
    score: int  # 0-3


@dataclass
class MEWSResult:
    """Complete MEWS assessment."""
    total_score: int
    components: list[MEWSComponent]
    risk_level: str  # "Low", "Medium", "High", "Critical"


@dataclass
class TrendAssessment:
    """Trend analysis for a single vital sign across events."""
    vital_name: str
    direction: str  # "improving", "deteriorating", "stable"
    slope: float
    values: list[float]
    p_value: float | None = None
    significant: bool | None = None


@dataclass
class ClinicalSummary:
    """Aggregated clinical analysis for a file."""
    mews_scores: list[MEWSResult]
    ecg_vital_correlations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MEWS scoring tables
# ---------------------------------------------------------------------------

def _score_hr(hr: float) -> int:
    if hr < 40:
        return 3
    if hr <= 50:
        return 2
    if hr <= 100:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3  # >130


def _score_systolic(sbp: float) -> int:
    if sbp < 70:
        return 3
    if sbp <= 80:
        return 2
    if sbp <= 100:
        return 1
    if sbp <= 200:
        return 0
    return 2  # >200


def _score_resp_rate(rr: float) -> int:
    if rr < 9:
        return 2
    if rr <= 14:
        return 0
    if rr <= 20:
        return 1
    if rr <= 29:
        return 2
    return 3  # >=30


def _score_temp(temp_f: float) -> int:
    """Score temperature. Input in °F, converted to °C internally."""
    temp_c = (temp_f - 32) * 5 / 9
    if temp_c < 35.0:
        return 2
    if temp_c <= 38.4:
        return 0
    if temp_c <= 39.0:
        return 1
    return 2  # >39.0


def _score_spo2(spo2: float) -> int:
    """SpO2 scoring (replaces AVPU): 3=<85%, 2=85-89%, 1=90-93%, 0=94+%."""
    if spo2 < 85:
        return 3
    if spo2 < 90:
        return 2
    if spo2 < 94:
        return 1
    return 0


def _risk_level(total: int) -> str:
    if total <= 2:
        return "Low"
    if total <= 4:
        return "Medium"
    if total <= 6:
        return "High"
    return "Critical"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_mews(
    hr: float,
    systolic: float,
    resp_rate: float,
    temp_f: float,
    spo2: float,
) -> MEWSResult:
    """Calculate MEWS with SpO2 as the 5th component (replacing AVPU)."""
    components = [
        MEWSComponent("Heart Rate", hr, _score_hr(hr)),
        MEWSComponent("Systolic BP", systolic, _score_systolic(systolic)),
        MEWSComponent("Respiratory Rate", resp_rate, _score_resp_rate(resp_rate)),
        MEWSComponent("Temperature", temp_f, _score_temp(temp_f)),
        MEWSComponent("SpO2", spo2, _score_spo2(spo2)),
    ]
    total = sum(c.score for c in components)
    return MEWSResult(total_score=total, components=components, risk_level=_risk_level(total))


_MEWS_VITALS = ("HR", "Systolic", "RespRate", "Temp", "SpO2")


def compute_mews_history(vitals_history: dict) -> list[dict]:
    """Compute MEWS at every aligned timestamp from vitals history.

    Uses forward-fill: at each timestamp the most recent sample for each
    vital is used.  Only timestamps where all 5 MEWS vitals have at least
    one prior sample are scored.

    Args:
        vitals_history: mapping vital name → list of
            ``{"value": float, "timestamp": float}`` sorted by timestamp.

    Returns:
        List of ``{"timestamp": float, "mews": MEWSResult}`` ordered by time.
    """
    # Collect and sort per-vital histories
    sorted_hists: dict[str, list[tuple[float, float]]] = {}
    all_ts: set[float] = set()
    for key in _MEWS_VITALS:
        samples = vitals_history.get(key, [])
        pairs = sorted(((s["timestamp"], s["value"]) for s in samples), key=lambda p: p[0])
        sorted_hists[key] = pairs
        all_ts.update(t for t, _ in pairs)

    if not all_ts:
        return []

    timestamps = sorted(all_ts)

    # Build forward-fill index per vital: position in sorted_hists
    cursors = {key: 0 for key in _MEWS_VITALS}
    results: list[dict] = []

    for ts in timestamps:
        current: dict[str, float] = {}
        skip = False
        for key in _MEWS_VITALS:
            pairs = sorted_hists[key]
            c = cursors[key]
            # Advance cursor to last sample <= ts
            while c < len(pairs) and pairs[c][0] <= ts:
                c += 1
            cursors[key] = c
            if c == 0:
                # No sample yet for this vital
                skip = True
                break
            current[key] = pairs[c - 1][1]

        if skip:
            continue

        mews = calculate_mews(
            hr=current["HR"],
            systolic=current["Systolic"],
            resp_rate=current["RespRate"],
            temp_f=current["Temp"],
            spo2=current["SpO2"],
        )
        results.append({"timestamp": ts, "mews": mews})

    return results


def mann_kendall(values: list[float]) -> MKResult:
    """Mann-Kendall trend test with Sen's slope estimator.

    No external dependencies — uses math.erfc for the p-value.
    Returns 'no trend' if fewer than 3 observations.
    """
    n = len(values)
    if n < 3:
        return MKResult(trend="no trend", p_value=1.0, slope=0.0, s_statistic=0)

    # S statistic
    s = 0
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
            denom = j - i
            slopes.append(diff / denom)

    # Tie correction
    from collections import Counter
    tie_counts = Counter(values)
    tie_sum = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18

    # Z-score and two-tailed p-value
    if var_s == 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p_value = math.erfc(abs(z) / math.sqrt(2))

    # Sen's slope
    sens_slope = median(slopes) if slopes else 0.0

    # Trend direction
    if p_value < _MK_ALPHA:
        trend = "increasing" if s > 0 else "decreasing"
    else:
        trend = "no trend"

    return MKResult(trend=trend, p_value=p_value, slope=sens_slope, s_statistic=s)


def _linreg_slope(x_vals: list[float], y_vals: list[float]) -> float:
    """Simple linear regression slope."""
    n = len(x_vals)
    if n < 2:
        return 0.0
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_vals, y_vals))
    den = sum((xi - x_mean) ** 2 for xi in x_vals)
    return num / den if den > 0 else 0.0


def _classify_direction(key: str, mk: MKResult) -> str:
    """Classify trend direction from Mann-Kendall result."""
    if mk.p_value >= _MK_ALPHA:
        return "stable"
    if key in ("SpO2", "Systolic", "Diastolic"):
        return "deteriorating" if mk.trend == "decreasing" else "improving"
    return "deteriorating" if mk.trend == "increasing" else "improving"


def assess_trends(
    event_vitals: list[dict],
    event_histories: list[dict] | None = None,
) -> list[TrendAssessment]:
    """Compute linear trends for each vital across events.

    Args:
        event_vitals: list of dicts, each with keys HR, SpO2, Systolic, etc.
        event_histories: optional list of dicts mapping vital name to history
            arrays (list of {"value": ..., "timestamp": ...}).  When provided
            and a vital has history data, the trend is computed from the full
            time-series rather than single-value-per-event.
    """
    if len(event_vitals) < 2:
        return []

    vital_keys = ["HR", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp"]
    trends: list[TrendAssessment] = []

    for key in vital_keys:
        # Try rich history path first
        if event_histories is not None:
            all_samples: list[tuple[float, float]] = []
            for hist_dict in event_histories:
                for sample in hist_dict.get(key, []):
                    all_samples.append((sample["timestamp"], sample["value"]))
            if len(all_samples) >= 2:
                all_samples.sort(key=lambda s: s[0])
                y_vals = [s[1] for s in all_samples]
                mk = mann_kendall(y_vals)
                direction = _classify_direction(key, mk)
                trends.append(TrendAssessment(
                    vital_name=key, direction=direction, slope=mk.slope,
                    values=y_vals, p_value=mk.p_value, significant=mk.p_value < _MK_ALPHA,
                ))
                continue

        # Fall back to single-value-per-event
        values = [v.get(key) for v in event_vitals]
        if any(v is None for v in values):
            continue
        vals = [float(v) for v in values]
        mk = mann_kendall(vals)
        direction = _classify_direction(key, mk)
        trends.append(TrendAssessment(
            vital_name=key, direction=direction, slope=mk.slope,
            values=vals, p_value=mk.p_value, significant=mk.p_value < _MK_ALPHA,
        ))

    return trends


def assess_event_trends(vitals_history: dict) -> list[TrendAssessment]:
    """Compute per-vital trends from a single event's history samples.

    Args:
        vitals_history: mapping of vital name to list of
            ``{"value": float, "timestamp": float}`` samples.

    Returns:
        List of TrendAssessment, one per vital with >= 2 history samples.
    """
    vital_keys = ["HR", "SpO2", "Systolic", "Diastolic", "RespRate", "Temp"]
    trends: list[TrendAssessment] = []

    for key in vital_keys:
        samples = vitals_history.get(key, [])
        if len(samples) < 2:
            continue
        sorted_samples = sorted(samples, key=lambda s: s["timestamp"])
        y_vals = [s["value"] for s in sorted_samples]
        mk = mann_kendall(y_vals)
        direction = _classify_direction(key, mk)
        trends.append(TrendAssessment(
            vital_name=key, direction=direction, slope=mk.slope,
            values=y_vals, p_value=mk.p_value, significant=mk.p_value < _MK_ALPHA,
        ))

    return trends


def assess_mews_trend(mews_history: list[dict]) -> MKResult | None:
    """Run Mann-Kendall on MEWS total scores from compute_mews_history output."""
    scores = [h["mews"].total_score for h in mews_history]
    if len(scores) < 3:
        return None
    return mann_kendall([float(s) for s in scores])


def correlate_ecg_vitals(
    condition: str,
    vitals: dict,
    mews: MEWSResult,
) -> list[str]:
    """Generate rule-based clinical notes correlating ECG findings with vitals."""
    notes: list[str] = []
    cond = condition.upper()

    spo2 = vitals.get("SpO2")
    hr = vitals.get("HR")
    systolic = vitals.get("Systolic")

    # Ventricular arrhythmias with hemodynamic compromise
    if cond in ("VENTRICULAR_TACHYCARDIA", "VT") and spo2 is not None and spo2 < 90:
        notes.append("Critical: VT with hypoxemia — immediate intervention required")
    if cond in ("VENTRICULAR_FIBRILLATION", "VF"):
        notes.append("Critical: VF detected — initiate ACLS protocol")

    # Bradycardia with hypotension
    if hr is not None and hr < 50 and systolic is not None and systolic < 90:
        notes.append("Warning: Bradycardia with hypotension — consider atropine/pacing")

    # Tachycardia with desaturation
    if hr is not None and hr > 130 and spo2 is not None and spo2 < 92:
        notes.append("Warning: Tachycardia with desaturation — assess perfusion")

    # AFib with rapid ventricular response
    if cond in ("ATRIAL_FIBRILLATION", "AFIB") and hr is not None and hr > 120:
        notes.append("AFib with rapid ventricular response — consider rate control")

    # High MEWS
    if mews.total_score >= 5:
        notes.append(f"MEWS {mews.total_score} ({mews.risk_level}) — escalate care")

    return notes


def analyze_file(events: list[dict]) -> ClinicalSummary:
    """Orchestrate MEWS calculation, trend analysis, and correlations for a file.

    Args:
        events: list of dicts with keys: condition, vitals (dict with HR, SpO2, etc.)
    """
    mews_scores: list[MEWSResult] = []
    all_correlations: list[str] = []

    for ev in events:
        v = ev.get("vitals", {})
        hr = v.get("HR", 80)
        systolic = v.get("Systolic", 120)
        rr = v.get("RespRate", 16)
        temp = v.get("Temp", 98.6)
        spo2 = v.get("SpO2", 98)

        mews = calculate_mews(hr, systolic, rr, temp, spo2)
        mews_scores.append(mews)

        notes = correlate_ecg_vitals(ev.get("condition", ""), v, mews)
        all_correlations.extend(notes)

    return ClinicalSummary(
        mews_scores=mews_scores,
        ecg_vital_correlations=all_correlations,
    )
