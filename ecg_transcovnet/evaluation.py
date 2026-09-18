"""Head-sized evaluation metrics, grouped breakdowns and report writing."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

OUTSIDE_HEAD = -1


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """One-vs-rest AUROC via the Mann-Whitney rank statistic (ties averaged)."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    subjects: Sequence[str] | None = None,
    probs: np.ndarray | None = None,
) -> dict:
    """Per-class and macro metrics for a head of ``len(class_names)`` classes.

    ``y_pred`` may contain :data:`OUTSIDE_HEAD` (baseline mode); such
    predictions are wrong for every class.  Macro averages use classes with
    support > 0 only.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_cls = len(class_names)
    subj = np.asarray(subjects) if subjects is not None else None

    per_class: dict[str, dict] = {}
    present: list[str] = []
    for i, name in enumerate(class_names):
        is_true = y_true == i
        is_pred = y_pred == i
        tp = int((is_true & is_pred).sum())
        fp = int((~is_true & is_pred).sum())
        fn = int((is_true & ~is_pred).sum())
        tn = int((~is_true & ~is_pred).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        entry = dict(precision=prec, recall=rec, specificity=spec, f1=f1, support=tp + fn)
        if subj is not None:
            entry["subjects"] = int(len(np.unique(subj[is_true])))
        if probs is not None:
            entry["auroc"] = _auroc(probs[:, i], is_true)
        per_class[name] = entry
        if tp + fn > 0:
            present.append(name)

    macro = {
        k: float(np.mean([per_class[n][k] for n in present])) if present else 0.0
        for k in ("precision", "recall", "specificity", "f1")
    }
    aucs = [per_class[n].get("auroc") for n in present if per_class[n].get("auroc") is not None]
    if probs is not None:
        macro["auroc"] = float(np.mean(aucs)) if aucs else None
    macro["accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    outside = y_pred == OUTSIDE_HEAD
    cm = np.zeros((n_cls, n_cls + (1 if outside.any() else 0)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p if p != OUTSIDE_HEAD else n_cls] += 1

    return {
        "n": int(len(y_true)),
        "accuracy": macro["accuracy"],
        "macro": macro,
        "per_class": per_class,
        "present_classes": present,
        "absent_classes": [n for n in class_names if n not in present],
        "predicted_outside_head": int(outside.sum()),
        "confusion_matrix": cm.tolist(),
    }


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_classes: int,
    include: Sequence[int] | None = None,
) -> float:
    """Macro-F1 over classes with support > 0 (fast path for model selection).

    *include* restricts the average to those class indices, which is how the
    primary metric drops classes ecg_sigma flags as not measuring what
    deployment needs (see ``primary_macro_f1``).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    allowed = None if include is None else set(int(i) for i in include)
    f1s = []
    for i in range(n_classes):
        if allowed is not None and i not in allowed:
            continue
        is_true = y_true == i
        if not is_true.any():
            continue
        is_pred = y_pred == i
        tp = (is_true & is_pred).sum()
        denom = is_true.sum() + is_pred.sum()
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    subjects: Sequence[str],
    n_classes: int,
    n_boot: int = 500,
    seed: int = 0,
    include: Sequence[int] | None = None,
) -> tuple[float, float]:
    """95 % interval of macro-F1 resampling *subjects* with replacement.

    Resampling subjects rather than events is essential for the thin classes:
    ATRIAL_FLUTTER's 48 test events come from 6 subjects, so an event-level
    bootstrap would treat them as 48 independent samples and report an
    interval far too tight.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(subjects):
        groups[s].append(i)
    members = [np.asarray(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        pick = rng.integers(len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick])
        scores.append(macro_f1(y_true[idx], y_pred[idx], n_classes, include))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


# Flags that make a class's score uninterpretable rather than merely noisy.  Only these
# are dropped from the primary metric: precision flags (few subjects, one record
# dominating) mean the estimate is noisy but unbiased, and dropping those would leave
# only the easy classes and flatter the model.
VALIDITY_FLAGS = ("no_seven_real_lead_events",)


def excluded_from_primary(
    class_names: Sequence[str],
    eval_flags: Mapping[str, Sequence[str]] | None,
    reporting: Mapping[str, object] | None = None,
) -> list[str]:
    """Head classes whose score cannot be interpreted, in head order.

    A package shipping a ``reporting`` block (ecgpkg v2.1+) defines this
    contract itself; we honour it and cross-check it against our own
    flag-derived answer so a silent divergence cannot happen.
    """
    flags = eval_flags or {}
    derived = [c for c in class_names
               if any(f in VALIDITY_FLAGS for f in flags.get(c, ()))]
    declared_map = (reporting or {}).get("excluded_from_primary")
    if declared_map is None:
        return derived
    declared = [c for c in class_names if c in declared_map]
    if declared != derived:
        raise ValueError(
            f"package reporting block excludes {declared} from the primary metric but its "
            f"eval_flags imply {derived}; refusing to guess which is right"
        )
    return declared


def primary_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    subjects: Sequence[str] | None = None,
    eval_flags: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 500,
    reporting: Mapping[str, object] | None = None,
) -> dict:
    """The agreed headline metric: macro-F1 over every class but the invalid ones.

    Returns the score, its subject-level CI, and which classes were dropped and
    why, so a report can never show the number without the caveat.
    """
    names = list(class_names)
    dropped = excluded_from_primary(names, eval_flags, reporting)
    include = [i for i, n in enumerate(names) if n not in dropped]
    out = {
        "macro_f1": macro_f1(y_true, y_pred, len(names), include),
        "n_classes": len(include),
        "excluded_classes": dropped,
        "excluded_reason": {c: list((eval_flags or {}).get(c, ())) for c in dropped},
    }
    if subjects is not None:
        out["ci95"] = bootstrap_macro_f1(
            y_true, y_pred, subjects, len(names), n_boot=n_boot, include=include,
        )
    return out


def summary_metrics(y_true, y_pred, class_names, subjects=None) -> dict:
    """Compact metrics for a subgroup."""
    m = compute_metrics(y_true, y_pred, class_names, subjects)
    return {
        "n": m["n"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro"]["f1"],
        "macro_recall": m["macro"]["recall"],
        "per_class": {
            n: {k: v for k, v in m["per_class"][n].items() if k in ("f1", "recall", "support", "subjects")}
            for n in m["present_classes"]
        },
    }


def grouped_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    group_values: Sequence[str],
    subjects: Sequence[str] | None = None,
) -> dict[str, dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values = np.asarray(group_values)
    subj = np.asarray(subjects) if subjects is not None else None
    out = {}
    for v in sorted(set(values.tolist())):
        sel = values == v
        out[v] = summary_metrics(
            y_true[sel], y_pred[sel], class_names, subj[sel] if subj is not None else None,
        )
    return out


# ---------------------------------------------------------------------------
# Head mapping and calibration
# ---------------------------------------------------------------------------

def map_probs_to_head(
    probs: np.ndarray, src_names: Sequence[str], dst_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map softmax outputs of a *src* head onto a *dst* head by class name.

    Returns ``(dst_probs, pred)`` where ``pred`` is the arg-max over the
    source head translated into dst indices, or :data:`OUTSIDE_HEAD` when the
    source arg-max class is not in the destination head.
    """
    src_index = {n: i for i, n in enumerate(src_names)}
    dst_probs = np.zeros((probs.shape[0], len(dst_names)), dtype=probs.dtype)
    for j, name in enumerate(dst_names):
        if name in src_index:
            dst_probs[:, j] = probs[:, src_index[name]]
    lookup = np.array(
        [list(dst_names).index(n) if n in dst_names else OUTSIDE_HEAD for n in src_names]
    )
    return dst_probs, lookup[probs.argmax(axis=1)]


def fit_logit_bias(
    logits: np.ndarray,
    y_true: Sequence[int],
    grid: Iterable[float] = np.linspace(-2.0, 2.0, 21),
    rounds: int = 3,
) -> np.ndarray:
    """Per-class additive logit bias maximising macro-F1 (coordinate ascent)."""
    y_true = np.asarray(y_true)
    n_cls = logits.shape[1]
    bias = np.zeros(n_cls)
    grid = np.asarray(list(grid))
    best = macro_f1(y_true, (logits + bias).argmax(1), n_cls)
    for _ in range(rounds):
        improved = False
        for c in range(n_cls):
            current = bias[c]
            for g in grid:
                bias[c] = g
                score = macro_f1(y_true, (logits + bias).argmax(1), n_cls)
                if score > best + 1e-9:
                    best, current, improved = score, g, True
            bias[c] = current
        if not improved:
            break
    return bias


# ---------------------------------------------------------------------------
# Full split report
# ---------------------------------------------------------------------------

def build_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    probs: np.ndarray | None = None,
    low_subject_threshold: int = 5,
    n_boot: int = 500,
    eval_flags: Mapping[str, Sequence[str]] | None = None,
    reporting: Mapping[str, object] | None = None,
) -> dict:
    """Overall + grouped metrics for evaluated package rows (same order as predictions)."""
    subjects = [r["subject_id"] for r in rows]
    overall = compute_metrics(y_true, y_pred, class_names, subjects, probs)
    overall["macro_f1_ci95"] = bootstrap_macro_f1(
        y_true, y_pred, subjects, len(class_names), n_boot=n_boot,
    )
    report = {
        "overall": overall,
        "primary": primary_macro_f1(
            y_true, y_pred, class_names, subjects, eval_flags, n_boot=n_boot,
            reporting=reporting,
        ),
        "eval_flags": {k: list(v) for k, v in (eval_flags or {}).items()},
        "low_subject_classes": [
            n for n in overall["present_classes"]
            if overall["per_class"][n]["subjects"] < low_subject_threshold
        ],
    }
    for key, column in (("by_dataset", "dataset"), ("by_label_method", "label_method"),
                        ("by_real_lead_mask", "real_lead_mask")):
        report[key] = grouped_metrics(
            y_true, y_pred, class_names, [r[column] for r in rows], subjects,
        )
    # Paced events are absent from train and present only in val/test (two MIT-BIH
    # patients), so they are a subgroup to watch, not a training population.
    paced_col = next((c for c in ("paced_record", "paced_beats") if rows and c in rows[0]), None)
    if paced_col:
        def _paced(r) -> str:
            v = r.get(paced_col)
            flag = bool(v) if isinstance(v, bool) else int(v or 0) > 0
            return "paced" if flag else "unpaced"
        report["by_paced"] = grouped_metrics(
            y_true, y_pred, class_names, [_paced(r) for r in rows], subjects,
        )
        report["paced_column"] = paced_col
    return report


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def report_markdown(title: str, sections: Mapping[str, dict], class_names: Sequence[str]) -> str:
    """Markdown for one or more named reports (e.g. ``crop2000`` and ``full``)."""
    lines = [f"# {title}", ""]
    for section, rep in sections.items():
        ov = rep["overall"]
        lo, hi = ov["macro_f1_ci95"]
        lines += [
            f"## {section}", "",
        ]
        prim = rep.get("primary")
        if prim and prim["excluded_classes"]:
            plo, phi = prim.get("ci95", (None, None))
            ci = f" (subject-bootstrap 95 % CI {plo:.3f}–{phi:.3f})" if plo is not None else ""
            lines += [
                f"**Primary metric — macro-F1 over {prim['n_classes']} classes: "
                f"{prim['macro_f1']:.3f}**{ci}",
                "",
                "Excludes "
                + "; ".join(f"{c} ({', '.join(f)})" for c, f in prim["excluded_reason"].items())
                + ". Scored below, but on fabricated-lead data only — not validated in the "
                  "deployment configuration.",
                "",
            ]
        lines += [
            f"Events {ov['n']} · accuracy **{ov['accuracy']:.3f}** · macro-F1 (all classes) "
            f"**{ov['macro']['f1']:.3f}** "
            f"(subject-bootstrap 95 % CI {lo:.3f}–{hi:.3f}) · macro recall {ov['macro']['recall']:.3f} · "
            f"macro specificity {ov['macro']['specificity']:.3f} · macro AUROC {_fmt(ov['macro'].get('auroc'))}",
            "",
        ]
        if ov["predicted_outside_head"]:
            lines += [f"Predictions outside the package head: {ov['predicted_outside_head']}", ""]
        if ov["absent_classes"]:
            lines += [f"Classes absent from this split: {', '.join(ov['absent_classes'])}", ""]
        if rep["low_subject_classes"]:
            lines += [f"Fewer than 5 subjects (indicative only): {', '.join(rep['low_subject_classes'])}", ""]
        pkg_flags = rep.get("eval_flags") or {}
        has_flags = any(pkg_flags.values())
        header = "| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |"
        rule = "|---|---:|---:|---:|---:|---:|---:|---:|"
        if has_flags:
            header += " Package flags |"
            rule += "---|"
        lines += [header, rule]
        for name in class_names:
            m = ov["per_class"][name]
            flag = " ⚠" if name in rep["low_subject_classes"] else ""
            row = (
                f"| {name}{flag} | {m['precision']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | "
                f"{m['f1']:.3f} | {_fmt(m.get('auroc'))} | {m['support']} | {m.get('subjects', '—')} |"
            )
            if has_flags:
                row += f" {', '.join(pkg_flags.get(name, ())) or '—'} |"
            lines.append(row)
        lines.append("")
        groups = [("by_dataset", "Dataset"), ("by_label_method", "Label method"),
                  ("by_real_lead_mask", "Real-lead mask")]
        if "by_paced" in rep:
            groups.append(("by_paced", "Pacing"))
        for key, label in groups:
            lines += [f"### By {label.lower()}", "",
                      f"| {label} | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |",
                      "|---|---:|---:|---:|---|"]
            for value, g in rep[key].items():
                per = ", ".join(
                    f"{n} {c['f1']:.2f} ({c['support']}/{c.get('subjects', '—')})"
                    for n, c in g["per_class"].items()
                )
                lines.append(f"| `{value}` | {g['n']} | {g['accuracy']:.3f} | {g['macro_f1']:.3f} | {per} |")
            lines.append("")
    return "\n".join(lines)


def save_confusion_png(cm: np.ndarray, class_names: Sequence[str], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    x_names = list(class_names) + (["(outside head)"] if cm.shape[1] > len(class_names) else [])
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    frac = cm / row_sums
    fig, ax = plt.subplots(figsize=(1.0 * len(x_names) + 3, 0.9 * len(class_names) + 2.5))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="fraction of true class")
    ax.set(xticks=range(len(x_names)), yticks=range(len(class_names)),
           xticklabels=x_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_reports(
    out_dir: str | Path,
    stem: str,
    title: str,
    sections: Mapping[str, dict],
    class_names: Sequence[str],
    extra: Mapping | None = None,
) -> dict[str, Path]:
    """Write ``<stem>.json``, ``<stem>.md`` and one confusion PNG per section."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / f"{stem}.json", "md": out / f"{stem}.md"}
    payload = {"title": title, "class_names": list(class_names), "sections": sections, **(extra or {})}
    paths["json"].write_text(json.dumps(payload, indent=2))
    paths["md"].write_text(report_markdown(title, sections, class_names))
    for section, rep in sections.items():
        png = out / f"{stem}_{section}_confusion.png"
        save_confusion_png(np.asarray(rep["overall"]["confusion_matrix"]), class_names, png,
                           f"{title} — {section}")
        paths[f"png_{section}"] = png
    return paths
