"""Tests for head-sized metrics and report helpers on hand-built predictions."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_transcovnet.evaluation import (
    OUTSIDE_HEAD,
    bootstrap_macro_f1,
    build_report,
    compute_metrics,
    fit_logit_bias,
    grouped_metrics,
    macro_f1,
    map_probs_to_head,
    report_markdown,
)

NAMES = ["NORMAL_SINUS", "ATRIAL_FIBRILLATION", "PVC", "LBBB"]


def test_per_class_and_macro_over_present_classes():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    m = compute_metrics(y_true, y_pred, NAMES, subjects=["a", "a", "b", "c", "c", "d"])
    ns, af, pvc, lbbb = (m["per_class"][n] for n in NAMES)
    # NORMAL_SINUS: tp 2, fp 0, fn 1, tn 3
    assert ns["precision"] == pytest.approx(1.0)
    assert ns["recall"] == pytest.approx(2 / 3)
    assert ns["specificity"] == pytest.approx(1.0)
    assert ns["f1"] == pytest.approx(0.8)
    assert ns["subjects"] == 2
    # ATRIAL_FIBRILLATION: tp 1, fp 1, fn 1, tn 3
    assert af["precision"] == pytest.approx(0.5)
    assert af["recall"] == pytest.approx(0.5)
    assert af["specificity"] == pytest.approx(0.75)
    # PVC: tp 1, fp 1, fn 0, tn 4
    assert pvc["f1"] == pytest.approx(2 / 3)
    assert lbbb["support"] == 0
    assert m["absent_classes"] == ["LBBB"]
    assert m["macro"]["f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert m["macro"]["recall"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert np.asarray(m["confusion_matrix"]).shape == (4, 4)
    assert macro_f1(y_true, y_pred, 4) == pytest.approx(m["macro"]["f1"])


def test_outside_head_predictions_count_as_wrong():
    m = compute_metrics([0, 1, 1], [0, OUTSIDE_HEAD, 1], NAMES)
    cm = np.asarray(m["confusion_matrix"])
    assert cm.shape == (4, 5)
    assert cm[1, 4] == 1
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["per_class"]["ATRIAL_FIBRILLATION"]["recall"] == pytest.approx(0.5)
    assert m["predicted_outside_head"] == 1


def test_auroc_perfect_and_random():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], ["PVC", "LBBB"], probs=probs)
    assert m["per_class"]["PVC"]["auroc"] == pytest.approx(1.0)
    tied = np.full((4, 2), 0.5)
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0], ["PVC", "LBBB"], probs=tied)
    assert m["per_class"]["LBBB"]["auroc"] == pytest.approx(0.5)


def test_grouped_metrics_split_by_value():
    g = grouped_metrics([0, 0, 1, 1], [0, 1, 1, 1], NAMES, ["mitbih", "mitbih", "ptbxl", "ptbxl"])
    assert set(g) == {"mitbih", "ptbxl"}
    assert g["ptbxl"]["accuracy"] == pytest.approx(1.0)
    assert g["mitbih"]["n"] == 2


def test_map_probs_to_head_by_name():
    src = ["NORMAL_SINUS", "ST_ELEVATION", "PVC"]
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    dst_probs, pred = map_probs_to_head(probs, src, NAMES)
    assert pred.tolist() == [0, OUTSIDE_HEAD, 2]
    assert dst_probs[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert dst_probs[2, 2] == pytest.approx(0.7)


def test_logit_bias_does_not_reduce_macro_f1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3))
    logits[np.arange(300), y] += 0.8
    logits[:, 0] += 1.0  # systematic over-prediction of class 0
    before = macro_f1(y, logits.argmax(1), 3)
    bias = fit_logit_bias(logits, y)
    assert macro_f1(y, (logits + bias).argmax(1), 3) >= before
    assert bias[0] < 0


def test_bootstrap_interval_is_deterministic_and_bounded():
    y = [0, 1, 0, 1, 0, 1, 2, 2]
    p = [0, 1, 1, 1, 0, 0, 2, 2]
    subjects = ["a", "a", "b", "b", "c", "c", "d", "d"]
    lo, hi = bootstrap_macro_f1(y, p, subjects, 3, n_boot=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == bootstrap_macro_f1(y, p, subjects, 3, n_boot=200, seed=1)


def test_build_report_and_markdown():
    rows = [
        {"subject_id": s, "dataset": d, "label_method": "beat_morphology", "real_lead_mask": m}
        for s, d, m in [("a", "incart", "1111111"), ("b", "incart", "1111111"),
                        ("c", "mitbih", "0100001"), ("d", "mitbih", "0100001")]
    ]
    rep = build_report([0, 1, 2, 2], [0, 1, 2, 0], NAMES, rows, n_boot=50)
    assert set(rep) >= {"overall", "by_dataset", "by_label_method", "by_real_lead_mask"}
    assert rep["low_subject_classes"] == ["NORMAL_SINUS", "ATRIAL_FIBRILLATION", "PVC"]
    md = report_markdown("t", {"crop2000": rep}, NAMES)
    assert "By real-lead mask" in md and "`0100001`" in md


# --- primary metric: validity-flagged classes are excluded, precision-flagged are not ---

def test_primary_macro_f1_excludes_only_validity_flagged():
    from ecg_transcovnet.evaluation import excluded_from_primary, macro_f1, primary_macro_f1

    names = ["A", "B", "VF_LIKE"]
    flags = {
        "VF_LIKE": ["no_seven_real_lead_events"],   # validity -> excluded
        "B": ["few_test_subjects", "test_record_dominates"],  # precision -> kept
    }
    assert excluded_from_primary(names, flags) == ["VF_LIKE"]

    # A perfect, B perfect, VF_LIKE always wrong.
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 0, 0]
    subjects = ["s1", "s2", "s3", "s4", "s5", "s6"]
    out = primary_macro_f1(y_true, y_pred, names, subjects, flags, n_boot=20)
    assert out["n_classes"] == 2
    assert out["excluded_classes"] == ["VF_LIKE"]
    assert out["excluded_reason"]["VF_LIKE"] == ["no_seven_real_lead_events"]
    assert len(out["ci95"]) == 2

    # Excluding a class from the average does NOT hide the damage its misclassifications
    # do elsewhere: the two VF_LIKE events predicted as A are still false positives for A
    # (precision 0.5, recall 1.0 -> F1 0.667), so the primary metric is 0.833, not 1.0.
    assert out["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)
    # ...but it is still above the all-class macro, which also averages in VF_LIKE's 0.0.
    assert out["macro_f1"] > macro_f1(y_true, y_pred, len(names))


def test_primary_macro_f1_without_flags_matches_all_classes():
    from ecg_transcovnet.evaluation import macro_f1, primary_macro_f1

    names = ["A", "B", "C"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    out = primary_macro_f1(y_true, y_pred, names, eval_flags=None)
    assert out["excluded_classes"] == []
    assert out["macro_f1"] == pytest.approx(macro_f1(y_true, y_pred, len(names)))
