"""MEWS tables (Subbe et al. 2001, SpO2 replacing AVPU) and missing-vitals handling."""

import pytest

from ecg_transcovnet.mews import (_score_hr, _score_resp_rate, _score_systolic, _score_temp,
                                  analyze_file)


@pytest.mark.parametrize("score, cases", [
    (_score_hr, [(40, 2), (41, 1), (50, 1), (51, 0), (100, 0), (101, 1), (110, 1),
                 (111, 2), (129, 2), (130, 3)]),
    (_score_systolic, [(70, 3), (71, 2), (80, 2), (81, 1), (100, 1), (101, 0), (199, 0),
                       (200, 2)]),
    (_score_resp_rate, [(8, 2), (9, 0), (14, 0), (15, 1), (20, 1), (21, 2), (29, 2), (30, 3)]),
    # degrees F for 34.9, 35.0, 38.4, 38.56 C
    (_score_temp, [(94.82, 2), (95.0, 0), (101.12, 0), (101.4, 2)]),
])
def test_mews_table_edges(score, cases):
    assert [(v, score(v)) for v, _ in cases] == cases


def test_missing_vitals_are_not_scored():
    full = {"HR": 80, "Systolic": 120, "RespRate": 12, "Temp": 98.6, "SpO2": 98}
    partial = {k: v for k, v in full.items() if k != "Temp"}
    summary = analyze_file([{"condition": "VF", "vitals": full},
                            {"condition": "VF", "vitals": partial},
                            {"condition": "VF"}])
    assert summary.mews_scores[0].total_score == 0
    assert summary.mews_scores[1:] == [None, None]
