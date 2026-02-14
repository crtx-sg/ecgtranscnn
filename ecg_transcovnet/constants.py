"""Shared constants for the ECG-TransCovNet package."""

from __future__ import annotations

from .simulator.conditions import Condition

NUM_CLASSES = len(Condition)  # 16
CONDITION_LIST = list(Condition)
CONDITION_TO_IDX = {c: i for i, c in enumerate(CONDITION_LIST)}
CLASS_NAMES = [c.name for c in CONDITION_LIST]

SIGNAL_LENGTH = 2400  # 12 s * 200 Hz
ALL_LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]

# MIT-BIH-like proportions (approximation of real-world prevalence)
MIT_BIH_PROPORTIONS: dict[Condition, float] = {
    Condition.NORMAL_SINUS: 0.30,
    Condition.SINUS_BRADYCARDIA: 0.05,
    Condition.SINUS_TACHYCARDIA: 0.05,
    Condition.ATRIAL_FIBRILLATION: 0.10,
    Condition.ATRIAL_FLUTTER: 0.03,
    Condition.PAC: 0.05,
    Condition.SVT: 0.03,
    Condition.PVC: 0.10,
    Condition.VENTRICULAR_TACHYCARDIA: 0.05,
    Condition.VENTRICULAR_FIBRILLATION: 0.02,
    Condition.LBBB: 0.05,
    Condition.RBBB: 0.05,
    Condition.AV_BLOCK_1: 0.04,
    Condition.AV_BLOCK_2_TYPE1: 0.03,
    Condition.AV_BLOCK_2_TYPE2: 0.02,
    Condition.ST_ELEVATION: 0.03,
}
