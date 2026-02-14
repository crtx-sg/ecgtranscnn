"""ECG Simulator — synthetic MIT-BIH-like ECG data generation."""

from .conditions import Condition, ConditionConfig, CONDITION_REGISTRY
from .morphology import PatientParams, generate_patient_params
from .noise import NoiseConfig, NOISE_PRESETS
from .ecg_simulator import ECGSimulator, SimulatedEvent
from .hdf5_writer import HDF5EventWriter

__all__ = [
    "Condition",
    "ConditionConfig",
    "CONDITION_REGISTRY",
    "PatientParams",
    "generate_patient_params",
    "NoiseConfig",
    "NOISE_PRESETS",
    "ECGSimulator",
    "SimulatedEvent",
    "HDF5EventWriter",
]
