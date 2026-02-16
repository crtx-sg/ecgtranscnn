## 1. Preprocessing Module
- [x] 1.1 Create `ecg_transcovnet/preprocessing.py` with `FilterConfig`, `FILTER_PRESETS`, `PreprocessingPipeline`, `preprocess_ecg()`
- [x] 1.2 Implement per-lead filter pipeline: median (optional) -> highpass -> notch 50 Hz -> notch 60 Hz -> lowpass -> z-score
- [x] 1.3 Implement four presets: none, default, conservative, aggressive
- [x] 1.4 Precompute IIR filter coefficients in `PreprocessingPipeline.__init__()`
- [x] 1.5 Use lazy scipy imports (only when filtering is enabled)

## 2. Integration
- [x] 2.1 Export `FilterConfig`, `FILTER_PRESETS`, `PreprocessingPipeline`, `preprocess_ecg` from `__init__.py`
- [x] 2.2 Update `data.py`: add `filter_config` parameter to `generate_dataset()` and `load_hdf5_test_samples()`; replace inline z-score with pipeline
- [x] 2.3 Update `data.py`: include filter preset in cache key for `load_or_generate_data()`
- [x] 2.4 Update `scripts/processor.py`: add `--filter-preset` CLI arg, create pipeline once, pass to `process_file()`
- [x] 2.5 Update `scripts/train.py`: add `--filter-preset` CLI arg, pass through to data generation
- [x] 2.6 Update `scripts/evaluate.py`: add `--filter-preset` CLI arg, pass through to dataset generation and HDF5 evaluation
- [x] 2.7 Add `scipy>=1.10` to `pyproject.toml` dependencies

## 3. Testing
- [x] 3.1 Create `tests/test_preprocessing.py` with 14 unit tests
- [x] 3.2 Verify: shape preservation (7, 2400) -> (7, 2400) float32
- [x] 3.3 Verify: passthrough with preset `none` only applies z-score
- [x] 3.4 Verify: baseline wander (0.2 Hz) attenuated by >=20 dB
- [x] 3.5 Verify: 50 Hz powerline attenuated by >=20 dB
- [x] 3.6 Verify: 60 Hz powerline attenuated by >=20 dB
- [x] 3.7 Verify: QRS peak timing and amplitude preserved
- [x] 3.8 Verify: all presets instantiate and run without error
- [x] 3.9 Verify: performance <50 ms for 7 leads x 2400 samples
- [x] 3.10 Verify: existing tests still pass (`pytest tests/ -v`)

## 4. Documentation
- [x] 4.1 Update README.md with Preprocessing section (pipeline, presets, CLI usage, design notes)
- [x] 4.2 Update README.md CLI option tables to include `--filter-preset`
- [x] 4.3 Update README.md package structure and requirements
- [x] 4.4 Create OpenSpec change proposal, design, tasks, and delta specs
