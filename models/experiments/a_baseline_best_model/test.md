# v1 test — models

## crop2000

Events 4046 · accuracy **0.124** · macro-F1 **0.098** (subject-bootstrap 95 % CI 0.063–0.125) · macro recall 0.191 · macro specificity 0.956 · macro AUROC 0.667

Predictions outside the package head: 1553

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.716 | 0.081 | 0.979 | 0.145 | 0.609 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.298 | 0.391 | 0.900 | 0.338 | 0.752 | 396 | 59 |
| SINUS_TACHYCARDIA | 1.000 | 0.010 | 1.000 | 0.021 | 0.622 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.162 | 0.199 | 0.880 | 0.178 | 0.651 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.130 | 0.585 | 0.948 | 0.212 | 0.902 | 53 | 4 |
| PAC | 0.000 | 0.000 | 0.993 | 0.000 | 0.636 | 142 | 34 |
| PVC | 0.214 | 0.014 | 0.986 | 0.027 | 0.442 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 0.994 | 0.000 | 0.363 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.116 | 0.945 | 0.834 | 0.207 | 0.961 | 91 | 6 |
| LBBB | 0.000 | 0.000 | 0.971 | 0.000 | 0.756 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.600 | 71 | 20 |
| AV_BLOCK_1 | 0.033 | 0.067 | 0.986 | 0.044 | 0.715 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.269 | 0.492 | ATRIAL_FIBRILLATION 0.33 (210/5), ATRIAL_FLUTTER 0.66 (50/1) |
| `cudb` | 41 | 0.878 | 0.935 | VENTRICULAR_FIBRILLATION 0.94 (41/5) |
| `incart` | 1512 | 0.102 | 0.116 | NORMAL_SINUS 0.16 (458/5), SINUS_BRADYCARDIA 0.52 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.02 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.022 | 0.025 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.17 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.01 (100/2), PAC 0.00 (67/5), PVC 0.04 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.139 | 0.092 | NORMAL_SINUS 0.19 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.08 (46/44), ATRIAL_FIBRILLATION 0.32 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.03 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.06 (30/28) |
| `vfdb` | 63 | 0.794 | 0.446 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.89 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.029 | 0.025 | NORMAL_SINUS 0.10 (758/11), PAC 0.00 (117/9), PVC 0.03 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.252 | 0.263 | SINUS_BRADYCARDIA 0.53 (342/7), SINUS_TACHYCARDIA 0.00 (146/6) |
| `record_level` | 1233 | 0.139 | 0.092 | NORMAL_SINUS 0.19 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.08 (46/44), ATRIAL_FIBRILLATION 0.32 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.03 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.06 (30/28) |
| `rhythm_annotation` | 476 | 0.330 | 0.365 | ATRIAL_FIBRILLATION 0.24 (310/7), ATRIAL_FLUTTER 0.66 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.56 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.429 | 0.396 | ATRIAL_FIBRILLATION 0.32 (210/5), ATRIAL_FLUTTER 0.66 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.60 (91/6) |
| `0100001` | 937 | 0.022 | 0.025 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.17 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.01 (100/2), PAC 0.00 (67/5), PVC 0.04 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.119 | 0.076 | NORMAL_SINUS 0.18 (1319/794), SINUS_BRADYCARDIA 0.39 (251/55), SINUS_TACHYCARDIA 0.02 (182/48), ATRIAL_FIBRILLATION 0.15 (112/96), ATRIAL_FLUTTER 0.02 (3/3), PAC 0.00 (75/29), PVC 0.02 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.05 (30/28) |

## full

Events 4046 · accuracy **0.125** · macro-F1 **0.100** (subject-bootstrap 95 % CI 0.064–0.130) · macro recall 0.197 · macro specificity 0.955 · macro AUROC 0.665

Predictions outside the package head: 1501

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.760 | 0.070 | 0.985 | 0.129 | 0.604 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.330 | 0.419 | 0.908 | 0.369 | 0.758 | 396 | 59 |
| SINUS_TACHYCARDIA | 1.000 | 0.010 | 1.000 | 0.021 | 0.586 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.150 | 0.206 | 0.864 | 0.174 | 0.640 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.146 | 0.660 | 0.949 | 0.240 | 0.915 | 53 | 4 |
| PAC | 0.000 | 0.000 | 0.993 | 0.000 | 0.633 | 142 | 34 |
| PVC | 0.211 | 0.018 | 0.983 | 0.033 | 0.467 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 0.995 | 0.000 | 0.371 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.108 | 0.912 | 0.828 | 0.194 | 0.944 | 91 | 6 |
| LBBB | 0.000 | 0.000 | 0.967 | 0.000 | 0.746 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.600 | 71 | 20 |
| AV_BLOCK_1 | 0.035 | 0.067 | 0.986 | 0.046 | 0.719 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.300 | 0.547 | ATRIAL_FIBRILLATION 0.35 (210/5), ATRIAL_FLUTTER 0.74 (50/1) |
| `cudb` | 41 | 0.829 | 0.907 | VENTRICULAR_FIBRILLATION 0.91 (41/5) |
| `incart` | 1512 | 0.099 | 0.117 | NORMAL_SINUS 0.10 (458/5), SINUS_BRADYCARDIA 0.59 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.02 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.022 | 0.025 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.16 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.07 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.139 | 0.092 | NORMAL_SINUS 0.19 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.08 (46/44), ATRIAL_FIBRILLATION 0.32 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.03 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.06 (30/28) |
| `vfdb` | 63 | 0.778 | 0.437 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.87 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.021 | 0.018 | NORMAL_SINUS 0.06 (758/11), PAC 0.00 (117/9), PVC 0.03 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.275 | 0.280 | SINUS_BRADYCARDIA 0.56 (342/7), SINUS_TACHYCARDIA 0.00 (146/6) |
| `record_level` | 1233 | 0.139 | 0.092 | NORMAL_SINUS 0.19 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.08 (46/44), ATRIAL_FIBRILLATION 0.32 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.03 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.06 (30/28) |
| `rhythm_annotation` | 476 | 0.338 | 0.382 | ATRIAL_FIBRILLATION 0.25 (310/7), ATRIAL_FLUTTER 0.74 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.53 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.442 | 0.419 | ATRIAL_FIBRILLATION 0.35 (210/5), ATRIAL_FLUTTER 0.74 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.58 (91/6) |
| `0100001` | 937 | 0.022 | 0.025 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.16 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.07 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.117 | 0.076 | NORMAL_SINUS 0.16 (1319/794), SINUS_BRADYCARDIA 0.42 (251/55), SINUS_TACHYCARDIA 0.02 (182/48), ATRIAL_FIBRILLATION 0.14 (112/96), ATRIAL_FLUTTER 0.02 (3/3), PAC 0.00 (75/29), PVC 0.02 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.05 (30/28) |

## leadconv_0100001

Events 2745 · accuracy **0.040** · macro-F1 **0.044** (subject-bootstrap 95 % CI 0.025–0.072) · macro recall 0.047 · macro specificity 0.985 · macro AUROC 0.560

Predictions outside the package head: 1263

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.000 | 0.000 | 0.995 | 0.000 | 0.620 | 1319 | 794 |
| SINUS_BRADYCARDIA | 0.336 | 0.359 | 0.929 | 0.347 | 0.732 | 251 | 55 |
| SINUS_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.533 | 182 | 48 |
| ATRIAL_FIBRILLATION | 0.046 | 0.045 | 0.961 | 0.045 | 0.639 | 112 | 96 |
| ATRIAL_FLUTTER ⚠ | 0.000 | 0.000 | 0.996 | 0.000 | 0.390 | 3 | 3 |
| PAC | 0.000 | 0.000 | 1.000 | 0.000 | 0.784 | 75 | 29 |
| PVC | 0.379 | 0.017 | 0.991 | 0.033 | 0.448 | 635 | 55 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.000 | 0.000 | 0.995 | 0.000 | 0.218 | 86 | 1 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.647 | 0.000 | — | 0 | 0 |
| LBBB | 0.038 | 0.097 | 0.972 | 0.055 | 0.826 | 31 | 25 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.351 | 21 | 19 |
| AV_BLOCK_1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.618 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.049 | 0.066 | NORMAL_SINUS 0.00 (458/5), SINUS_BRADYCARDIA 0.37 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.03 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `ptbxl` | 1233 | 0.028 | 0.059 | NORMAL_SINUS 0.00 (861/789), SINUS_BRADYCARDIA 0.30 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.07 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.06 (50/50), LBBB 0.15 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1093 | 0.008 | 0.010 | NORMAL_SINUS 0.00 (458/5), PAC 0.00 (50/4), PVC 0.03 (585/5) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 333 | 0.195 | 0.247 | SINUS_BRADYCARDIA 0.49 (197/3), SINUS_TACHYCARDIA 0.00 (136/4) |
| `record_level` | 1233 | 0.028 | 0.059 | NORMAL_SINUS 0.00 (861/789), SINUS_BRADYCARDIA 0.30 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.07 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.06 (50/50), LBBB 0.15 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2745 | 0.040 | 0.044 | NORMAL_SINUS 0.00 (1319/794), SINUS_BRADYCARDIA 0.35 (251/55), SINUS_TACHYCARDIA 0.00 (182/48), ATRIAL_FIBRILLATION 0.05 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (75/29), PVC 0.03 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.05 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

## leadconv_0100000

Events 3682 · accuracy **0.159** · macro-F1 **0.082** (subject-bootstrap 95 % CI 0.060–0.100) · macro recall 0.115 · macro specificity 0.950 · macro AUROC 0.653

Predictions outside the package head: 907

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.821 | 0.116 | 0.980 | 0.203 | 0.683 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.323 | 0.874 | 0.780 | 0.472 | 0.880 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.857 | 0.031 | 1.000 | 0.060 | 0.834 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.104 | 0.198 | 0.896 | 0.137 | 0.744 | 212 | 98 |
| ATRIAL_FLUTTER ⚠ | 0.000 | 0.000 | 0.873 | 0.000 | 0.628 | 3 | 3 |
| PAC | 0.000 | 0.000 | 0.992 | 0.000 | 0.715 | 142 | 34 |
| PVC | 0.500 | 0.001 | 1.000 | 0.002 | 0.425 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 0.991 | 0.000 | 0.435 | 98 | 5 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.920 | 0.000 | — | 0 | 0 |
| LBBB | 0.010 | 0.012 | 0.974 | 0.011 | 0.769 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.576 | 71 | 20 |
| AV_BLOCK_1 | 0.007 | 0.033 | 0.961 | 0.011 | 0.494 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.195 | 0.169 | NORMAL_SINUS 0.35 (458/5), SINUS_BRADYCARDIA 0.58 (197/3), SINUS_TACHYCARDIA 0.08 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.188 | 0.109 | NORMAL_SINUS 0.20 (300/6), SINUS_BRADYCARDIA 0.58 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.20 (100/2), PAC 0.00 (67/5), PVC 0.01 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.092 | 0.056 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.20 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.18 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.06 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.02 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.078 | 0.060 | NORMAL_SINUS 0.30 (758/11), PAC 0.00 (117/9), PVC 0.00 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.641 | 0.512 | SINUS_BRADYCARDIA 0.94 (342/7), SINUS_TACHYCARDIA 0.08 (146/6) |
| `record_level` | 1233 | 0.092 | 0.056 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.20 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.18 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.06 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.02 (30/28) |
| `rhythm_annotation` | 112 | 0.188 | 0.168 | ATRIAL_FIBRILLATION 0.34 (100/2), VENTRICULAR_TACHYCARDIA 0.00 (12/4) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 937 | 0.188 | 0.109 | NORMAL_SINUS 0.20 (300/6), SINUS_BRADYCARDIA 0.58 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.20 (100/2), PAC 0.00 (67/5), PVC 0.01 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.149 | 0.076 | NORMAL_SINUS 0.21 (1319/794), SINUS_BRADYCARDIA 0.43 (251/55), SINUS_TACHYCARDIA 0.06 (182/48), ATRIAL_FIBRILLATION 0.10 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.02 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.02 (30/28) |
