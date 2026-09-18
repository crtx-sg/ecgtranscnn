# v1 test — avblock_fix

## crop2000

Events 4046 · accuracy **0.101** · macro-F1 **0.090** (subject-bootstrap 95 % CI 0.060–0.121) · macro recall 0.144 · macro specificity 0.945 · macro AUROC 0.623

Predictions outside the package head: 1084

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.927 | 0.062 | 0.997 | 0.117 | 0.625 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.515 | 0.255 | 0.974 | 0.341 | 0.694 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.575 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.068 | 0.114 | 0.820 | 0.085 | 0.462 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.016 | 0.151 | 0.875 | 0.029 | 0.717 | 53 | 4 |
| PAC | 0.060 | 0.028 | 0.984 | 0.038 | 0.388 | 142 | 34 |
| PVC | 0.674 | 0.069 | 0.991 | 0.126 | 0.603 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.442 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.075 | 0.901 | 0.744 | 0.138 | 0.919 | 91 | 6 |
| LBBB | 0.016 | 0.037 | 0.955 | 0.023 | 0.692 | 81 | 26 |
| RBBB | 0.111 | 0.014 | 0.998 | 0.025 | 0.715 | 71 | 20 |
| AV_BLOCK_1 | 0.333 | 0.100 | 0.999 | 0.154 | 0.648 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.042 | 0.118 | ATRIAL_FIBRILLATION 0.05 (210/5), ATRIAL_FLUTTER 0.19 (50/1) |
| `cudb` | 41 | 0.780 | 0.877 | VENTRICULAR_FIBRILLATION 0.88 (41/5) |
| `incart` | 1512 | 0.132 | 0.156 | NORMAL_SINUS 0.20 (458/5), SINUS_BRADYCARDIA 0.54 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.02 (50/4), PVC 0.17 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.001 | 0.002 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.00 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.093 | 0.104 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.19 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.24 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.14 (25/25), PVC 0.07 (50/50), LBBB 0.06 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |
| `vfdb` | 63 | 0.794 | 0.442 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.88 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.062 | 0.054 | NORMAL_SINUS 0.13 (758/11), PAC 0.01 (117/9), PVC 0.13 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.186 | 0.210 | SINUS_BRADYCARDIA 0.42 (342/7), SINUS_TACHYCARDIA 0.00 (146/6) |
| `record_level` | 1233 | 0.093 | 0.104 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.19 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.24 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.14 (25/25), PVC 0.07 (50/50), LBBB 0.06 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |
| `rhythm_annotation` | 476 | 0.197 | 0.166 | ATRIAL_FIBRILLATION 0.04 (310/7), ATRIAL_FLUTTER 0.19 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.44 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.255 | 0.197 | ATRIAL_FIBRILLATION 0.05 (210/5), ATRIAL_FLUTTER 0.19 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.55 (91/6) |
| `0100001` | 937 | 0.001 | 0.002 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.00 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.115 | 0.108 | NORMAL_SINUS 0.14 (1319/794), SINUS_BRADYCARDIA 0.45 (251/55), SINUS_TACHYCARDIA 0.00 (182/48), ATRIAL_FIBRILLATION 0.10 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.06 (75/29), PVC 0.16 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.04 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |

## full

Events 4046 · accuracy **0.106** · macro-F1 **0.096** (subject-bootstrap 95 % CI 0.066–0.123) · macro recall 0.159 · macro specificity 0.942 · macro AUROC 0.626

Predictions outside the package head: 936

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.947 | 0.056 | 0.998 | 0.105 | 0.621 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.544 | 0.268 | 0.976 | 0.359 | 0.720 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.512 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.063 | 0.116 | 0.799 | 0.082 | 0.469 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.030 | 0.264 | 0.885 | 0.053 | 0.805 | 53 | 4 |
| PAC | 0.048 | 0.021 | 0.985 | 0.029 | 0.326 | 142 | 34 |
| PVC | 0.743 | 0.089 | 0.992 | 0.160 | 0.629 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 0.999 | 0.000 | 0.446 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.070 | 0.923 | 0.719 | 0.130 | 0.903 | 91 | 6 |
| LBBB | 0.016 | 0.037 | 0.954 | 0.022 | 0.670 | 81 | 26 |
| RBBB | 0.200 | 0.028 | 0.998 | 0.049 | 0.763 | 71 | 20 |
| AV_BLOCK_1 | 0.375 | 0.100 | 0.999 | 0.158 | 0.650 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.069 | 0.204 | ATRIAL_FIBRILLATION 0.06 (210/5), ATRIAL_FLUTTER 0.35 (50/1) |
| `cudb` | 41 | 0.829 | 0.907 | VENTRICULAR_FIBRILLATION 0.91 (41/5) |
| `incart` | 1512 | 0.138 | 0.158 | NORMAL_SINUS 0.16 (458/5), SINUS_BRADYCARDIA 0.57 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.22 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.003 | 0.007 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.00 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.01 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.04 (50/1) |
| `ptbxl` | 1233 | 0.093 | 0.104 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.19 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.24 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.14 (25/25), PVC 0.07 (50/50), LBBB 0.06 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |
| `vfdb` | 63 | 0.794 | 0.446 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.89 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.065 | 0.062 | NORMAL_SINUS 0.10 (758/11), PAC 0.00 (117/9), PVC 0.17 (788/11), LBBB 0.00 (50/1), RBBB 0.04 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.197 | 0.219 | SINUS_BRADYCARDIA 0.44 (342/7), SINUS_TACHYCARDIA 0.00 (146/6) |
| `record_level` | 1233 | 0.093 | 0.104 | NORMAL_SINUS 0.11 (861/789), SINUS_BRADYCARDIA 0.19 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.24 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.14 (25/25), PVC 0.07 (50/50), LBBB 0.06 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |
| `rhythm_annotation` | 476 | 0.216 | 0.204 | ATRIAL_FIBRILLATION 0.04 (310/7), ATRIAL_FLUTTER 0.35 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.42 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.280 | 0.235 | ATRIAL_FIBRILLATION 0.06 (210/5), ATRIAL_FLUTTER 0.35 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.53 (91/6) |
| `0100001` | 937 | 0.003 | 0.007 | NORMAL_SINUS 0.00 (300/6), SINUS_BRADYCARDIA 0.00 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.01 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.04 (50/1) |
| `1111111` | 2745 | 0.118 | 0.110 | NORMAL_SINUS 0.13 (1319/794), SINUS_BRADYCARDIA 0.48 (251/55), SINUS_TACHYCARDIA 0.00 (182/48), ATRIAL_FIBRILLATION 0.10 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.04 (75/29), PVC 0.20 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.03 (31/25), RBBB 0.07 (21/19), AV_BLOCK_1 0.16 (30/28) |

## leadconv_0100001

Events 2745 · accuracy **0.001** · macro-F1 **0.001** (subject-bootstrap 95 % CI 0.000–0.003) · macro recall 0.003 · macro specificity 0.991 · macro AUROC 0.490

Predictions outside the package head: 772

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.000 | 0.000 | 1.000 | 0.000 | 0.493 | 1319 | 794 |
| SINUS_BRADYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.364 | 251 | 55 |
| SINUS_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.362 | 182 | 48 |
| ATRIAL_FIBRILLATION | 0.000 | 0.000 | 0.986 | 0.000 | 0.525 | 112 | 96 |
| ATRIAL_FLUTTER ⚠ | 0.000 | 0.000 | 0.998 | 0.000 | 0.208 | 3 | 3 |
| PAC | 0.000 | 0.000 | 1.000 | 0.000 | 0.371 | 75 | 29 |
| PVC | 1.000 | 0.002 | 1.000 | 0.003 | 0.657 | 635 | 55 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.000 | 0.000 | 1.000 | 0.000 | 0.778 | 86 | 1 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.380 | 0.000 | — | 0 | 0 |
| LBBB | 0.004 | 0.032 | 0.918 | 0.008 | 0.474 | 31 | 25 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.694 | 21 | 19 |
| AV_BLOCK_1 | 0.000 | 0.000 | 0.999 | 0.000 | 0.467 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.001 | 0.001 | NORMAL_SINUS 0.00 (458/5), SINUS_BRADYCARDIA 0.00 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `ptbxl` | 1233 | 0.001 | 0.001 | NORMAL_SINUS 0.00 (861/789), SINUS_BRADYCARDIA 0.00 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.00 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.01 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1093 | 0.001 | 0.001 | NORMAL_SINUS 0.00 (458/5), PAC 0.00 (50/4), PVC 0.00 (585/5) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 333 | 0.000 | 0.000 | SINUS_BRADYCARDIA 0.00 (197/3), SINUS_TACHYCARDIA 0.00 (136/4) |
| `record_level` | 1233 | 0.001 | 0.001 | NORMAL_SINUS 0.00 (861/789), SINUS_BRADYCARDIA 0.00 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.00 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.01 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2745 | 0.001 | 0.001 | NORMAL_SINUS 0.00 (1319/794), SINUS_BRADYCARDIA 0.00 (251/55), SINUS_TACHYCARDIA 0.00 (182/48), ATRIAL_FIBRILLATION 0.00 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.00 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.01 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

## leadconv_0100000

Events 3682 · accuracy **0.097** · macro-F1 **0.072** (subject-bootstrap 95 % CI 0.043–0.092) · macro recall 0.101 · macro specificity 0.963 · macro AUROC 0.648

Predictions outside the package head: 1667

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.930 | 0.057 | 0.997 | 0.108 | 0.621 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.528 | 0.624 | 0.933 | 0.572 | 0.899 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.794 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.056 | 0.042 | 0.956 | 0.048 | 0.645 | 212 | 98 |
| ATRIAL_FLUTTER ⚠ | 0.001 | 0.333 | 0.776 | 0.002 | 0.583 | 3 | 3 |
| PAC | 0.012 | 0.007 | 0.977 | 0.009 | 0.352 | 142 | 34 |
| PVC | 0.333 | 0.004 | 0.998 | 0.007 | 0.606 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.786 | 98 | 5 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.938 | 0.000 | — | 0 | 0 |
| LBBB | 0.009 | 0.012 | 0.970 | 0.011 | 0.693 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 0.999 | 0.000 | 0.544 | 71 | 20 |
| AV_BLOCK_1 | 0.034 | 0.033 | 0.992 | 0.034 | 0.601 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.130 | 0.146 | NORMAL_SINUS 0.18 (458/5), SINUS_BRADYCARDIA 0.70 (197/3), SINUS_TACHYCARDIA 0.00 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.101 | 0.078 | NORMAL_SINUS 0.08 (300/6), SINUS_BRADYCARDIA 0.56 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.03 (100/2), PAC 0.00 (67/5), PVC 0.03 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.052 | 0.052 | NORMAL_SINUS 0.08 (861/789), SINUS_BRADYCARDIA 0.22 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.09 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.04 (25/25), PVC 0.00 (50/50), LBBB 0.04 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.05 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.034 | 0.029 | NORMAL_SINUS 0.14 (758/11), PAC 0.00 (117/9), PVC 0.01 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.471 | 0.402 | SINUS_BRADYCARDIA 0.80 (342/7), SINUS_TACHYCARDIA 0.00 (146/6) |
| `record_level` | 1233 | 0.052 | 0.052 | NORMAL_SINUS 0.08 (861/789), SINUS_BRADYCARDIA 0.22 (54/52), SINUS_TACHYCARDIA 0.00 (46/44), ATRIAL_FIBRILLATION 0.09 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.04 (25/25), PVC 0.00 (50/50), LBBB 0.04 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.05 (30/28) |
| `rhythm_annotation` | 112 | 0.018 | 0.020 | ATRIAL_FIBRILLATION 0.04 (100/2), VENTRICULAR_TACHYCARDIA 0.00 (12/4) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 937 | 0.101 | 0.078 | NORMAL_SINUS 0.08 (300/6), SINUS_BRADYCARDIA 0.56 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.03 (100/2), PAC 0.00 (67/5), PVC 0.03 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.095 | 0.075 | NORMAL_SINUS 0.12 (1319/794), SINUS_BRADYCARDIA 0.58 (251/55), SINUS_TACHYCARDIA 0.00 (182/48), ATRIAL_FIBRILLATION 0.05 (112/96), ATRIAL_FLUTTER 0.00 (3/3), PAC 0.02 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.02 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.05 (30/28) |
