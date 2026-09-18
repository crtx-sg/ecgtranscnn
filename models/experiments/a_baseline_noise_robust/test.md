# v1 test — noise_robust

## crop2000

Events 4046 · accuracy **0.255** · macro-F1 **0.176** (subject-bootstrap 95 % CI 0.133–0.211) · macro recall 0.256 · macro specificity 0.960 · macro AUROC 0.650

Predictions outside the package head: 1243

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.765 | 0.307 | 0.937 | 0.438 | 0.685 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.356 | 0.593 | 0.883 | 0.445 | 0.807 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.612 | 0.599 | 0.981 | 0.605 | 0.910 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.333 | 0.187 | 0.956 | 0.240 | 0.653 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.084 | 0.509 | 0.926 | 0.144 | 0.915 | 53 | 4 |
| PAC | 0.000 | 0.000 | 0.996 | 0.000 | 0.536 | 142 | 34 |
| PVC | 1.000 | 0.001 | 1.000 | 0.002 | 0.369 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.211 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.126 | 0.846 | 0.865 | 0.219 | 0.922 | 91 | 6 |
| LBBB | 0.000 | 0.000 | 0.999 | 0.000 | 0.599 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 0.999 | 0.000 | 0.578 | 71 | 20 |
| AV_BLOCK_1 | 0.009 | 0.033 | 0.973 | 0.014 | 0.610 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.250 | 0.468 | ATRIAL_FIBRILLATION 0.32 (210/5), ATRIAL_FLUTTER 0.62 (50/1) |
| `cudb` | 41 | 0.854 | 0.921 | VENTRICULAR_FIBRILLATION 0.92 (41/5) |
| `incart` | 1512 | 0.313 | 0.337 | NORMAL_SINUS 0.57 (458/5), SINUS_BRADYCARDIA 0.74 (197/3), SINUS_TACHYCARDIA 0.71 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.094 | 0.064 | NORMAL_SINUS 0.30 (300/6), SINUS_BRADYCARDIA 0.28 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.267 | 0.147 | NORMAL_SINUS 0.42 (861/789), SINUS_BRADYCARDIA 0.18 (54/52), SINUS_TACHYCARDIA 0.44 (46/44), ATRIAL_FIBRILLATION 0.36 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.03 (30/28) |
| `vfdb` | 63 | 0.667 | 0.404 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.81 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.149 | 0.098 | NORMAL_SINUS 0.49 (758/11), PAC 0.00 (117/9), PVC 0.00 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.611 | 0.771 | SINUS_BRADYCARDIA 0.74 (342/7), SINUS_TACHYCARDIA 0.80 (146/6) |
| `record_level` | 1233 | 0.267 | 0.147 | NORMAL_SINUS 0.42 (861/789), SINUS_BRADYCARDIA 0.18 (54/52), SINUS_TACHYCARDIA 0.44 (46/44), ATRIAL_FIBRILLATION 0.36 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.03 (30/28) |
| `rhythm_annotation` | 476 | 0.298 | 0.328 | ATRIAL_FIBRILLATION 0.23 (310/7), ATRIAL_FLUTTER 0.55 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.54 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.390 | 0.356 | ATRIAL_FIBRILLATION 0.31 (210/5), ATRIAL_FLUTTER 0.56 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.55 (91/6) |
| `0100001` | 937 | 0.094 | 0.064 | NORMAL_SINUS 0.30 (300/6), SINUS_BRADYCARDIA 0.28 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.292 | 0.175 | NORMAL_SINUS 0.48 (1319/794), SINUS_BRADYCARDIA 0.49 (251/55), SINUS_TACHYCARDIA 0.65 (182/48), ATRIAL_FIBRILLATION 0.26 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.02 (30/28) |

## full

Events 4046 · accuracy **0.263** · macro-F1 **0.181** (subject-bootstrap 95 % CI 0.134–0.222) · macro recall 0.259 · macro specificity 0.960 · macro AUROC 0.647

Predictions outside the package head: 1229

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.745 | 0.319 | 0.927 | 0.447 | 0.689 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.383 | 0.588 | 0.897 | 0.464 | 0.788 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.615 | 0.641 | 0.980 | 0.628 | 0.912 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.345 | 0.192 | 0.958 | 0.247 | 0.651 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 0.067 | 0.415 | 0.924 | 0.116 | 0.906 | 53 | 4 |
| PAC | 0.038 | 0.007 | 0.994 | 0.012 | 0.538 | 142 | 34 |
| PVC | 0.667 | 0.002 | 1.000 | 0.005 | 0.376 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.210 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.137 | 0.912 | 0.868 | 0.239 | 0.920 | 91 | 6 |
| LBBB | 0.000 | 0.000 | 0.999 | 0.000 | 0.606 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 0.999 | 0.000 | 0.564 | 71 | 20 |
| AV_BLOCK_1 | 0.009 | 0.033 | 0.972 | 0.014 | 0.608 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.235 | 0.426 | ATRIAL_FIBRILLATION 0.33 (210/5), ATRIAL_FLUTTER 0.53 (50/1) |
| `cudb` | 41 | 0.829 | 0.907 | VENTRICULAR_FIBRILLATION 0.91 (41/5) |
| `incart` | 1512 | 0.333 | 0.363 | NORMAL_SINUS 0.60 (458/5), SINUS_BRADYCARDIA 0.80 (197/3), SINUS_TACHYCARDIA 0.74 (136/4), PAC 0.03 (50/4), PVC 0.01 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.093 | 0.062 | NORMAL_SINUS 0.31 (300/6), SINUS_BRADYCARDIA 0.23 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.267 | 0.147 | NORMAL_SINUS 0.42 (861/789), SINUS_BRADYCARDIA 0.18 (54/52), SINUS_TACHYCARDIA 0.44 (46/44), ATRIAL_FIBRILLATION 0.36 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.03 (30/28) |
| `vfdb` | 63 | 0.778 | 0.441 | VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.88 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.162 | 0.106 | NORMAL_SINUS 0.51 (758/11), PAC 0.02 (117/9), PVC 0.01 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.623 | 0.787 | SINUS_BRADYCARDIA 0.73 (342/7), SINUS_TACHYCARDIA 0.84 (146/6) |
| `record_level` | 1233 | 0.267 | 0.147 | NORMAL_SINUS 0.42 (861/789), SINUS_BRADYCARDIA 0.18 (54/52), SINUS_TACHYCARDIA 0.44 (46/44), ATRIAL_FIBRILLATION 0.36 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.03 (30/28) |
| `rhythm_annotation` | 476 | 0.305 | 0.326 | ATRIAL_FIBRILLATION 0.24 (310/7), ATRIAL_FLUTTER 0.51 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (25/6), VENTRICULAR_FIBRILLATION 0.56 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.396 | 0.352 | ATRIAL_FIBRILLATION 0.32 (210/5), ATRIAL_FLUTTER 0.51 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (13/2), VENTRICULAR_FIBRILLATION 0.57 (91/6) |
| `0100001` | 937 | 0.093 | 0.062 | NORMAL_SINUS 0.31 (300/6), SINUS_BRADYCARDIA 0.23 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.02 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.303 | 0.184 | NORMAL_SINUS 0.49 (1319/794), SINUS_BRADYCARDIA 0.52 (251/55), SINUS_TACHYCARDIA 0.68 (182/48), ATRIAL_FIBRILLATION 0.27 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.02 (75/29), PVC 0.01 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.02 (30/28) |

## leadconv_0100001

Events 2745 · accuracy **0.185** · macro-F1 **0.117** (subject-bootstrap 95 % CI 0.095–0.135) · macro recall 0.152 · macro specificity 0.973 · macro AUROC 0.619

Predictions outside the package head: 930

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.819 | 0.265 | 0.946 | 0.400 | 0.777 | 1319 | 794 |
| SINUS_BRADYCARDIA | 0.464 | 0.414 | 0.952 | 0.438 | 0.871 | 251 | 55 |
| SINUS_TACHYCARDIA | 0.652 | 0.236 | 0.991 | 0.347 | 0.893 | 182 | 48 |
| ATRIAL_FIBRILLATION | 0.105 | 0.089 | 0.968 | 0.097 | 0.720 | 112 | 96 |
| ATRIAL_FLUTTER ⚠ | 0.005 | 0.667 | 0.854 | 0.010 | 0.908 | 3 | 3 |
| PAC | 0.000 | 0.000 | 0.999 | 0.000 | 0.516 | 75 | 29 |
| PVC | 0.000 | 0.000 | 1.000 | 0.000 | 0.231 | 635 | 55 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.000 | 0.000 | 1.000 | 0.000 | 0.035 | 86 | 1 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.787 | 0.000 | — | 0 | 0 |
| LBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.766 | 31 | 25 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.456 | 21 | 19 |
| AV_BLOCK_1 | 0.000 | 0.000 | 0.994 | 0.000 | 0.639 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.189 | 0.241 | NORMAL_SINUS 0.51 (458/5), SINUS_BRADYCARDIA 0.53 (197/3), SINUS_TACHYCARDIA 0.41 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `ptbxl` | 1233 | 0.180 | 0.089 | NORMAL_SINUS 0.34 (861/789), SINUS_BRADYCARDIA 0.28 (54/52), SINUS_TACHYCARDIA 0.12 (46/44), ATRIAL_FIBRILLATION 0.14 (112/96), ATRIAL_FLUTTER 0.02 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1093 | 0.154 | 0.174 | NORMAL_SINUS 0.52 (458/5), PAC 0.00 (50/4), PVC 0.00 (585/5) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 333 | 0.354 | 0.511 | SINUS_BRADYCARDIA 0.57 (197/3), SINUS_TACHYCARDIA 0.45 (136/4) |
| `record_level` | 1233 | 0.180 | 0.089 | NORMAL_SINUS 0.34 (861/789), SINUS_BRADYCARDIA 0.28 (54/52), SINUS_TACHYCARDIA 0.12 (46/44), ATRIAL_FIBRILLATION 0.14 (112/96), ATRIAL_FLUTTER 0.02 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2745 | 0.185 | 0.117 | NORMAL_SINUS 0.40 (1319/794), SINUS_BRADYCARDIA 0.44 (251/55), SINUS_TACHYCARDIA 0.35 (182/48), ATRIAL_FIBRILLATION 0.10 (112/96), ATRIAL_FLUTTER 0.01 (3/3), PAC 0.00 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.00 (30/28) |

## leadconv_0100000

Events 3682 · accuracy **0.260** · macro-F1 **0.140** (subject-bootstrap 95 % CI 0.099–0.169) · macro recall 0.263 · macro specificity 0.954 · macro AUROC 0.614

Predictions outside the package head: 812

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.730 | 0.302 | 0.912 | 0.427 | 0.728 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.359 | 0.848 | 0.817 | 0.504 | 0.915 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.387 | 0.573 | 0.950 | 0.462 | 0.913 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.158 | 0.071 | 0.977 | 0.098 | 0.682 | 212 | 98 |
| ATRIAL_FLUTTER ⚠ | 0.006 | 1.000 | 0.873 | 0.013 | 0.990 | 3 | 3 |
| PAC | 0.000 | 0.000 | 0.999 | 0.000 | 0.550 | 142 | 34 |
| PVC | 0.000 | 0.000 | 1.000 | 0.000 | 0.307 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.000 | 0.000 | 1.000 | 0.000 | 0.085 | 98 | 5 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.922 | 0.000 | — | 0 | 0 |
| LBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.645 | 81 | 26 |
| RBBB | 0.000 | 0.000 | 1.000 | 0.000 | 0.386 | 71 | 20 |
| AV_BLOCK_1 | 0.025 | 0.100 | 0.968 | 0.040 | 0.549 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.333 | 0.299 | NORMAL_SINUS 0.58 (458/5), SINUS_BRADYCARDIA 0.69 (197/3), SINUS_TACHYCARDIA 0.52 (136/4), PAC 0.00 (50/4), PVC 0.00 (585/5), VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `mitbih` | 937 | 0.171 | 0.094 | NORMAL_SINUS 0.28 (300/6), SINUS_BRADYCARDIA 0.56 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `ptbxl` | 1233 | 0.237 | 0.119 | NORMAL_SINUS 0.38 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.31 (46/44), ATRIAL_FIBRILLATION 0.19 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.08 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.155 | 0.100 | NORMAL_SINUS 0.50 (758/11), PAC 0.00 (117/9), PVC 0.00 (788/11), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `beat_run` | 86 | 0.000 | 0.000 | VENTRICULAR_TACHYCARDIA 0.00 (86/1) |
| `rate_derived` | 488 | 0.801 | 0.865 | SINUS_BRADYCARDIA 0.92 (342/7), SINUS_TACHYCARDIA 0.81 (146/6) |
| `record_level` | 1233 | 0.237 | 0.119 | NORMAL_SINUS 0.38 (861/789), SINUS_BRADYCARDIA 0.21 (54/52), SINUS_TACHYCARDIA 0.31 (46/44), ATRIAL_FIBRILLATION 0.19 (112/96), ATRIAL_FLUTTER 0.03 (3/3), PAC 0.00 (25/25), PVC 0.00 (50/50), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.08 (30/28) |
| `rhythm_annotation` | 112 | 0.000 | 0.000 | ATRIAL_FIBRILLATION 0.00 (100/2), VENTRICULAR_TACHYCARDIA 0.00 (12/4) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 937 | 0.171 | 0.094 | NORMAL_SINUS 0.28 (300/6), SINUS_BRADYCARDIA 0.56 (145/4), SINUS_TACHYCARDIA 0.00 (10/2), ATRIAL_FIBRILLATION 0.00 (100/2), PAC 0.00 (67/5), PVC 0.00 (203/6), VENTRICULAR_TACHYCARDIA 0.00 (12/4), LBBB 0.00 (50/1), RBBB 0.00 (50/1) |
| `1111111` | 2745 | 0.290 | 0.152 | NORMAL_SINUS 0.46 (1319/794), SINUS_BRADYCARDIA 0.48 (251/55), SINUS_TACHYCARDIA 0.49 (182/48), ATRIAL_FIBRILLATION 0.16 (112/96), ATRIAL_FLUTTER 0.02 (3/3), PAC 0.00 (75/29), PVC 0.00 (635/55), VENTRICULAR_TACHYCARDIA 0.00 (86/1), LBBB 0.00 (31/25), RBBB 0.00 (21/19), AV_BLOCK_1 0.06 (30/28) |
