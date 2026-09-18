# v1 test — v1_f_leadfab05, v1_h_seed43, v1_h_seed44

## crop2000

Events 4046 · accuracy **0.790** · macro-F1 **0.628** (subject-bootstrap 95 % CI 0.574–0.744) · macro recall 0.674 · macro specificity 0.976 · macro AUROC 0.963

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.787 | 0.910 | 0.836 | 0.844 | 0.938 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.884 | 0.192 | 0.997 | 0.315 | 0.978 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.801 | 0.901 | 0.989 | 0.848 | 0.996 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.879 | 0.860 | 0.986 | 0.869 | 0.986 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.038 | 1.000 | 0.073 | 0.909 | 53 | 4 |
| PAC | 0.690 | 0.549 | 0.991 | 0.612 | 0.852 | 142 | 34 |
| PVC | 0.825 | 0.919 | 0.949 | 0.870 | 0.965 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.615 | 0.072 | 0.999 | 0.129 | 0.970 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.905 | 0.945 | 0.998 | 0.925 | 0.999 | 91 | 6 |
| LBBB | 0.950 | 0.938 | 0.999 | 0.944 | 0.999 | 81 | 26 |
| RBBB | 0.430 | 0.958 | 0.977 | 0.594 | 0.978 | 71 | 20 |
| AV_BLOCK_1 | 0.375 | 0.800 | 0.990 | 0.511 | 0.993 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.685 | 0.418 | ATRIAL_FIBRILLATION 0.84 (210/5), ATRIAL_FLUTTER 0.00 (50/1) |
| `cudb` | 41 | 0.927 | 0.962 | VENTRICULAR_FIBRILLATION 0.96 (41/5) |
| `incart` | 1512 | 0.735 | 0.591 | NORMAL_SINUS 0.74 (458/5), SINUS_BRADYCARDIA 0.42 (197/3), SINUS_TACHYCARDIA 0.89 (136/4), PAC 0.60 (50/4), PVC 0.88 (585/5), VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `mitbih` | 937 | 0.729 | 0.655 | NORMAL_SINUS 0.72 (300/6), SINUS_BRADYCARDIA 0.03 (145/4), SINUS_TACHYCARDIA 0.64 (10/2), ATRIAL_FIBRILLATION 0.88 (100/2), PAC 0.62 (67/5), PVC 0.89 (203/6), VENTRICULAR_TACHYCARDIA 0.15 (12/4), LBBB 0.95 (50/1), RBBB 1.00 (50/1) |
| `ptbxl` | 1233 | 0.920 | 0.817 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.52 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.62 (25/25), PVC 0.90 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.75 (30/28) |
| `vfdb` | 63 | 0.857 | 0.743 | VENTRICULAR_TACHYCARDIA 0.57 (13/2), VENTRICULAR_FIBRILLATION 0.91 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.863 | 0.786 | NORMAL_SINUS 0.88 (758/11), PAC 0.62 (117/9), PVC 0.95 (788/11), LBBB 0.95 (50/1), RBBB 0.54 (50/1) |
| `beat_run` | 86 | 0.012 | 0.023 | VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `rate_derived` | 488 | 0.383 | 0.613 | SINUS_BRADYCARDIA 0.28 (342/7), SINUS_TACHYCARDIA 0.95 (146/6) |
| `record_level` | 1233 | 0.920 | 0.817 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.52 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.62 (25/25), PVC 0.90 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.75 (30/28) |
| `rhythm_annotation` | 476 | 0.744 | 0.539 | ATRIAL_FIBRILLATION 0.85 (310/7), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.38 (25/6), VENTRICULAR_FIBRILLATION 0.92 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.742 | 0.565 | ATRIAL_FIBRILLATION 0.84 (210/5), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.50 (13/2), VENTRICULAR_FIBRILLATION 0.92 (91/6) |
| `0100001` | 937 | 0.729 | 0.655 | NORMAL_SINUS 0.72 (300/6), SINUS_BRADYCARDIA 0.03 (145/4), SINUS_TACHYCARDIA 0.64 (10/2), ATRIAL_FIBRILLATION 0.88 (100/2), PAC 0.62 (67/5), PVC 0.89 (203/6), VENTRICULAR_TACHYCARDIA 0.15 (12/4), LBBB 0.95 (50/1), RBBB 1.00 (50/1) |
| `1111111` | 2745 | 0.818 | 0.675 | NORMAL_SINUS 0.88 (1319/794), SINUS_BRADYCARDIA 0.44 (251/55), SINUS_TACHYCARDIA 0.89 (182/48), ATRIAL_FIBRILLATION 0.92 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.61 (75/29), PVC 0.88 (635/55), VENTRICULAR_TACHYCARDIA 0.02 (86/1), LBBB 0.95 (31/25), RBBB 0.28 (21/19), AV_BLOCK_1 0.75 (30/28) |

## full

Events 4046 · accuracy **0.808** · macro-F1 **0.644** (subject-bootstrap 95 % CI 0.589–0.760) · macro recall 0.687 · macro specificity 0.978 · macro AUROC 0.974

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.809 | 0.915 | 0.856 | 0.859 | 0.944 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.921 | 0.207 | 0.998 | 0.338 | 0.978 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.823 | 0.922 | 0.990 | 0.870 | 0.997 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.861 | 0.867 | 0.984 | 0.864 | 0.986 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.038 | 1.000 | 0.073 | 0.949 | 53 | 4 |
| PAC | 0.730 | 0.570 | 0.992 | 0.640 | 0.899 | 142 | 34 |
| PVC | 0.837 | 0.971 | 0.951 | 0.899 | 0.985 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.643 | 0.081 | 0.999 | 0.144 | 0.973 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.897 | 0.956 | 0.997 | 0.926 | 0.999 | 91 | 6 |
| LBBB | 0.951 | 0.963 | 0.999 | 0.957 | 0.999 | 81 | 26 |
| RBBB | 0.453 | 0.958 | 0.979 | 0.615 | 0.981 | 71 | 20 |
| AV_BLOCK_1 | 0.407 | 0.800 | 0.991 | 0.539 | 0.994 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.692 | 0.418 | ATRIAL_FIBRILLATION 0.84 (210/5), ATRIAL_FLUTTER 0.00 (50/1) |
| `cudb` | 41 | 0.902 | 0.949 | VENTRICULAR_FIBRILLATION 0.95 (41/5) |
| `incart` | 1512 | 0.769 | 0.621 | NORMAL_SINUS 0.77 (458/5), SINUS_BRADYCARDIA 0.45 (197/3), SINUS_TACHYCARDIA 0.92 (136/4), PAC 0.65 (50/4), PVC 0.91 (585/5), VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `mitbih` | 937 | 0.746 | 0.678 | NORMAL_SINUS 0.74 (300/6), SINUS_BRADYCARDIA 0.04 (145/4), SINUS_TACHYCARDIA 0.65 (10/2), ATRIAL_FIBRILLATION 0.88 (100/2), PAC 0.64 (67/5), PVC 0.91 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.97 (50/1), RBBB 1.00 (50/1) |
| `ptbxl` | 1233 | 0.920 | 0.817 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.52 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.62 (25/25), PVC 0.90 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.75 (30/28) |
| `vfdb` | 63 | 0.889 | 0.783 | VENTRICULAR_TACHYCARDIA 0.63 (13/2), VENTRICULAR_FIBRILLATION 0.93 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.895 | 0.814 | NORMAL_SINUS 0.91 (758/11), PAC 0.65 (117/9), PVC 0.97 (788/11), LBBB 0.97 (50/1), RBBB 0.56 (50/1) |
| `beat_run` | 86 | 0.012 | 0.023 | VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `rate_derived` | 488 | 0.404 | 0.634 | SINUS_BRADYCARDIA 0.30 (342/7), SINUS_TACHYCARDIA 0.96 (146/6) |
| `record_level` | 1233 | 0.920 | 0.817 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.52 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.62 (25/25), PVC 0.90 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.75 (30/28) |
| `rhythm_annotation` | 476 | 0.754 | 0.550 | ATRIAL_FIBRILLATION 0.85 (310/7), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.42 (25/6), VENTRICULAR_FIBRILLATION 0.93 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.750 | 0.571 | ATRIAL_FIBRILLATION 0.84 (210/5), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.52 (13/2), VENTRICULAR_FIBRILLATION 0.93 (91/6) |
| `0100001` | 937 | 0.746 | 0.678 | NORMAL_SINUS 0.74 (300/6), SINUS_BRADYCARDIA 0.04 (145/4), SINUS_TACHYCARDIA 0.65 (10/2), ATRIAL_FIBRILLATION 0.88 (100/2), PAC 0.64 (67/5), PVC 0.91 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.97 (50/1), RBBB 1.00 (50/1) |
| `1111111` | 2745 | 0.837 | 0.686 | NORMAL_SINUS 0.89 (1319/794), SINUS_BRADYCARDIA 0.47 (251/55), SINUS_TACHYCARDIA 0.91 (182/48), ATRIAL_FIBRILLATION 0.90 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.64 (75/29), PVC 0.91 (635/55), VENTRICULAR_TACHYCARDIA 0.02 (86/1), LBBB 0.95 (31/25), RBBB 0.30 (21/19), AV_BLOCK_1 0.75 (30/28) |

## leadconv_0100001

Events 2745 · accuracy **0.808** · macro-F1 **0.656** (subject-bootstrap 95 % CI 0.595–0.783) · macro recall 0.691 · macro specificity 0.974 · macro AUROC 0.963

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.837 | 0.914 | 0.835 | 0.874 | 0.927 | 1319 | 794 |
| SINUS_BRADYCARDIA | 0.857 | 0.263 | 0.996 | 0.402 | 0.968 | 251 | 55 |
| SINUS_TACHYCARDIA | 0.875 | 0.923 | 0.991 | 0.898 | 0.996 | 182 | 48 |
| ATRIAL_FIBRILLATION | 0.856 | 0.902 | 0.994 | 0.878 | 0.984 | 112 | 96 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 3 | 3 |
| PAC | 0.558 | 0.573 | 0.987 | 0.566 | 0.824 | 75 | 29 |
| PVC | 0.848 | 0.888 | 0.952 | 0.868 | 0.957 | 635 | 55 |
| VENTRICULAR_TACHYCARDIA ⚠ | 1.000 | 0.023 | 1.000 | 0.045 | 0.987 | 86 | 1 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 1.000 | 0.000 | — | 0 | 0 |
| LBBB | 0.909 | 0.968 | 0.999 | 0.937 | 0.999 | 31 | 25 |
| RBBB | 0.144 | 0.714 | 0.967 | 0.240 | 0.959 | 21 | 19 |
| AV_BLOCK_1 | 0.657 | 0.767 | 0.996 | 0.708 | 0.997 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.723 | 0.574 | NORMAL_SINUS 0.73 (458/5), SINUS_BRADYCARDIA 0.35 (197/3), SINUS_TACHYCARDIA 0.90 (136/4), PAC 0.54 (50/4), PVC 0.87 (585/5), VENTRICULAR_TACHYCARDIA 0.05 (86/1) |
| `ptbxl` | 1233 | 0.913 | 0.800 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.54 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.91 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), PVC 0.88 (50/50), LBBB 0.94 (31/25), RBBB 0.77 (21/19), AV_BLOCK_1 0.71 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1093 | 0.843 | 0.783 | NORMAL_SINUS 0.87 (458/5), PAC 0.54 (50/4), PVC 0.93 (585/5) |
| `beat_run` | 86 | 0.023 | 0.045 | VENTRICULAR_TACHYCARDIA 0.05 (86/1) |
| `rate_derived` | 333 | 0.511 | 0.662 | SINUS_BRADYCARDIA 0.36 (197/3), SINUS_TACHYCARDIA 0.97 (136/4) |
| `record_level` | 1233 | 0.913 | 0.800 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.54 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.91 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), PVC 0.88 (50/50), LBBB 0.94 (31/25), RBBB 0.77 (21/19), AV_BLOCK_1 0.71 (30/28) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2745 | 0.808 | 0.656 | NORMAL_SINUS 0.87 (1319/794), SINUS_BRADYCARDIA 0.40 (251/55), SINUS_TACHYCARDIA 0.90 (182/48), ATRIAL_FIBRILLATION 0.88 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.57 (75/29), PVC 0.87 (635/55), VENTRICULAR_TACHYCARDIA 0.05 (86/1), LBBB 0.94 (31/25), RBBB 0.24 (21/19), AV_BLOCK_1 0.71 (30/28) |

## leadconv_0100000

Events 3682 · accuracy **0.779** · macro-F1 **0.631** (subject-bootstrap 95 % CI 0.528–0.710) · macro recall 0.657 · macro specificity 0.971 · macro AUROC 0.964

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.785 | 0.910 | 0.805 | 0.843 | 0.925 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.881 | 0.187 | 0.997 | 0.308 | 0.966 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.848 | 0.901 | 0.991 | 0.874 | 0.995 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.843 | 0.858 | 0.990 | 0.850 | 0.989 | 212 | 98 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 3 | 3 |
| PAC | 0.626 | 0.542 | 0.987 | 0.581 | 0.845 | 142 | 34 |
| PVC | 0.839 | 0.889 | 0.950 | 0.863 | 0.955 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.600 | 0.031 | 0.999 | 0.058 | 0.978 | 98 | 5 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 1.000 | 0.000 | — | 0 | 0 |
| LBBB | 0.909 | 0.741 | 0.998 | 0.816 | 0.994 | 81 | 26 |
| RBBB | 0.388 | 0.831 | 0.974 | 0.529 | 0.972 | 71 | 20 |
| AV_BLOCK_1 | 0.308 | 0.667 | 0.988 | 0.421 | 0.988 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.714 | 0.565 | NORMAL_SINUS 0.74 (458/5), SINUS_BRADYCARDIA 0.36 (197/3), SINUS_TACHYCARDIA 0.88 (136/4), PAC 0.53 (50/4), PVC 0.86 (585/5), VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `mitbih` | 937 | 0.720 | 0.649 | NORMAL_SINUS 0.73 (300/6), SINUS_BRADYCARDIA 0.05 (145/4), SINUS_TACHYCARDIA 0.62 (10/2), ATRIAL_FIBRILLATION 0.87 (100/2), PAC 0.67 (67/5), PVC 0.89 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.76 (50/1), RBBB 0.98 (50/1) |
| `ptbxl` | 1233 | 0.903 | 0.757 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.60 (54/52), SINUS_TACHYCARDIA 0.92 (46/44), ATRIAL_FIBRILLATION 0.89 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.50 (25/25), PVC 0.85 (50/50), LBBB 0.92 (31/25), RBBB 0.49 (21/19), AV_BLOCK_1 0.65 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.846 | 0.739 | NORMAL_SINUS 0.89 (758/11), PAC 0.60 (117/9), PVC 0.93 (788/11), LBBB 0.74 (50/1), RBBB 0.54 (50/1) |
| `beat_run` | 86 | 0.012 | 0.023 | VENTRICULAR_TACHYCARDIA 0.02 (86/1) |
| `rate_derived` | 488 | 0.369 | 0.598 | SINUS_BRADYCARDIA 0.25 (342/7), SINUS_TACHYCARDIA 0.95 (146/6) |
| `record_level` | 1233 | 0.903 | 0.757 | NORMAL_SINUS 0.96 (861/789), SINUS_BRADYCARDIA 0.60 (54/52), SINUS_TACHYCARDIA 0.92 (46/44), ATRIAL_FIBRILLATION 0.89 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.50 (25/25), PVC 0.85 (50/50), LBBB 0.92 (31/25), RBBB 0.49 (21/19), AV_BLOCK_1 0.65 (30/28) |
| `rhythm_annotation` | 112 | 0.732 | 0.570 | ATRIAL_FIBRILLATION 0.87 (100/2), VENTRICULAR_TACHYCARDIA 0.27 (12/4) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 937 | 0.720 | 0.649 | NORMAL_SINUS 0.73 (300/6), SINUS_BRADYCARDIA 0.05 (145/4), SINUS_TACHYCARDIA 0.62 (10/2), ATRIAL_FIBRILLATION 0.87 (100/2), PAC 0.67 (67/5), PVC 0.89 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.76 (50/1), RBBB 0.98 (50/1) |
| `1111111` | 2745 | 0.799 | 0.628 | NORMAL_SINUS 0.87 (1319/794), SINUS_BRADYCARDIA 0.42 (251/55), SINUS_TACHYCARDIA 0.89 (182/48), ATRIAL_FIBRILLATION 0.83 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.52 (75/29), PVC 0.86 (635/55), VENTRICULAR_TACHYCARDIA 0.02 (86/1), LBBB 0.89 (31/25), RBBB 0.15 (21/19), AV_BLOCK_1 0.65 (30/28) |
