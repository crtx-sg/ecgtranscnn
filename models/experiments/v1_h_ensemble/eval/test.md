# v1 test — v1_f_leadfab05, v1_h_seed43, v1_h_seed44

## crop2000

Events 4046 · accuracy **0.814** · macro-F1 **0.668** (subject-bootstrap 95 % CI 0.610–0.777) · macro recall 0.741 · macro specificity 0.982 · macro AUROC 0.963

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.911 | 0.810 | 0.947 | 0.858 | 0.938 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.728 | 0.785 | 0.968 | 0.756 | 0.978 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.660 | 0.990 | 0.975 | 0.792 | 0.996 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.872 | 0.905 | 0.985 | 0.888 | 0.986 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.057 | 1.000 | 0.107 | 0.909 | 53 | 4 |
| PAC | 0.568 | 0.556 | 0.985 | 0.562 | 0.852 | 142 | 34 |
| PVC | 0.873 | 0.878 | 0.967 | 0.876 | 0.965 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.742 | 0.207 | 0.998 | 0.324 | 0.970 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.878 | 0.945 | 0.997 | 0.910 | 0.999 | 91 | 6 |
| LBBB | 0.895 | 0.951 | 0.998 | 0.922 | 0.999 | 81 | 26 |
| RBBB | 0.404 | 0.972 | 0.974 | 0.570 | 0.978 | 71 | 20 |
| AV_BLOCK_1 | 0.305 | 0.833 | 0.986 | 0.446 | 0.993 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.715 | 0.453 | ATRIAL_FIBRILLATION 0.87 (210/5), ATRIAL_FLUTTER 0.04 (50/1) |
| `cudb` | 41 | 0.927 | 0.962 | VENTRICULAR_FIBRILLATION 0.96 (41/5) |
| `incart` | 1512 | 0.786 | 0.699 | NORMAL_SINUS 0.81 (458/5), SINUS_BRADYCARDIA 0.87 (197/3), SINUS_TACHYCARDIA 0.85 (136/4), PAC 0.51 (50/4), PVC 0.86 (585/5), VENTRICULAR_TACHYCARDIA 0.29 (86/1) |
| `mitbih` | 937 | 0.793 | 0.727 | NORMAL_SINUS 0.75 (300/6), SINUS_BRADYCARDIA 0.75 (145/4), SINUS_TACHYCARDIA 0.43 (10/2), ATRIAL_FIBRILLATION 0.93 (100/2), PAC 0.55 (67/5), PVC 0.92 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.96 (50/1), RBBB 1.00 (50/1) |
| `ptbxl` | 1233 | 0.878 | 0.814 | NORMAL_SINUS 0.92 (861/789), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.91 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.68 (25/25), PVC 0.91 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.68 (30/28) |
| `vfdb` | 63 | 0.857 | 0.743 | VENTRICULAR_TACHYCARDIA 0.57 (13/2), VENTRICULAR_FIBRILLATION 0.91 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.799 | 0.751 | NORMAL_SINUS 0.83 (758/11), PAC 0.53 (117/9), PVC 0.92 (788/11), LBBB 0.96 (50/1), RBBB 0.51 (50/1) |
| `beat_run` | 86 | 0.174 | 0.297 | VENTRICULAR_TACHYCARDIA 0.30 (86/1) |
| `rate_derived` | 488 | 0.848 | 0.939 | SINUS_BRADYCARDIA 0.88 (342/7), SINUS_TACHYCARDIA 1.00 (146/6) |
| `record_level` | 1233 | 0.878 | 0.814 | NORMAL_SINUS 0.92 (861/789), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.91 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.68 (25/25), PVC 0.91 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.68 (30/28) |
| `rhythm_annotation` | 476 | 0.784 | 0.562 | ATRIAL_FIBRILLATION 0.89 (310/7), ATRIAL_FLUTTER 0.04 (50/1), VENTRICULAR_TACHYCARDIA 0.41 (25/6), VENTRICULAR_FIBRILLATION 0.91 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.764 | 0.579 | ATRIAL_FIBRILLATION 0.87 (210/5), ATRIAL_FLUTTER 0.04 (50/1), VENTRICULAR_TACHYCARDIA 0.50 (13/2), VENTRICULAR_FIBRILLATION 0.91 (91/6) |
| `0100001` | 937 | 0.793 | 0.727 | NORMAL_SINUS 0.75 (300/6), SINUS_BRADYCARDIA 0.75 (145/4), SINUS_TACHYCARDIA 0.43 (10/2), ATRIAL_FIBRILLATION 0.93 (100/2), PAC 0.55 (67/5), PVC 0.92 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.96 (50/1), RBBB 1.00 (50/1) |
| `1111111` | 2745 | 0.828 | 0.713 | NORMAL_SINUS 0.88 (1319/794), SINUS_BRADYCARDIA 0.76 (251/55), SINUS_TACHYCARDIA 0.87 (182/48), ATRIAL_FIBRILLATION 0.90 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.57 (75/29), PVC 0.87 (635/55), VENTRICULAR_TACHYCARDIA 0.29 (86/1), LBBB 0.95 (31/25), RBBB 0.27 (21/19), AV_BLOCK_1 0.68 (30/28) |

## full

Events 4046 · accuracy **0.829** · macro-F1 **0.678** (subject-bootstrap 95 % CI 0.615–0.790) · macro recall 0.749 · macro specificity 0.983 · macro AUROC 0.974

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.930 | 0.815 | 0.959 | 0.869 | 0.944 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.737 | 0.778 | 0.970 | 0.757 | 0.978 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.676 | 0.990 | 0.976 | 0.803 | 0.997 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.855 | 0.922 | 0.982 | 0.887 | 0.986 | 422 | 103 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.038 | 1.000 | 0.073 | 0.949 | 53 | 4 |
| PAC | 0.620 | 0.563 | 0.987 | 0.590 | 0.899 | 142 | 34 |
| PVC | 0.884 | 0.932 | 0.968 | 0.908 | 0.985 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.828 | 0.216 | 0.999 | 0.343 | 0.973 | 111 | 7 |
| VENTRICULAR_FIBRILLATION | 0.888 | 0.956 | 0.997 | 0.921 | 0.999 | 91 | 6 |
| LBBB | 0.888 | 0.975 | 0.997 | 0.929 | 0.999 | 81 | 26 |
| RBBB | 0.408 | 0.972 | 0.975 | 0.575 | 0.981 | 71 | 20 |
| AV_BLOCK_1 | 0.338 | 0.833 | 0.988 | 0.481 | 0.994 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 260 | 0.723 | 0.434 | ATRIAL_FIBRILLATION 0.87 (210/5), ATRIAL_FLUTTER 0.00 (50/1) |
| `cudb` | 41 | 0.902 | 0.949 | VENTRICULAR_FIBRILLATION 0.95 (41/5) |
| `incart` | 1512 | 0.813 | 0.722 | NORMAL_SINUS 0.83 (458/5), SINUS_BRADYCARDIA 0.88 (197/3), SINUS_TACHYCARDIA 0.88 (136/4), PAC 0.53 (50/4), PVC 0.90 (585/5), VENTRICULAR_TACHYCARDIA 0.31 (86/1) |
| `mitbih` | 937 | 0.812 | 0.737 | NORMAL_SINUS 0.78 (300/6), SINUS_BRADYCARDIA 0.76 (145/4), SINUS_TACHYCARDIA 0.38 (10/2), ATRIAL_FIBRILLATION 0.94 (100/2), PAC 0.60 (67/5), PVC 0.93 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.98 (50/1), RBBB 1.00 (50/1) |
| `ptbxl` | 1233 | 0.878 | 0.814 | NORMAL_SINUS 0.92 (861/789), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.91 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.68 (25/25), PVC 0.91 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.68 (30/28) |
| `vfdb` | 63 | 0.889 | 0.783 | VENTRICULAR_TACHYCARDIA 0.63 (13/2), VENTRICULAR_FIBRILLATION 0.93 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.830 | 0.775 | NORMAL_SINUS 0.85 (758/11), PAC 0.57 (117/9), PVC 0.96 (788/11), LBBB 0.98 (50/1), RBBB 0.52 (50/1) |
| `beat_run` | 86 | 0.186 | 0.314 | VENTRICULAR_TACHYCARDIA 0.31 (86/1) |
| `rate_derived` | 488 | 0.842 | 0.936 | SINUS_BRADYCARDIA 0.88 (342/7), SINUS_TACHYCARDIA 1.00 (146/6) |
| `record_level` | 1233 | 0.878 | 0.814 | NORMAL_SINUS 0.92 (861/789), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.91 (46/44), ATRIAL_FIBRILLATION 0.93 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.68 (25/25), PVC 0.91 (50/50), LBBB 0.95 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.68 (30/28) |
| `rhythm_annotation` | 476 | 0.798 | 0.560 | ATRIAL_FIBRILLATION 0.90 (310/7), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.42 (25/6), VENTRICULAR_FIBRILLATION 0.92 (91/6) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 364 | 0.772 | 0.578 | ATRIAL_FIBRILLATION 0.87 (210/5), ATRIAL_FLUTTER 0.00 (50/1), VENTRICULAR_TACHYCARDIA 0.52 (13/2), VENTRICULAR_FIBRILLATION 0.92 (91/6) |
| `0100001` | 937 | 0.812 | 0.737 | NORMAL_SINUS 0.78 (300/6), SINUS_BRADYCARDIA 0.76 (145/4), SINUS_TACHYCARDIA 0.38 (10/2), ATRIAL_FIBRILLATION 0.94 (100/2), PAC 0.60 (67/5), PVC 0.93 (203/6), VENTRICULAR_TACHYCARDIA 0.27 (12/4), LBBB 0.98 (50/1), RBBB 1.00 (50/1) |
| `1111111` | 2745 | 0.842 | 0.721 | NORMAL_SINUS 0.89 (1319/794), SINUS_BRADYCARDIA 0.76 (251/55), SINUS_TACHYCARDIA 0.89 (182/48), ATRIAL_FIBRILLATION 0.88 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.59 (75/29), PVC 0.90 (635/55), VENTRICULAR_TACHYCARDIA 0.31 (86/1), LBBB 0.95 (31/25), RBBB 0.28 (21/19), AV_BLOCK_1 0.68 (30/28) |

## leadconv_0100001

Events 2745 · accuracy **0.812** · macro-F1 **0.706** (subject-bootstrap 95 % CI 0.640–0.823) · macro recall 0.773 · macro specificity 0.979 · macro AUROC 0.963

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.924 | 0.819 | 0.938 | 0.868 | 0.927 | 1319 | 794 |
| SINUS_BRADYCARDIA | 0.640 | 0.757 | 0.957 | 0.693 | 0.968 | 251 | 55 |
| SINUS_TACHYCARDIA | 0.772 | 0.984 | 0.979 | 0.865 | 0.996 | 182 | 48 |
| ATRIAL_FIBRILLATION | 0.837 | 0.920 | 0.992 | 0.877 | 0.984 | 112 | 96 |
| ATRIAL_FLUTTER ⚠ | 1.000 | 0.667 | 1.000 | 0.800 | 1.000 | 3 | 3 |
| PAC | 0.484 | 0.587 | 0.982 | 0.530 | 0.824 | 75 | 29 |
| PVC | 0.876 | 0.844 | 0.964 | 0.860 | 0.957 | 635 | 55 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.909 | 0.233 | 0.999 | 0.370 | 0.987 | 86 | 1 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 1.000 | 0.000 | — | 0 | 0 |
| LBBB | 0.909 | 0.968 | 0.999 | 0.937 | 0.999 | 31 | 25 |
| RBBB | 0.149 | 0.857 | 0.962 | 0.254 | 0.959 | 21 | 19 |
| AV_BLOCK_1 | 0.605 | 0.867 | 0.994 | 0.712 | 0.997 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.773 | 0.698 | NORMAL_SINUS 0.80 (458/5), SINUS_BRADYCARDIA 0.82 (197/3), SINUS_TACHYCARDIA 0.86 (136/4), PAC 0.49 (50/4), PVC 0.86 (585/5), VENTRICULAR_TACHYCARDIA 0.37 (86/1) |
| `ptbxl` | 1233 | 0.859 | 0.798 | NORMAL_SINUS 0.91 (861/789), SINUS_BRADYCARDIA 0.45 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.92 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), PVC 0.90 (50/50), LBBB 0.94 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.71 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1093 | 0.793 | 0.746 | NORMAL_SINUS 0.84 (458/5), PAC 0.49 (50/4), PVC 0.91 (585/5) |
| `beat_run` | 86 | 0.233 | 0.377 | VENTRICULAR_TACHYCARDIA 0.38 (86/1) |
| `rate_derived` | 333 | 0.847 | 0.925 | SINUS_BRADYCARDIA 0.85 (197/3), SINUS_TACHYCARDIA 1.00 (136/4) |
| `record_level` | 1233 | 0.859 | 0.798 | NORMAL_SINUS 0.91 (861/789), SINUS_BRADYCARDIA 0.45 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.92 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), PVC 0.90 (50/50), LBBB 0.94 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.71 (30/28) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2745 | 0.812 | 0.706 | NORMAL_SINUS 0.87 (1319/794), SINUS_BRADYCARDIA 0.69 (251/55), SINUS_TACHYCARDIA 0.86 (182/48), ATRIAL_FIBRILLATION 0.88 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.53 (75/29), PVC 0.86 (635/55), VENTRICULAR_TACHYCARDIA 0.37 (86/1), LBBB 0.94 (31/25), RBBB 0.25 (21/19), AV_BLOCK_1 0.71 (30/28) |

## leadconv_0100000

Events 3682 · accuracy **0.784** · macro-F1 **0.653** (subject-bootstrap 95 % CI 0.560–0.730) · macro recall 0.719 · macro specificity 0.975 · macro AUROC 0.964

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.873 | 0.804 | 0.908 | 0.837 | 0.925 | 1619 | 800 |
| SINUS_BRADYCARDIA | 0.666 | 0.629 | 0.962 | 0.647 | 0.966 | 396 | 59 |
| SINUS_TACHYCARDIA | 0.681 | 0.979 | 0.975 | 0.803 | 0.995 | 192 | 50 |
| ATRIAL_FIBRILLATION | 0.816 | 0.901 | 0.988 | 0.857 | 0.989 | 212 | 98 |
| ATRIAL_FLUTTER ⚠ | 0.667 | 0.667 | 1.000 | 0.667 | 1.000 | 3 | 3 |
| PAC | 0.543 | 0.535 | 0.982 | 0.539 | 0.845 | 142 | 34 |
| PVC | 0.868 | 0.858 | 0.962 | 0.863 | 0.955 | 838 | 61 |
| VENTRICULAR_TACHYCARDIA | 0.750 | 0.122 | 0.999 | 0.211 | 0.978 | 98 | 5 |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 1.000 | 0.000 | — | 0 | 0 |
| LBBB | 0.884 | 0.753 | 0.998 | 0.813 | 0.994 | 81 | 26 |
| RBBB | 0.355 | 0.859 | 0.969 | 0.502 | 0.972 | 71 | 20 |
| AV_BLOCK_1 | 0.308 | 0.800 | 0.985 | 0.444 | 0.988 | 30 | 28 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1512 | 0.755 | 0.653 | NORMAL_SINUS 0.79 (458/5), SINUS_BRADYCARDIA 0.79 (197/3), SINUS_TACHYCARDIA 0.85 (136/4), PAC 0.46 (50/4), PVC 0.84 (585/5), VENTRICULAR_TACHYCARDIA 0.19 (86/1) |
| `mitbih` | 937 | 0.755 | 0.693 | NORMAL_SINUS 0.73 (300/6), SINUS_BRADYCARDIA 0.60 (145/4), SINUS_TACHYCARDIA 0.37 (10/2), ATRIAL_FIBRILLATION 0.91 (100/2), PAC 0.62 (67/5), PVC 0.92 (203/6), VENTRICULAR_TACHYCARDIA 0.38 (12/4), LBBB 0.77 (50/1), RBBB 0.93 (50/1) |
| `ptbxl` | 1233 | 0.841 | 0.725 | NORMAL_SINUS 0.91 (861/789), SINUS_BRADYCARDIA 0.43 (54/52), SINUS_TACHYCARDIA 0.90 (46/44), ATRIAL_FIBRILLATION 0.90 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.52 (25/25), PVC 0.86 (50/50), LBBB 0.92 (31/25), RBBB 0.48 (21/19), AV_BLOCK_1 0.69 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1763 | 0.791 | 0.709 | NORMAL_SINUS 0.84 (758/11), PAC 0.55 (117/9), PVC 0.92 (788/11), LBBB 0.74 (50/1), RBBB 0.51 (50/1) |
| `beat_run` | 86 | 0.105 | 0.189 | VENTRICULAR_TACHYCARDIA 0.19 (86/1) |
| `rate_derived` | 488 | 0.723 | 0.876 | SINUS_BRADYCARDIA 0.76 (342/7), SINUS_TACHYCARDIA 0.99 (146/6) |
| `record_level` | 1233 | 0.841 | 0.725 | NORMAL_SINUS 0.91 (861/789), SINUS_BRADYCARDIA 0.43 (54/52), SINUS_TACHYCARDIA 0.90 (46/44), ATRIAL_FIBRILLATION 0.90 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.52 (25/25), PVC 0.86 (50/50), LBBB 0.92 (31/25), RBBB 0.48 (21/19), AV_BLOCK_1 0.69 (30/28) |
| `rhythm_annotation` | 112 | 0.812 | 0.646 | ATRIAL_FIBRILLATION 0.92 (100/2), VENTRICULAR_TACHYCARDIA 0.38 (12/4) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 937 | 0.755 | 0.693 | NORMAL_SINUS 0.73 (300/6), SINUS_BRADYCARDIA 0.60 (145/4), SINUS_TACHYCARDIA 0.37 (10/2), ATRIAL_FIBRILLATION 0.91 (100/2), PAC 0.62 (67/5), PVC 0.92 (203/6), VENTRICULAR_TACHYCARDIA 0.38 (12/4), LBBB 0.77 (50/1), RBBB 0.93 (50/1) |
| `1111111` | 2745 | 0.793 | 0.645 | NORMAL_SINUS 0.86 (1319/794), SINUS_BRADYCARDIA 0.67 (251/55), SINUS_TACHYCARDIA 0.86 (182/48), ATRIAL_FIBRILLATION 0.82 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.48 (75/29), PVC 0.84 (635/55), VENTRICULAR_TACHYCARDIA 0.18 (86/1), LBBB 0.87 (31/25), RBBB 0.16 (21/19), AV_BLOCK_1 0.69 (30/28) |
