# v2 test — v2_cv_fold0, v2_cv_fold1, v2_cv_fold2, v2_cv_fold3, v2_cv_fold4

## crop2000

**Primary metric — macro-F1 over 12 classes: 0.587** (subject-bootstrap 95 % CI 0.506–0.684)

Excludes VENTRICULAR_FIBRILLATION (no_seven_real_lead_events, test_record_dominates). Scored below, but on fabricated-lead data only — not validated in the deployment configuration.

Events 3513 · accuracy **0.782** · macro-F1 (all classes) **0.606** (subject-bootstrap 95 % CI 0.528–0.696) · macro recall 0.653 · macro specificity 0.980 · macro AUROC 0.942

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects | Package flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NORMAL_SINUS | 0.924 | 0.827 | 0.955 | 0.873 | 0.963 | 1400 | 653 | — |
| SINUS_BRADYCARDIA | 0.676 | 0.824 | 0.970 | 0.743 | 0.979 | 245 | 58 | — |
| SINUS_TACHYCARDIA | 0.771 | 0.917 | 0.989 | 0.837 | 0.996 | 132 | 48 | — |
| ATRIAL_FIBRILLATION | 0.699 | 0.818 | 0.951 | 0.753 | 0.970 | 428 | 104 | — |
| ATRIAL_FLUTTER | 0.231 | 0.062 | 0.997 | 0.098 | 0.960 | 48 | 6 | few_test_subjects, few_val_subjects, test_record_dominates |
| PAC | 0.514 | 0.594 | 0.967 | 0.551 | 0.898 | 192 | 31 | — |
| SVT | 0.100 | 0.080 | 0.995 | 0.089 | 0.632 | 25 | 7 | few_test_subjects, few_val_subjects |
| PVC | 0.865 | 0.804 | 0.969 | 0.833 | 0.960 | 700 | 60 | — |
| VENTRICULAR_TACHYCARDIA | 0.387 | 0.462 | 0.983 | 0.421 | 0.945 | 78 | 9 | few_test_subjects, few_val_subjects, test_record_dominates |
| VENTRICULAR_FIBRILLATION | 0.826 | 0.855 | 0.996 | 0.840 | 0.998 | 83 | 10 | no_seven_real_lead_events, test_record_dominates |
| LBBB | 0.938 | 0.370 | 0.999 | 0.531 | 0.964 | 81 | 26 | real_lead_skew, label_method_skew, test_record_dominates |
| RBBB | 0.873 | 0.972 | 0.997 | 0.920 | 0.997 | 71 | 20 | real_lead_skew, test_record_dominates |
| AV_BLOCK_1 | 0.252 | 0.900 | 0.977 | 0.394 | 0.990 | 30 | 28 | — |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 210 | 0.781 | 0.520 | ATRIAL_FIBRILLATION 0.89 (200/4), ATRIAL_FLUTTER 0.15 (10/2) |
| `cudb` | 23 | 0.826 | 0.905 | VENTRICULAR_FIBRILLATION 0.90 (23/7) |
| `incart` | 1365 | 0.776 | 0.746 | NORMAL_SINUS 0.79 (500/6), SINUS_BRADYCARDIA 0.79 (145/3), SINUS_TACHYCARDIA 0.82 (86/4), ATRIAL_FIBRILLATION 0.84 (21/1), PAC 0.57 (116/3), PVC 0.85 (480/6), VENTRICULAR_TACHYCARDIA 0.56 (17/2) |
| `mitbih` | 728 | 0.659 | 0.484 | NORMAL_SINUS 0.87 (200/4), SINUS_BRADYCARDIA 0.92 (46/3), ATRIAL_FIBRILLATION 0.46 (95/3), ATRIAL_FLUTTER 0.00 (35/1), PAC 0.55 (51/3), SVT 0.00 (11/2), PVC 0.77 (170/4), VENTRICULAR_TACHYCARDIA 0.14 (8/4), VENTRICULAR_FIBRILLATION 0.67 (12/1), LBBB 0.00 (50/1), RBBB 0.95 (50/1) |
| `ptbxl` | 1076 | 0.889 | 0.794 | NORMAL_SINUS 0.93 (700/643), SINUS_BRADYCARDIA 0.56 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.95 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.70 (25/25), SVT 0.57 (4/4), PVC 0.93 (50/50), LBBB 0.95 (31/25), RBBB 0.84 (21/19), AV_BLOCK_1 0.76 (30/28) |
| `vfdb` | 111 | 0.604 | 0.456 | SVT 0.00 (10/1), VENTRICULAR_TACHYCARDIA 0.53 (53/3), VENTRICULAR_FIBRILLATION 0.84 (48/2) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1617 | 0.739 | 0.644 | NORMAL_SINUS 0.83 (700/10), PAC 0.58 (167/6), PVC 0.85 (650/10), LBBB 0.00 (50/1), RBBB 0.95 (50/1) |
| `beat_run` | 22 | 0.545 | 0.414 | SVT 0.00 (5/1), VENTRICULAR_TACHYCARDIA 0.83 (17/2) |
| `rate_derived` | 277 | 0.866 | 0.938 | SINUS_BRADYCARDIA 0.91 (191/6), SINUS_TACHYCARDIA 0.96 (86/4) |
| `record_level` | 1076 | 0.889 | 0.794 | NORMAL_SINUS 0.93 (700/643), SINUS_BRADYCARDIA 0.56 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.95 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.70 (25/25), SVT 0.57 (4/4), PVC 0.93 (50/50), LBBB 0.95 (31/25), RBBB 0.84 (21/19), AV_BLOCK_1 0.76 (30/28) |
| `rhythm_annotation` | 521 | 0.656 | 0.419 | ATRIAL_FIBRILLATION 0.79 (316/8), ATRIAL_FLUTTER 0.04 (45/3), SVT 0.00 (16/2), VENTRICULAR_TACHYCARDIA 0.42 (61/7), VENTRICULAR_FIBRILLATION 0.84 (83/10) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 344 | 0.727 | 0.464 | ATRIAL_FIBRILLATION 0.86 (200/4), ATRIAL_FLUTTER 0.11 (10/2), SVT 0.00 (10/1), VENTRICULAR_TACHYCARDIA 0.50 (53/3), VENTRICULAR_FIBRILLATION 0.86 (71/9) |
| `0100001` | 728 | 0.659 | 0.484 | NORMAL_SINUS 0.87 (200/4), SINUS_BRADYCARDIA 0.92 (46/3), ATRIAL_FIBRILLATION 0.46 (95/3), ATRIAL_FLUTTER 0.00 (35/1), PAC 0.55 (51/3), SVT 0.00 (11/2), PVC 0.77 (170/4), VENTRICULAR_TACHYCARDIA 0.14 (8/4), VENTRICULAR_FIBRILLATION 0.67 (12/1), LBBB 0.00 (50/1), RBBB 0.95 (50/1) |
| `1111111` | 2441 | 0.826 | 0.715 | NORMAL_SINUS 0.87 (1200/649), SINUS_BRADYCARDIA 0.71 (199/55), SINUS_TACHYCARDIA 0.84 (132/48), ATRIAL_FIBRILLATION 0.93 (133/97), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.60 (141/28), SVT 0.27 (4/4), PVC 0.86 (530/56), VENTRICULAR_TACHYCARDIA 0.56 (17/2), LBBB 0.95 (31/25), RBBB 0.84 (21/19), AV_BLOCK_1 0.48 (30/28) |

### By pacing

| Pacing | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `paced` | 86 | 0.616 | 0.366 | ATRIAL_FIBRILLATION 0.32 (35/1), PVC 0.78 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (1/1) |
| `unpaced` | 3427 | 0.786 | 0.611 | NORMAL_SINUS 0.87 (1400/653), SINUS_BRADYCARDIA 0.74 (245/58), SINUS_TACHYCARDIA 0.84 (132/48), ATRIAL_FIBRILLATION 0.78 (393/103), ATRIAL_FLUTTER 0.10 (48/6), PAC 0.55 (192/31), SVT 0.09 (25/7), PVC 0.84 (650/59), VENTRICULAR_TACHYCARDIA 0.45 (77/8), VENTRICULAR_FIBRILLATION 0.84 (83/10), LBBB 0.53 (81/26), RBBB 0.92 (71/20), AV_BLOCK_1 0.39 (30/28) |

## full

**Primary metric — macro-F1 over 12 classes: 0.593** (subject-bootstrap 95 % CI 0.512–0.691)

Excludes VENTRICULAR_FIBRILLATION (no_seven_real_lead_events, test_record_dominates). Scored below, but on fabricated-lead data only — not validated in the deployment configuration.

Events 3513 · accuracy **0.790** · macro-F1 (all classes) **0.611** (subject-bootstrap 95 % CI 0.533–0.703) · macro recall 0.660 · macro specificity 0.981 · macro AUROC 0.950

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects | Package flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NORMAL_SINUS | 0.939 | 0.818 | 0.965 | 0.874 | 0.968 | 1400 | 653 | — |
| SINUS_BRADYCARDIA | 0.694 | 0.816 | 0.973 | 0.750 | 0.981 | 245 | 58 | — |
| SINUS_TACHYCARDIA | 0.804 | 0.902 | 0.991 | 0.850 | 0.995 | 132 | 48 | — |
| ATRIAL_FIBRILLATION | 0.717 | 0.806 | 0.956 | 0.759 | 0.972 | 428 | 104 | — |
| ATRIAL_FLUTTER | 0.214 | 0.062 | 0.997 | 0.097 | 0.967 | 48 | 6 | few_test_subjects, few_val_subjects, test_record_dominates |
| PAC | 0.512 | 0.667 | 0.963 | 0.579 | 0.921 | 192 | 31 | — |
| SVT | 0.118 | 0.080 | 0.996 | 0.095 | 0.671 | 25 | 7 | few_test_subjects, few_val_subjects |
| PVC | 0.858 | 0.854 | 0.965 | 0.856 | 0.973 | 700 | 60 | — |
| VENTRICULAR_TACHYCARDIA | 0.407 | 0.449 | 0.985 | 0.427 | 0.944 | 78 | 9 | few_test_subjects, few_val_subjects, test_record_dominates |
| VENTRICULAR_FIBRILLATION | 0.785 | 0.880 | 0.994 | 0.830 | 0.997 | 83 | 10 | no_seven_real_lead_events, test_record_dominates |
| LBBB | 0.938 | 0.370 | 0.999 | 0.531 | 0.971 | 81 | 26 | real_lead_skew, label_method_skew, test_record_dominates |
| RBBB | 0.831 | 0.972 | 0.996 | 0.896 | 0.997 | 71 | 20 | real_lead_skew, test_record_dominates |
| AV_BLOCK_1 | 0.257 | 0.900 | 0.978 | 0.400 | 0.990 | 30 | 28 | — |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `afdb` | 210 | 0.776 | 0.517 | ATRIAL_FIBRILLATION 0.88 (200/4), ATRIAL_FLUTTER 0.15 (10/2) |
| `cudb` | 23 | 0.913 | 0.955 | VENTRICULAR_FIBRILLATION 0.95 (23/7) |
| `incart` | 1365 | 0.792 | 0.764 | NORMAL_SINUS 0.79 (500/6), SINUS_BRADYCARDIA 0.81 (145/3), SINUS_TACHYCARDIA 0.84 (86/4), ATRIAL_FIBRILLATION 0.84 (21/1), PAC 0.59 (116/3), PVC 0.88 (480/6), VENTRICULAR_TACHYCARDIA 0.60 (17/2) |
| `mitbih` | 728 | 0.670 | 0.497 | NORMAL_SINUS 0.88 (200/4), SINUS_BRADYCARDIA 0.89 (46/3), ATRIAL_FIBRILLATION 0.46 (95/3), ATRIAL_FLUTTER 0.00 (35/1), PAC 0.65 (51/3), SVT 0.00 (11/2), PVC 0.77 (170/4), VENTRICULAR_TACHYCARDIA 0.22 (8/4), VENTRICULAR_FIBRILLATION 0.67 (12/1), LBBB 0.00 (50/1), RBBB 0.93 (50/1) |
| `ptbxl` | 1076 | 0.889 | 0.794 | NORMAL_SINUS 0.93 (700/643), SINUS_BRADYCARDIA 0.56 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.95 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.70 (25/25), SVT 0.57 (4/4), PVC 0.93 (50/50), LBBB 0.95 (31/25), RBBB 0.84 (21/19), AV_BLOCK_1 0.76 (30/28) |
| `vfdb` | 111 | 0.577 | 0.427 | SVT 0.00 (10/1), VENTRICULAR_TACHYCARDIA 0.47 (53/3), VENTRICULAR_FIBRILLATION 0.81 (48/2) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1617 | 0.761 | 0.651 | NORMAL_SINUS 0.83 (700/10), PAC 0.62 (167/6), PVC 0.88 (650/10), LBBB 0.00 (50/1), RBBB 0.92 (50/1) |
| `beat_run` | 22 | 0.545 | 0.414 | SVT 0.00 (5/1), VENTRICULAR_TACHYCARDIA 0.83 (17/2) |
| `rate_derived` | 277 | 0.852 | 0.928 | SINUS_BRADYCARDIA 0.91 (191/6), SINUS_TACHYCARDIA 0.95 (86/4) |
| `record_level` | 1076 | 0.889 | 0.794 | NORMAL_SINUS 0.93 (700/643), SINUS_BRADYCARDIA 0.56 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.95 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.70 (25/25), SVT 0.57 (4/4), PVC 0.93 (50/50), LBBB 0.95 (31/25), RBBB 0.84 (21/19), AV_BLOCK_1 0.76 (30/28) |
| `rhythm_annotation` | 521 | 0.649 | 0.415 | ATRIAL_FIBRILLATION 0.79 (316/8), ATRIAL_FLUTTER 0.04 (45/3), SVT 0.00 (16/2), VENTRICULAR_TACHYCARDIA 0.42 (61/7), VENTRICULAR_FIBRILLATION 0.83 (83/10) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100000` | 344 | 0.721 | 0.454 | ATRIAL_FIBRILLATION 0.85 (200/4), ATRIAL_FLUTTER 0.11 (10/2), SVT 0.00 (10/1), VENTRICULAR_TACHYCARDIA 0.46 (53/3), VENTRICULAR_FIBRILLATION 0.85 (71/9) |
| `0100001` | 728 | 0.670 | 0.497 | NORMAL_SINUS 0.88 (200/4), SINUS_BRADYCARDIA 0.89 (46/3), ATRIAL_FIBRILLATION 0.46 (95/3), ATRIAL_FLUTTER 0.00 (35/1), PAC 0.65 (51/3), SVT 0.00 (11/2), PVC 0.77 (170/4), VENTRICULAR_TACHYCARDIA 0.22 (8/4), VENTRICULAR_FIBRILLATION 0.67 (12/1), LBBB 0.00 (50/1), RBBB 0.93 (50/1) |
| `1111111` | 2441 | 0.835 | 0.730 | NORMAL_SINUS 0.87 (1200/649), SINUS_BRADYCARDIA 0.72 (199/55), SINUS_TACHYCARDIA 0.85 (132/48), ATRIAL_FIBRILLATION 0.93 (133/97), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.61 (141/28), SVT 0.36 (4/4), PVC 0.89 (530/56), VENTRICULAR_TACHYCARDIA 0.60 (17/2), LBBB 0.95 (31/25), RBBB 0.83 (21/19), AV_BLOCK_1 0.48 (30/28) |

### By pacing

| Pacing | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `paced` | 86 | 0.593 | 0.313 | ATRIAL_FIBRILLATION 0.16 (35/1), PVC 0.78 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (1/1) |
| `unpaced` | 3427 | 0.795 | 0.616 | NORMAL_SINUS 0.87 (1400/653), SINUS_BRADYCARDIA 0.75 (245/58), SINUS_TACHYCARDIA 0.85 (132/48), ATRIAL_FIBRILLATION 0.79 (393/103), ATRIAL_FLUTTER 0.10 (48/6), PAC 0.58 (192/31), SVT 0.10 (25/7), PVC 0.86 (650/59), VENTRICULAR_TACHYCARDIA 0.46 (77/8), VENTRICULAR_FIBRILLATION 0.83 (83/10), LBBB 0.53 (81/26), RBBB 0.90 (71/20), AV_BLOCK_1 0.40 (30/28) |

## leadconv_0100001

**Primary metric — macro-F1 over 12 classes: 0.676** (subject-bootstrap 95 % CI 0.594–0.780)

Excludes VENTRICULAR_FIBRILLATION (no_seven_real_lead_events, test_record_dominates). Scored below, but on fabricated-lead data only — not validated in the deployment configuration.

Events 2441 · accuracy **0.805** · macro-F1 (all classes) **0.676** (subject-bootstrap 95 % CI 0.594–0.780) · macro recall 0.777 · macro specificity 0.981 · macro AUROC 0.973

Classes absent from this split: VENTRICULAR_FIBRILLATION

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, SVT, VENTRICULAR_TACHYCARDIA

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects | Package flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NORMAL_SINUS | 0.935 | 0.779 | 0.948 | 0.850 | 0.950 | 1200 | 649 | — |
| SINUS_BRADYCARDIA | 0.580 | 0.874 | 0.944 | 0.697 | 0.970 | 199 | 55 | — |
| SINUS_TACHYCARDIA | 0.761 | 0.917 | 0.984 | 0.832 | 0.992 | 132 | 48 | — |
| ATRIAL_FIBRILLATION | 0.908 | 0.895 | 0.995 | 0.902 | 0.984 | 133 | 97 | — |
| ATRIAL_FLUTTER ⚠ | 0.667 | 0.667 | 1.000 | 0.667 | 0.996 | 3 | 3 | few_test_subjects, few_val_subjects, test_record_dominates |
| PAC | 0.554 | 0.688 | 0.966 | 0.614 | 0.905 | 141 | 28 | — |
| SVT ⚠ | 0.250 | 0.500 | 0.998 | 0.333 | 0.982 | 4 | 4 | few_test_subjects, few_val_subjects |
| PVC | 0.927 | 0.815 | 0.982 | 0.867 | 0.958 | 530 | 56 | — |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.231 | 0.529 | 0.988 | 0.321 | 0.987 | 17 | 2 | few_test_subjects, few_val_subjects, test_record_dominates |
| VENTRICULAR_FIBRILLATION | 0.000 | 0.000 | 0.999 | 0.000 | — | 0 | 0 | no_seven_real_lead_events, test_record_dominates |
| LBBB | 0.906 | 0.935 | 0.999 | 0.921 | 0.997 | 31 | 25 | real_lead_skew, label_method_skew, test_record_dominates |
| RBBB | 0.581 | 0.857 | 0.995 | 0.692 | 0.970 | 21 | 19 | real_lead_skew, test_record_dominates |
| AV_BLOCK_1 | 0.277 | 0.867 | 0.972 | 0.419 | 0.988 | 30 | 28 | — |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1365 | 0.763 | 0.709 | NORMAL_SINUS 0.77 (500/6), SINUS_BRADYCARDIA 0.82 (145/3), SINUS_TACHYCARDIA 0.80 (86/4), ATRIAL_FIBRILLATION 0.79 (21/1), PAC 0.60 (116/3), PVC 0.86 (480/6), VENTRICULAR_TACHYCARDIA 0.33 (17/2) |
| `ptbxl` | 1076 | 0.858 | 0.768 | NORMAL_SINUS 0.91 (700/643), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.92 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.69 (25/25), SVT 0.44 (4/4), PVC 0.93 (50/50), LBBB 0.92 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.72 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1096 | 0.736 | 0.753 | NORMAL_SINUS 0.78 (500/6), PAC 0.61 (116/3), PVC 0.87 (480/6) |
| `beat_run` | 17 | 0.529 | 0.692 | VENTRICULAR_TACHYCARDIA 0.69 (17/2) |
| `rate_derived` | 231 | 0.900 | 0.950 | SINUS_BRADYCARDIA 0.94 (145/3), SINUS_TACHYCARDIA 0.96 (86/4) |
| `record_level` | 1076 | 0.858 | 0.768 | NORMAL_SINUS 0.91 (700/643), SINUS_BRADYCARDIA 0.49 (54/52), SINUS_TACHYCARDIA 0.89 (46/44), ATRIAL_FIBRILLATION 0.92 (112/96), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.69 (25/25), SVT 0.44 (4/4), PVC 0.93 (50/50), LBBB 0.92 (31/25), RBBB 0.86 (21/19), AV_BLOCK_1 0.72 (30/28) |
| `rhythm_annotation` | 21 | 0.810 | 0.895 | ATRIAL_FIBRILLATION 0.89 (21/1) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `1111111` | 2441 | 0.805 | 0.676 | NORMAL_SINUS 0.85 (1200/649), SINUS_BRADYCARDIA 0.70 (199/55), SINUS_TACHYCARDIA 0.83 (132/48), ATRIAL_FIBRILLATION 0.90 (133/97), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.61 (141/28), SVT 0.33 (4/4), PVC 0.87 (530/56), VENTRICULAR_TACHYCARDIA 0.32 (17/2), LBBB 0.92 (31/25), RBBB 0.69 (21/19), AV_BLOCK_1 0.42 (30/28) |

### By pacing

| Pacing | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `unpaced` | 2441 | 0.805 | 0.676 | NORMAL_SINUS 0.85 (1200/649), SINUS_BRADYCARDIA 0.70 (199/55), SINUS_TACHYCARDIA 0.83 (132/48), ATRIAL_FIBRILLATION 0.90 (133/97), ATRIAL_FLUTTER 0.67 (3/3), PAC 0.61 (141/28), SVT 0.33 (4/4), PVC 0.87 (530/56), VENTRICULAR_TACHYCARDIA 0.32 (17/2), LBBB 0.92 (31/25), RBBB 0.69 (21/19), AV_BLOCK_1 0.42 (30/28) |

## leadconv_0100000

**Primary metric — macro-F1 over 12 classes: 0.485** (subject-bootstrap 95 % CI 0.416–0.609)

Excludes VENTRICULAR_FIBRILLATION (no_seven_real_lead_events, test_record_dominates). Scored below, but on fabricated-lead data only — not validated in the deployment configuration.

Events 3169 · accuracy **0.701** · macro-F1 (all classes) **0.490** (subject-bootstrap 95 % CI 0.430–0.609) · macro recall 0.566 · macro specificity 0.974 · macro AUROC 0.938

Fewer than 5 subjects (indicative only): ATRIAL_FLUTTER, VENTRICULAR_FIBRILLATION

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects | Package flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NORMAL_SINUS | 0.923 | 0.741 | 0.951 | 0.823 | 0.949 | 1400 | 653 | — |
| SINUS_BRADYCARDIA | 0.594 | 0.722 | 0.959 | 0.652 | 0.970 | 245 | 58 | — |
| SINUS_TACHYCARDIA | 0.756 | 0.894 | 0.987 | 0.819 | 0.991 | 132 | 48 | — |
| ATRIAL_FIBRILLATION | 0.586 | 0.820 | 0.955 | 0.684 | 0.970 | 228 | 100 | — |
| ATRIAL_FLUTTER ⚠ | 0.375 | 0.079 | 0.998 | 0.130 | 0.986 | 38 | 4 | few_test_subjects, few_val_subjects, test_record_dominates |
| PAC | 0.237 | 0.302 | 0.937 | 0.265 | 0.851 | 192 | 31 | — |
| SVT | 0.091 | 0.067 | 0.997 | 0.077 | 0.748 | 15 | 6 | few_test_subjects, few_val_subjects |
| PVC | 0.799 | 0.729 | 0.948 | 0.762 | 0.938 | 700 | 60 | — |
| VENTRICULAR_TACHYCARDIA | 0.113 | 0.360 | 0.977 | 0.171 | 0.909 | 25 | 6 | few_test_subjects, few_val_subjects, test_record_dominates |
| VENTRICULAR_FIBRILLATION ⚠ | 0.471 | 0.667 | 0.997 | 0.552 | 0.999 | 12 | 1 | no_seven_real_lead_events, test_record_dominates |
| LBBB | 0.824 | 0.346 | 0.998 | 0.487 | 0.944 | 81 | 26 | real_lead_skew, label_method_skew, test_record_dominates |
| RBBB | 0.378 | 0.831 | 0.969 | 0.520 | 0.956 | 71 | 20 | real_lead_skew, test_record_dominates |
| AV_BLOCK_1 | 0.289 | 0.800 | 0.981 | 0.425 | 0.988 | 30 | 28 | — |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1365 | 0.612 | 0.568 | NORMAL_SINUS 0.69 (500/6), SINUS_BRADYCARDIA 0.75 (145/3), SINUS_TACHYCARDIA 0.80 (86/4), ATRIAL_FIBRILLATION 0.71 (21/1), PAC 0.11 (116/3), PVC 0.73 (480/6), VENTRICULAR_TACHYCARDIA 0.19 (17/2) |
| `mitbih` | 728 | 0.681 | 0.503 | NORMAL_SINUS 0.87 (200/4), SINUS_BRADYCARDIA 0.76 (46/3), ATRIAL_FIBRILLATION 0.51 (95/3), ATRIAL_FLUTTER 0.05 (35/1), PAC 0.61 (51/3), SVT 0.00 (11/2), PVC 0.82 (170/4), VENTRICULAR_TACHYCARDIA 0.14 (8/4), VENTRICULAR_FIBRILLATION 0.80 (12/1), LBBB 0.00 (50/1), RBBB 0.98 (50/1) |
| `ptbxl` | 1076 | 0.825 | 0.692 | NORMAL_SINUS 0.89 (700/643), SINUS_BRADYCARDIA 0.46 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.89 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), SVT 0.22 (4/4), PVC 0.88 (50/50), LBBB 0.86 (31/25), RBBB 0.45 (21/19), AV_BLOCK_1 0.69 (30/28) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1617 | 0.628 | 0.477 | NORMAL_SINUS 0.77 (700/10), PAC 0.22 (167/6), PVC 0.77 (650/10), LBBB 0.00 (50/1), RBBB 0.63 (50/1) |
| `beat_run` | 22 | 0.273 | 0.261 | SVT 0.00 (5/1), VENTRICULAR_TACHYCARDIA 0.52 (17/2) |
| `rate_derived` | 277 | 0.765 | 0.887 | SINUS_BRADYCARDIA 0.83 (191/6), SINUS_TACHYCARDIA 0.94 (86/4) |
| `record_level` | 1076 | 0.825 | 0.692 | NORMAL_SINUS 0.89 (700/643), SINUS_BRADYCARDIA 0.46 (54/52), SINUS_TACHYCARDIA 0.87 (46/44), ATRIAL_FIBRILLATION 0.89 (112/96), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.60 (25/25), SVT 0.22 (4/4), PVC 0.88 (50/50), LBBB 0.86 (31/25), RBBB 0.45 (21/19), AV_BLOCK_1 0.69 (30/28) |
| `rhythm_annotation` | 177 | 0.554 | 0.364 | ATRIAL_FIBRILLATION 0.72 (116/4), ATRIAL_FLUTTER 0.05 (35/1), SVT 0.00 (6/1), VENTRICULAR_TACHYCARDIA 0.25 (8/4), VENTRICULAR_FIBRILLATION 0.80 (12/1) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 728 | 0.681 | 0.503 | NORMAL_SINUS 0.87 (200/4), SINUS_BRADYCARDIA 0.76 (46/3), ATRIAL_FIBRILLATION 0.51 (95/3), ATRIAL_FLUTTER 0.05 (35/1), PAC 0.61 (51/3), SVT 0.00 (11/2), PVC 0.82 (170/4), VENTRICULAR_TACHYCARDIA 0.14 (8/4), VENTRICULAR_FIBRILLATION 0.80 (12/1), LBBB 0.00 (50/1), RBBB 0.98 (50/1) |
| `1111111` | 2441 | 0.706 | 0.564 | NORMAL_SINUS 0.81 (1200/649), SINUS_BRADYCARDIA 0.64 (199/55), SINUS_TACHYCARDIA 0.83 (132/48), ATRIAL_FIBRILLATION 0.86 (133/97), ATRIAL_FLUTTER 0.80 (3/3), PAC 0.19 (141/28), SVT 0.15 (4/4), PVC 0.74 (530/56), VENTRICULAR_TACHYCARDIA 0.19 (17/2), LBBB 0.86 (31/25), RBBB 0.17 (21/19), AV_BLOCK_1 0.53 (30/28) |

### By pacing

| Pacing | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `paced` | 86 | 0.663 | 0.415 | ATRIAL_FIBRILLATION 0.43 (35/1), PVC 0.81 (50/1), VENTRICULAR_TACHYCARDIA 0.00 (1/1) |
| `unpaced` | 3083 | 0.702 | 0.493 | NORMAL_SINUS 0.82 (1400/653), SINUS_BRADYCARDIA 0.65 (245/58), SINUS_TACHYCARDIA 0.82 (132/48), ATRIAL_FIBRILLATION 0.71 (193/99), ATRIAL_FLUTTER 0.13 (38/4), PAC 0.27 (192/31), SVT 0.08 (15/6), PVC 0.76 (650/59), VENTRICULAR_TACHYCARDIA 0.19 (24/5), VENTRICULAR_FIBRILLATION 0.55 (12/1), LBBB 0.49 (81/26), RBBB 0.52 (71/20), AV_BLOCK_1 0.42 (30/28) |
