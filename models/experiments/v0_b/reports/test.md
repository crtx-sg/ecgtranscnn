# v0 test — v0_b

## crop2000

Events 1694 · accuracy **0.636** · macro-F1 **0.429** (subject-bootstrap 95 % CI 0.279–0.622) · macro recall 0.488 · macro specificity 0.947 · macro AUROC 0.819

Fewer than 5 subjects (indicative only): SINUS_BRADYCARDIA, SINUS_TACHYCARDIA, ATRIAL_FIBRILLATION, VENTRICULAR_TACHYCARDIA, LBBB, RBBB

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.668 | 0.729 | 0.841 | 0.698 | 0.868 | 517 | 7 |
| SINUS_BRADYCARDIA ⚠ | 0.804 | 0.708 | 0.979 | 0.753 | 0.977 | 185 | 4 |
| SINUS_TACHYCARDIA ⚠ | 0.641 | 0.990 | 0.965 | 0.778 | 0.996 | 101 | 2 |
| ATRIAL_FIBRILLATION ⚠ | 0.000 | 0.000 | 0.994 | 0.000 | 0.596 | 100 | 1 |
| PAC | 0.172 | 0.426 | 0.923 | 0.245 | 0.809 | 61 | 7 |
| PVC | 0.687 | 0.763 | 0.833 | 0.723 | 0.852 | 549 | 8 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.585 | 0.774 | 0.990 | 0.667 | 0.953 | 31 | 2 |
| LBBB ⚠ | 0.000 | 0.000 | 1.000 | 0.000 | 0.368 | 50 | 1 |
| RBBB ⚠ | 0.000 | 0.000 | 1.000 | 0.000 | 0.949 | 100 | 1 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1281 | 0.614 | 0.475 | NORMAL_SINUS 0.63 (317/3), SINUS_BRADYCARDIA 0.67 (103/2), SINUS_TACHYCARDIA 0.78 (101/2), ATRIAL_FIBRILLATION 0.00 (100/1), PAC 0.29 (54/3), PVC 0.73 (477/5), VENTRICULAR_TACHYCARDIA 0.70 (29/1), RBBB 0.00 (100/1) |
| `mitbih` | 413 | 0.702 | 0.395 | NORMAL_SINUS 0.80 (200/4), SINUS_BRADYCARDIA 0.82 (82/2), PAC 0.05 (7/4), PVC 0.70 (72/3), VENTRICULAR_TACHYCARDIA 0.00 (2/1), LBBB 0.00 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1277 | 0.644 | 0.357 | NORMAL_SINUS 0.75 (517/7), PAC 0.30 (61/7), PVC 0.73 (549/8), LBBB 0.00 (50/1), RBBB 0.00 (100/1) |
| `beat_run` | 29 | 0.828 | 0.906 | VENTRICULAR_TACHYCARDIA 0.91 (29/1) |
| `rate_derived` | 286 | 0.808 | 0.912 | SINUS_BRADYCARDIA 0.83 (185/4), SINUS_TACHYCARDIA 1.00 (101/2) |
| `rhythm_annotation` | 102 | 0.000 | 0.000 | ATRIAL_FIBRILLATION 0.00 (100/1), VENTRICULAR_TACHYCARDIA 0.00 (2/1) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 413 | 0.702 | 0.395 | NORMAL_SINUS 0.80 (200/4), SINUS_BRADYCARDIA 0.82 (82/2), PAC 0.05 (7/4), PVC 0.70 (72/3), VENTRICULAR_TACHYCARDIA 0.00 (2/1), LBBB 0.00 (50/1) |
| `1111111` | 1281 | 0.614 | 0.475 | NORMAL_SINUS 0.63 (317/3), SINUS_BRADYCARDIA 0.67 (103/2), SINUS_TACHYCARDIA 0.78 (101/2), ATRIAL_FIBRILLATION 0.00 (100/1), PAC 0.29 (54/3), PVC 0.73 (477/5), VENTRICULAR_TACHYCARDIA 0.70 (29/1), RBBB 0.00 (100/1) |

## full

Events 1694 · accuracy **0.664** · macro-F1 **0.439** (subject-bootstrap 95 % CI 0.278–0.648) · macro recall 0.488 · macro specificity 0.951 · macro AUROC 0.818

Fewer than 5 subjects (indicative only): SINUS_BRADYCARDIA, SINUS_TACHYCARDIA, ATRIAL_FIBRILLATION, VENTRICULAR_TACHYCARDIA, LBBB, RBBB

| Class | Prec | Recall | Spec | F1 | AUROC | Events | Subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL_SINUS | 0.702 | 0.747 | 0.861 | 0.724 | 0.892 | 517 | 7 |
| SINUS_BRADYCARDIA ⚠ | 0.800 | 0.670 | 0.979 | 0.729 | 0.975 | 185 | 4 |
| SINUS_TACHYCARDIA ⚠ | 0.680 | 0.990 | 0.970 | 0.806 | 0.997 | 101 | 2 |
| ATRIAL_FIBRILLATION ⚠ | 0.000 | 0.000 | 0.994 | 0.000 | 0.609 | 100 | 1 |
| PAC | 0.195 | 0.361 | 0.944 | 0.253 | 0.821 | 61 | 7 |
| PVC | 0.692 | 0.852 | 0.818 | 0.764 | 0.891 | 549 | 8 |
| VENTRICULAR_TACHYCARDIA ⚠ | 0.600 | 0.774 | 0.990 | 0.676 | 0.984 | 31 | 2 |
| LBBB ⚠ | 0.000 | 0.000 | 1.000 | 0.000 | 0.244 | 50 | 1 |
| RBBB ⚠ | 0.000 | 0.000 | 0.997 | 0.000 | 0.946 | 100 | 1 |

### By dataset

| Dataset | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `incart` | 1281 | 0.642 | 0.481 | NORMAL_SINUS 0.67 (317/3), SINUS_BRADYCARDIA 0.63 (103/2), SINUS_TACHYCARDIA 0.81 (101/2), ATRIAL_FIBRILLATION 0.00 (100/1), PAC 0.29 (54/3), PVC 0.76 (477/5), VENTRICULAR_TACHYCARDIA 0.70 (29/1), RBBB 0.00 (100/1) |
| `mitbih` | 413 | 0.729 | 0.424 | NORMAL_SINUS 0.81 (200/4), SINUS_BRADYCARDIA 0.82 (82/2), PAC 0.12 (7/4), PVC 0.80 (72/3), VENTRICULAR_TACHYCARDIA 0.00 (2/1), LBBB 0.00 (50/1) |

### By label method

| Label method | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `beat_morphology` | 1277 | 0.686 | 0.376 | NORMAL_SINUS 0.78 (517/7), PAC 0.32 (61/7), PVC 0.77 (549/8), LBBB 0.00 (50/1), RBBB 0.00 (100/1) |
| `beat_run` | 29 | 0.828 | 0.906 | VENTRICULAR_TACHYCARDIA 0.91 (29/1) |
| `rate_derived` | 286 | 0.783 | 0.899 | SINUS_BRADYCARDIA 0.80 (185/4), SINUS_TACHYCARDIA 1.00 (101/2) |
| `rhythm_annotation` | 102 | 0.000 | 0.000 | ATRIAL_FIBRILLATION 0.00 (100/1), VENTRICULAR_TACHYCARDIA 0.00 (2/1) |

### By real-lead mask

| Real-lead mask | Events | Accuracy | Macro-F1 | Per-class F1 (events/subjects) |
|---|---:|---:|---:|---|
| `0100001` | 413 | 0.729 | 0.424 | NORMAL_SINUS 0.81 (200/4), SINUS_BRADYCARDIA 0.82 (82/2), PAC 0.12 (7/4), PVC 0.80 (72/3), VENTRICULAR_TACHYCARDIA 0.00 (2/1), LBBB 0.00 (50/1) |
| `1111111` | 1281 | 0.642 | 0.481 | NORMAL_SINUS 0.67 (317/3), SINUS_BRADYCARDIA 0.63 (103/2), SINUS_TACHYCARDIA 0.81 (101/2), ATRIAL_FIBRILLATION 0.00 (100/1), PAC 0.29 (54/3), PVC 0.76 (477/5), VENTRICULAR_TACHYCARDIA 0.70 (29/1), RBBB 0.00 (100/1) |
