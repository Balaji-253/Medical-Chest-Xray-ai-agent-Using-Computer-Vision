# Medical AI Research Evaluation Report

## Model

**ResNet-18 weighted baseline**

The evaluation uses the currently generated test-set metrics and class-level error analysis artifacts.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| Mean Auroc | 0.6408 |
| Mean F1 | 0.1455 |
| Mean Precision | 0.1071 |
| Mean Sensitivity | 0.4308 |
| Mean Specificity | 0.6990 |

## Per-Class Results

| Class | Positive | AUROC | F1 | Precision | Sensitivity | Specificity | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 91 | 0.6257 | 0.2609 | 0.1875 | 0.4286 | 0.7436 | 0.65 |
| Cardiomegaly | 68 | 0.7158 | 0.0000 | 0.0000 | 0.0000 | 0.9985 | 0.60 |
| Effusion | 113 | 0.7147 | 0.3238 | 0.2525 | 0.4513 | 0.7630 | 0.75 |
| Infiltration | 171 | 0.5697 | 0.3562 | 0.2609 | 0.5614 | 0.5302 | 0.60 |
| Mass | 31 | 0.5216 | 0.0773 | 0.0404 | 0.8710 | 0.1085 | 0.20 |
| Nodule | 25 | 0.4860 | 0.0654 | 0.0339 | 0.9600 | 0.0552 | 0.20 |
| Pneumonia | 13 | 0.6036 | 0.0328 | 0.0183 | 0.1538 | 0.8548 | 0.30 |
| Pneumothorax | 89 | 0.6533 | 0.2551 | 0.1562 | 0.6966 | 0.4932 | 0.30 |
| Consolidation | 42 | 0.5903 | 0.1475 | 0.0842 | 0.5952 | 0.6158 | 0.45 |
| Edema | 24 | 0.7904 | 0.1111 | 0.1667 | 0.0833 | 0.9862 | 0.50 |
| Emphysema | 32 | 0.5339 | 0.0000 | 0.0000 | 0.0000 | 0.9777 | 0.55 |
| Fibrosis | 26 | 0.7935 | 0.1717 | 0.0966 | 0.7692 | 0.7417 | 0.40 |
| Pleural_Thickening | 45 | 0.5532 | 0.0597 | 0.0909 | 0.0444 | 0.9716 | 0.55 |
| Hernia | 12 | 0.8189 | 0.1754 | 0.1111 | 0.4167 | 0.9458 | 0.15 |

## Confusion-Matrix Error Analysis

| Class | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Hernia | 5 | 40 | 698 | 7 |
| Fibrosis | 20 | 187 | 537 | 6 |
| Edema | 2 | 10 | 716 | 22 |
| Cardiomegaly | 0 | 1 | 681 | 68 |
| Effusion | 51 | 151 | 486 | 62 |
| Pneumothorax | 62 | 335 | 326 | 27 |
| Atelectasis | 39 | 169 | 490 | 52 |
| Pneumonia | 2 | 107 | 630 | 11 |
| Consolidation | 25 | 272 | 436 | 17 |
| Infiltration | 96 | 272 | 307 | 75 |
| Pleural_Thickening | 2 | 20 | 685 | 43 |
| Emphysema | 0 | 16 | 702 | 32 |
| Mass | 27 | 641 | 78 | 4 |
| Nodule | 24 | 685 | 40 | 1 |

## Research Interpretation

- Mean AUROC across the 14 classes is **0.6408**.
- Mean F1 is **0.1455**, indicating substantial variation between ranking performance and thresholded classification performance.
- The highest AUROC in this evaluation is for **Hernia** (0.8189).
- The lowest AUROC in this evaluation is for **Nodule** (0.4860).
- The highest F1 in this evaluation is for **Infiltration** (0.3562).

## Important Limitations

- The current evaluation is based on the project's local test subset.
- The results should not be interpreted as clinical validation.
- Performance varies considerably across disease classes.
- Several classes have relatively few positive test samples.
- Threshold optimization was performed using the validation split.
- Further robustness, calibration, confidence-interval, and external-validation analysis should be performed before making stronger research claims.