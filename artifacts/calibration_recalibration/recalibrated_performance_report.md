# Recalibrated Model Performance Comparison

## Methodology

Calibration models and class-specific decision thresholds were fitted using the validation split. Final metrics were then calculated on the untouched test prediction set.

## Aggregate Test Results

| Method | AUROC | F1 | Precision | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.6408 | 0.1628 | 0.1299 | 0.3523 | 0.7697 | 0.1619 | 0.2738 |
| Platt | 0.6408 | 0.1215 | 0.1114 | 0.1529 | 0.9183 | 0.0655 | 0.0248 |
| Isotonic | 0.6274 | 0.1624 | 0.1397 | 0.2569 | 0.8657 | 0.0668 | 0.0308 |

## Per-Class Test Results

| Class | Method | AUROC | F1 | Precision | Sensitivity | Specificity | Brier | ECE | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | Original | 0.6257 | 0.2657 | 0.1949 | 0.4176 | 0.7618 | 0.2906 | 0.4111 | 0.66 |
| Atelectasis | Platt | 0.6257 | 0.2612 | 0.1977 | 0.3846 | 0.7845 | 0.1047 | 0.0141 | 0.14 |
| Atelectasis | Isotonic | 0.6207 | 0.2599 | 0.1935 | 0.3956 | 0.7724 | 0.1047 | 0.0175 | 0.10 |
| Cardiomegaly | Original | 0.7158 | 0.1985 | 0.2063 | 0.1912 | 0.9267 | 0.1306 | 0.2171 | 0.48 |
| Cardiomegaly | Platt | 0.7158 | 0.0000 | 0.0000 | 0.0000 | 0.9985 | 0.0849 | 0.0620 | 0.07 |
| Cardiomegaly | Isotonic | 0.6768 | 0.2680 | 0.2063 | 0.3824 | 0.8534 | 0.0840 | 0.0603 | 0.05 |
| Consolidation | Original | 0.5903 | 0.1159 | 0.0727 | 0.2857 | 0.7839 | 0.1900 | 0.3378 | 0.53 |
| Consolidation | Platt | 0.5903 | 0.1226 | 0.0765 | 0.3095 | 0.7782 | 0.0543 | 0.0184 | 0.10 |
| Consolidation | Isotonic | 0.5833 | 0.1221 | 0.0760 | 0.3095 | 0.7768 | 0.0547 | 0.0205 | 0.11 |
| Edema | Original | 0.7904 | 0.1538 | 0.2000 | 0.1250 | 0.9835 | 0.0717 | 0.1844 | 0.49 |
| Edema | Platt | 0.7904 | 0.1579 | 0.2143 | 0.1250 | 0.9848 | 0.0299 | 0.0151 | 0.21 |
| Edema | Isotonic | 0.7750 | 0.1538 | 0.2000 | 0.1250 | 0.9835 | 0.0317 | 0.0132 | 0.10 |
| Effusion | Original | 0.7147 | 0.2941 | 0.2516 | 0.3540 | 0.8132 | 0.3181 | 0.4267 | 0.77 |
| Effusion | Platt | 0.7147 | 0.2941 | 0.2516 | 0.3540 | 0.8132 | 0.1195 | 0.0186 | 0.27 |
| Effusion | Isotonic | 0.6924 | 0.3000 | 0.2653 | 0.3451 | 0.8305 | 0.1303 | 0.0636 | 0.17 |
| Emphysema | Original | 0.5339 | 0.0000 | 0.0000 | 0.0000 | 0.9805 | 0.1185 | 0.2436 | 0.56 |
| Emphysema | Platt | 0.5339 | 0.0000 | 0.0000 | 0.0000 | 0.9916 | 0.0412 | 0.0173 | 0.05 |
| Emphysema | Isotonic | 0.5582 | 0.0000 | 0.0000 | 0.0000 | 0.9805 | 0.0431 | 0.0226 | 0.05 |
| Fibrosis | Original | 0.7935 | 0.1773 | 0.1017 | 0.6923 | 0.7804 | 0.1289 | 0.2852 | 0.42 |
| Fibrosis | Platt | 0.7935 | 0.2143 | 0.2000 | 0.2308 | 0.9669 | 0.0329 | 0.0003 | 0.05 |
| Fibrosis | Isotonic | 0.7722 | 0.1744 | 0.1006 | 0.6538 | 0.7901 | 0.0327 | 0.0029 | 0.05 |
| Hernia | Original | 0.8189 | 0.2069 | 0.1765 | 0.2500 | 0.9810 | 0.0205 | 0.0718 | 0.17 |
| Hernia | Platt | 0.8189 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0158 | 0.0127 | 0.05 |
| Hernia | Isotonic | 0.7199 | 0.2400 | 0.2308 | 0.2500 | 0.9864 | 0.0153 | 0.0120 | 0.05 |
| Infiltration | Original | 0.5697 | 0.3732 | 0.2591 | 0.6667 | 0.4370 | 0.3059 | 0.3513 | 0.57 |
| Infiltration | Platt | 0.5697 | 0.3529 | 0.2878 | 0.4561 | 0.6667 | 0.1771 | 0.0537 | 0.21 |
| Infiltration | Isotonic | 0.5672 | 0.3732 | 0.2591 | 0.6667 | 0.4370 | 0.1807 | 0.0687 | 0.11 |
| Mass | Original | 0.5216 | 0.0571 | 0.0327 | 0.2258 | 0.7121 | 0.1577 | 0.3253 | 0.43 |
| Mass | Platt | 0.5216 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0397 | 0.0107 | 0.05 |
| Mass | Isotonic | 0.4879 | 0.0351 | 0.0385 | 0.0323 | 0.9652 | 0.0400 | 0.0081 | 0.05 |
| Nodule | Original | 0.4860 | 0.0692 | 0.0361 | 0.8400 | 0.2262 | 0.2185 | 0.4056 | 0.31 |
| Nodule | Platt | 0.4860 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0322 | 0.0054 | 0.05 |
| Nodule | Isotonic | 0.5380 | 0.0541 | 0.0408 | 0.0800 | 0.9352 | 0.0318 | 0.0098 | 0.05 |
| Pleural_Thickening | Original | 0.5532 | 0.0615 | 0.1000 | 0.0444 | 0.9745 | 0.1114 | 0.2031 | 0.55 |
| Pleural_Thickening | Platt | 0.5532 | 0.0597 | 0.0909 | 0.0444 | 0.9716 | 0.0570 | 0.0273 | 0.08 |
| Pleural_Thickening | Isotonic | 0.5350 | 0.0615 | 0.1000 | 0.0444 | 0.9745 | 0.0588 | 0.0364 | 0.07 |
| Pneumonia | Original | 0.6036 | 0.0500 | 0.0299 | 0.1538 | 0.9118 | 0.0557 | 0.1746 | 0.32 |
| Pneumonia | Platt | 0.6036 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0170 | 0.0055 | 0.05 |
| Pneumonia | Isotonic | 0.6049 | 0.0500 | 0.0299 | 0.1538 | 0.9118 | 0.0177 | 0.0109 | 0.05 |
| Pneumothorax | Original | 0.6533 | 0.2552 | 0.1568 | 0.6854 | 0.5038 | 0.1490 | 0.1959 | 0.30 |
| Pneumothorax | Platt | 0.6533 | 0.2386 | 0.2414 | 0.2360 | 0.9002 | 0.1105 | 0.0865 | 0.05 |
| Pneumothorax | Isotonic | 0.6521 | 0.1818 | 0.2154 | 0.1573 | 0.9228 | 0.1103 | 0.0843 | 0.05 |

## Interpretation

The comparison separates probability calibration from final test-set evaluation. Lower Brier score and ECE indicate improved probability calibration under these metrics. AUROC measures ranking performance and is generally unchanged by strictly monotonic probability transformations.

The calibrated methods are not assumed to improve disease classification performance merely because their probabilities are better calibrated. Classification metrics are reported separately.

These results are research-only and do not establish clinical diagnostic performance.