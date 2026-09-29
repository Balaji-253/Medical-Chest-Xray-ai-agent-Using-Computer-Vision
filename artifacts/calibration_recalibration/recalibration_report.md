# Probability Recalibration Experiment

## Methodology

Calibration methods were fitted using the **validation split only** and evaluated on the untouched test prediction set.

Methods compared:

1. Original model probabilities
2. Platt/logistic calibration
3. Isotonic regression calibration

## Aggregate Results

| Method | Mean Brier | Mean ECE |
|---|---:|---:|
| Original | 0.1619 | 0.2738 |
| Platt | 0.0655 | 0.0248 |
| Isotonic | 0.0668 | 0.0308 |

## Per-Class Results

| Class | Original Brier | Platt Brier | Isotonic Brier | Original ECE | Platt ECE | Isotonic ECE |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.2906 | 0.1047 | 0.1047 | 0.4111 | 0.0141 | 0.0175 |
| Cardiomegaly | 0.1306 | 0.0849 | 0.0840 | 0.2171 | 0.0620 | 0.0603 |
| Consolidation | 0.1900 | 0.0543 | 0.0547 | 0.3378 | 0.0184 | 0.0205 |
| Edema | 0.0717 | 0.0299 | 0.0317 | 0.1844 | 0.0151 | 0.0132 |
| Effusion | 0.3181 | 0.1195 | 0.1303 | 0.4267 | 0.0186 | 0.0636 |
| Emphysema | 0.1185 | 0.0412 | 0.0431 | 0.2436 | 0.0173 | 0.0226 |
| Fibrosis | 0.1289 | 0.0329 | 0.0327 | 0.2852 | 0.0003 | 0.0029 |
| Hernia | 0.0205 | 0.0158 | 0.0153 | 0.0718 | 0.0127 | 0.0120 |
| Infiltration | 0.3059 | 0.1771 | 0.1807 | 0.3513 | 0.0537 | 0.0687 |
| Mass | 0.1577 | 0.0397 | 0.0400 | 0.3253 | 0.0107 | 0.0081 |
| Nodule | 0.2185 | 0.0322 | 0.0318 | 0.4056 | 0.0054 | 0.0098 |
| Pleural_Thickening | 0.1114 | 0.0570 | 0.0588 | 0.2031 | 0.0273 | 0.0364 |
| Pneumonia | 0.0557 | 0.0170 | 0.0177 | 0.1746 | 0.0055 | 0.0109 |
| Pneumothorax | 0.1490 | 0.1105 | 0.1103 | 0.1959 | 0.0865 | 0.0843 |

## Interpretation

A lower Brier score indicates lower probabilistic prediction error. A lower ECE indicates closer agreement between predicted probabilities and observed frequencies under the selected binning procedure.

Calibration method selection should be based on validation methodology and independent test evaluation rather than optimizing directly on the test set.

These results are research-only and do not establish clinical reliability.