# Probability Calibration Evaluation

## Model

**ResNet-18 weighted baseline**

## Aggregate Calibration Metrics

- Mean Brier Score: **0.1619**
- Mean Expected Calibration Error: **0.2738**
- Calibration bins: **10**

## Per-Class Results

| Class | Samples | Positive | Brier Score | ECE |
|---|---:|---:|---:|---:|
| Atelectasis | 750 | 91 | 0.2906 | 0.4111 |
| Cardiomegaly | 750 | 68 | 0.1306 | 0.2171 |
| Consolidation | 750 | 42 | 0.1900 | 0.3378 |
| Edema | 750 | 24 | 0.0717 | 0.1844 |
| Effusion | 750 | 113 | 0.3181 | 0.4267 |
| Emphysema | 750 | 32 | 0.1185 | 0.2436 |
| Fibrosis | 750 | 26 | 0.1289 | 0.2852 |
| Hernia | 750 | 12 | 0.0205 | 0.0718 |
| Infiltration | 750 | 171 | 0.3059 | 0.3513 |
| Mass | 750 | 31 | 0.1577 | 0.3253 |
| Nodule | 750 | 25 | 0.2185 | 0.4056 |
| Pleural_Thickening | 750 | 45 | 0.1114 | 0.2031 |
| Pneumonia | 750 | 13 | 0.0557 | 0.1746 |
| Pneumothorax | 750 | 89 | 0.1490 | 0.1959 |

## Interpretation

The Brier score measures the mean squared difference between predicted probability and the binary outcome. Lower values indicate smaller probabilistic error.

Expected Calibration Error (ECE) summarizes the difference between predicted probability and observed frequency across probability bins. Lower values indicate closer agreement between predicted confidence and observed frequency under this evaluation procedure.

Calibration metrics are descriptive research measurements for the current test subset. They do not establish clinical reliability or diagnostic validity.
