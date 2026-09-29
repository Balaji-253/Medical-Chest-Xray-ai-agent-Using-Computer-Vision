# Bootstrap Statistical Evaluation

## Experimental Configuration

- Model: **ResNet-18 weighted baseline**
- Bootstrap iterations: **1000**
- Confidence level: **95%**
- Random seed: **42**

## Results

| Class | AUROC | 95% CI | F1 | 95% CI |
|---|---:|---|---:|---|
| Atelectasis | 0.6257 | [0.5651, 0.6817] | 0.2609 | [0.1935, 0.3236] |
| Cardiomegaly | 0.7158 | [0.6587, 0.7712] | 0.0000 | [0.0000, 0.0000] |
| Consolidation | 0.5903 | [0.5082, 0.6746] | 0.1475 | [0.1021, 0.1976] |
| Edema | 0.7904 | [0.6946, 0.8768] | 0.1111 | [0.0000, 0.2581] |
| Effusion | 0.7147 | [0.6686, 0.7588] | 0.3238 | [0.2564, 0.3897] |
| Emphysema | 0.5339 | [0.4565, 0.6154] | 0.0000 | [0.0000, 0.0000] |
| Fibrosis | 0.7935 | [0.6925, 0.8779] | 0.1717 | [0.1043, 0.2387] |
| Hernia | 0.8189 | [0.6834, 0.9290] | 0.1754 | [0.0392, 0.3175] |
| Infiltration | 0.5697 | [0.5208, 0.6150] | 0.3562 | [0.3019, 0.4070] |
| Mass | 0.5216 | [0.4143, 0.6336] | 0.0773 | [0.0502, 0.1065] |
| Nodule | 0.4860 | [0.3702, 0.5979] | 0.0654 | [0.0391, 0.0914] |
| Pleural_Thickening | 0.5532 | [0.4641, 0.6410] | 0.0597 | [0.0000, 0.1538] |
| Pneumonia | 0.6036 | [0.4548, 0.7506] | 0.0328 | [0.0000, 0.0834] |
| Pneumothorax | 0.6533 | [0.5923, 0.7158] | 0.2551 | [0.2063, 0.3023] |

## Interpretation

The confidence intervals quantify sampling uncertainty within the current local test subset. Wider intervals indicate greater uncertainty in the estimated metric. Classes with fewer positive samples can have substantially wider intervals.

These results are research evaluation results and should not be interpreted as clinical validation or evidence of diagnostic performance in clinical practice.
