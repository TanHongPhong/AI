# So sánh Models Change3D

## Overall Metrics

| Metric | RGB | RGB+Delta | RGB+Delta+Hadamard |
|--------|-----------|-----------|-----------|
| **Accuracy** | 0.9489 | 0.9416 | 0.9489 |
| **Macro F1** | 0.9484 | 0.9409 | 0.9482 |
| **ROC-AUC** | 0.9735 | 0.9716 | 0.9789 |
| **False Alarm Rate (FAR)** | 0.0511 | 0.0584 | 0.0511 |
| **Số lượng mẫu** | 137 | 137 | 137 |

## Per-Class Metrics

### F1 Score

| Class | RGB | RGB+Delta | RGB+Delta+Hadamard |
|-------|-----------|-----------|-----------|
| mốc | 0.9431 | 0.9344 | 0.9421 |
| xoay&dịch chuyển | 0.9536 | 0.9474 | 0.9542 |

### Precision

| Class | RGB | RGB+Delta | RGB+Delta+Hadamard |
|-------|-----------|-----------|-----------|
| mốc | 0.9508 | 0.9500 | 0.9661 |
| xoay&dịch chuyển | 0.9474 | 0.9351 | 0.9359 |

### Recall

| Class | RGB | RGB+Delta | RGB+Delta+Hadamard |
|-------|-----------|-----------|-----------|
| mốc | 0.9355 | 0.9194 | 0.9194 |
| xoay&dịch chuyển | 0.9600 | 0.9600 | 0.9733 |
