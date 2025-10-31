# So sánh Model: Có Delta vs Không Delta

## Overall Metrics

| Metric                     | Không Delta | Có Delta |
| -------------------------- | ----------- | -------- |
| **Accuracy**               | 0.9416      | 0.9197   |
| **Macro F1**               | 0.9409      | 0.9191   |
| **ROC-AUC**                | 0.9716      | 0.9768   |
| **False Alarm Rate (FAR)** | 0.0584      | 0.0803   |
| **Số lượng mẫu**           | 137         | 137      |

## Per-Class Metrics

### F1 Score

| Class       | Không Delta | Có Delta |
| ----------- | ----------- | -------- |
| mốc         | 0.9344      | 0.9120   |
| dịch chuyển | 0.9474      | 0.9262   |

### Precision

| Class            | Không Delta | Có Delta |
| ---------------- | ----------- | -------- |
| mốc              | 0.9500      | 0.9048   |
| xoay&dịch chuyển | 0.9351      | 0.9324   |

### Recall

| Class            | Không Delta | Có Delta |
| ---------------- | ----------- | -------- |
| mốc              | 0.9194      | 0.9194   |
| xoay&dịch chuyển | 0.9600      | 0.9200   |
