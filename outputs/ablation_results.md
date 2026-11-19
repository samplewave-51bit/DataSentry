# Phase 3: Deep Learning Component Ablation Results

Ablation study comparing PyTorch Autoencoder against Isolation Forest, Mahalanobis Distance, and PCA Reconstruction Error on sensor telemetry with injected anomalies.

| model                   | mean_f1 | mean_auc | mean_precision | mean_recall | avg_latency_ms | fit_time_ms |
| ----------------------- | ------- | -------- | -------------- | ----------- | -------------- | ----------- |
| pytorch_autoencoder     | 0.7278  | 0.9996   | 0.5744         | 1.0         | 10.7775        | 5324.86     |
| pca_reconstruction      | 0.6572  | 1.0      | 0.4913         | 1.0         | 8.642          | 17.94       |
| mahalanobis             | 0.6538  | 0.9804   | 0.7839         | 0.57        | 15.9856        | 0.0         |
| isolation_forest_prefit | 0.5527  | 0.9456   | 0.5064         | 0.63        | 40.6506        | 238.76      |
| isolation_forest_online | 0.5067  | 0.9673   | 0.76           | 0.38        | 238.2326       | 0.0         |

### Learned Domain Classifier Drift Metrics
- **Clean Negative Control ROC AUC:** 0.531 (Alert: False)
- **Shifted Batch (+2.5 sigma) ROC AUC:** 0.959 (Alert: True)
