# Fraud Detection Deep Learning Model

This project trains a PyTorch MLP to predict `is_fraud` for the Kaggle-style
fraud dataset.

Run the full training and generate `submission.csv`:

```powershell
.\.venv\Scripts\python.exe train_deep_model.py --epochs 10 --batch-size 4096
```

The code is split by responsibility:

- `data_preprocessing.py` engineers features and scales/encodes the data.
- `fraud_model.py` defines the PyTorch network and prediction helpers.
- `train_deep_model.py` runs the stratified training/validation workflow,
  early stopping, model saving, and test submission generation.

The pipeline uses class-weighted BCE loss for the 1.67% fraud class and keeps
the validation rows separate when fitting the preprocessing transformations.
It saves the best model using validation PR-AUC and writes fraud probabilities
to `submission.csv`.

Validation result from the last full run:

- ROC-AUC: 0.93928
- PR-AUC: 0.49788