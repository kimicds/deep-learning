import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FraudMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(1)


def predict(model, features, device, batch_size):
    model.eval()
    predictions = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions)


def choose_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    thresholds = np.append(thresholds, 1.0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    return float(thresholds[best_index]), float(f1[best_index])
