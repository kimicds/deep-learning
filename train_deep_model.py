import argparse
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import (
    add_features,
    build_preprocessor,
    numeric_feature_columns,
    transform_features,
)
from fraud_model import FraudMLP, choose_f1_threshold, predict


RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    project_dir = Path(__file__).resolve().parent
    train_df = pd.read_csv(project_dir / "train.csv")
    test_df = pd.read_csv(project_dir / "test.csv")

    if "is_fraud" not in train_df:
        raise ValueError("train.csv must contain an is_fraud target column")

    train_df = add_features(train_df)
    test_df = add_features(test_df)
    preprocessor = build_preprocessor(train_df.drop(columns=["is_fraud"]))

    numeric_columns = numeric_feature_columns(train_df)
    for column in numeric_columns:
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce").fillna(0)
        test_df[column] = pd.to_numeric(test_df[column], errors="coerce").fillna(0)

    X = train_df.drop(columns=["id", "is_fraud"])
    y = train_df["is_fraud"].astype(np.float32).to_numpy()
    X_test = test_df.drop(columns=["id"])
    test_ids = test_df["id"].to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_valid, X_test = transform_features(
        preprocessor, X_train, X_valid, X_test
    )
    joblib.dump(preprocessor, project_dir / "fraud_preprocessor.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudMLP(X_train.shape[1]).to(device)

    fraud_count = float(y_train.sum())
    normal_count = float(len(y_train) - fraud_count)
    pos_weight = torch.tensor([np.sqrt(normal_count / fraud_count)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_ap = -np.inf
    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print(
        f"Training on {device}: {len(y_train):,} train rows, "
        f"{len(y_valid):,} validation rows, {X_train.shape[1]} features"
    )
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        validation_probabilities = predict(
            model, X_valid, device, args.batch_size
        )
        roc_auc = roc_auc_score(y_valid, validation_probabilities)
        average_precision = average_precision_score(
            y_valid, validation_probabilities
        )
        threshold, validation_f1 = choose_f1_threshold(
            y_valid, validation_probabilities
        )
        scheduler.step(average_precision)

        print(
            f"Epoch {epoch:02d} | loss={total_loss / len(y_train):.5f} "
            f"| ROC-AUC={roc_auc:.5f} | PR-AUC={average_precision:.5f} "
            f"| F1={validation_f1:.5f} | threshold={threshold:.4f} "
            f"| time={time.time() - start:.1f}s"
        )

        if average_precision > best_ap:
            best_ap = average_precision
            best_auc = roc_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            np.save(project_dir / "fraud_threshold.npy", np.array([threshold]))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), project_dir / "fraud_mlp.pt")

    test_probabilities = predict(model, X_test, device, args.batch_size)
    submission = pd.DataFrame(
        {"id": test_ids, "is_fraud": test_probabilities}
    )
    submission.to_csv(project_dir / "submission.csv", index=False)

    print(f"Best ROC-AUC: {best_auc:.5f}")
    print(f"Best PR-AUC: {best_ap:.5f}")
    print("Saved: fraud_mlp.pt, fraud_preprocessor.joblib, submission.csv")


if __name__ == "__main__":
    main()
