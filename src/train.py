import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def load_config(path: str = "configs/exp1.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config is empty or invalid yaml")
    return cfg


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Minimal preprocessing
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"]
    return X, y


def cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(f1_score(y_va, pred))
    return float(np.mean(scores))


def main() -> None:
    cfg = load_config()

    seed = int(cfg["seed"])
    n_splits = int(cfg["n_splits"])

    df = pd.read_csv("data/train.csv")
    X, y = preprocess(df)

    # Models from config
    lr_params = cfg.get("models", {}).get("logistic_regression", {})
    lgb_params = cfg.get("models", {}).get("lightgbm", {})

    lr = LogisticRegression(
        C=float(lr_params.get("C", 1.0)),
        max_iter=int(lr_params.get("max_iter", 1000)),
        solver="lbfgs",
        random_state=seed,
    )

    lgb = LGBMClassifier(
        random_state=seed,
        n_estimators=int(lgb_params.get("n_estimators", 300)),
        learning_rate=float(lgb_params.get("learning_rate", 0.05)),
        num_leaves=int(lgb_params.get("num_leaves", 31)),
    )

    results = {}
    results["LogisticRegression"] = cv_score(lr, X, y, n_splits=n_splits, seed=seed)
    results["LightGBM"] = cv_score(lgb, X, y, n_splits=n_splits, seed=seed)

    # print
    for k, v in results.items():
        print(f"{k} CV F1: {v:.6f}")

    # log
    with open("logs/cv_compare.txt", "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}\tCV_F1\t{v:.6f}\n")


if __name__ == "__main__":
    main()
