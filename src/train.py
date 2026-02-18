import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# ======================
# config読み込み
# ======================
with open("configs/exp1.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED = config["seed"]
N_SPLITS = config["n_splits"]
TARGET = config["target"]

np.random.seed(SEED)

# ======================
# データ読み込み
# ======================
df = pd.read_csv("data/train.csv")

# ======================
# 簡易前処理
# ======================
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")

# カテゴリを数値化
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df[TARGET]

# ======================
# CV
# ======================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

scores = []

for train_idx, valid_idx in skf.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    score = f1_score(y_valid, preds)
    scores.append(score)

print("CV F1:", np.mean(scores))
