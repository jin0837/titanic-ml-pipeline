# Titanic ML Pipeline

## 目的
再現性ある表データMLパイプライン構築

## モデル
LightGBM

## 評価指標
F1（クラス不均衡のため）

## 検証方法
StratifiedKFold 5-fold

## 結果
CV F1: 0.7757

## 実行方法
pip install -r requirements.txt
python src/train.py