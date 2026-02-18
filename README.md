# Titanic ML Pipeline

## 目的
再現性ある表データMLパイプライン構築

## モデル
LightGBM

## 評価指標
F1（クラス不均衡のため）

## 検証方法
StratifiedKFold 5-fold

## モデル比較と考察

同一のStratifiedKFold（5-fold）とF1指標で
LogisticRegressionとLightGBMを比較した。

結果：
- LogisticRegression: CV F1 = 0.7229
- LightGBM: CV F1 = 0.7738

LightGBMの方が高い性能を示した。
本データは数値・カテゴリ特徴が混在しており、
特徴量間の非線形関係が存在すると考えられる。
線形モデルではその関係を表現しきれない一方、
GBDTは非線形な分割により相互作用を学習できるため、
性能差が生じたと解釈している。


## 実行方法
pip install -r requirements.txt
python src/train.py
