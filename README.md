# Titanic ML Pipeline

## 目的
再現性ある表データMLパイプライン構築

## モデル
LightGBM, LogisticRegression

## 特徴量と前処理

使用特徴量：
- Pclass
- Sex（male=0, female=1）
- Age（中央値補完）
- SibSp
- Parch
- Fare
- Embarked（欠損をSで補完し数値化）

最小限の前処理でベースライン性能を評価した。

## 評価指標
F1（クラス不均衡のため）

## 再現性

- seed: 42
- StratifiedKFold: 5-fold
- 設定は configs/exp1.yaml で管理
- CV結果は logs/ に保存

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

## 今後の改善

- LightGBMのハイパーパラメータチューニング
- 特徴量エンジニアリング（FamilySize, Titleなど）
- 誤分類サンプルの分析
- EDAノートブックの追加

## 実行方法
- pip install -r requirements.txt
- python src/train.py
- ログは logs/cv_compare.txt に出力される。

