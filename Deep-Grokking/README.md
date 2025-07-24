# Replication Study of "Beyond Memorization: The Role of Robustness in Grokking"

This repository contains the code for a replication study of the paper:

> **Deep Grokking: Would Deep Neural Networks Generalize Better?** \> Simin Fan, Razvan Pascanu, Martin Jaggi  
> [https://arxiv.org/abs/2405.19454](https://arxiv.org/abs/2405.19454)

-----

## 🛠️ セットアップ (Setup)

### 1\. リポジトリのクローン

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2\. 依存関係のインストール

仮想環境を作成し、必要なライブラリをインストールすることをお勧めします。

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

`requirements.txt` ファイルの内容は以下の通りです。

```text
torch
torchvision
numpy
PyYAML
tqdm
wandb
```

-----

## 🚀 実験の実行 (Running Experiments)

実験は `config/` ディレクトリ内のYAMLファイルで定義します。

### 1\. 単一の実験を実行する

単一の実験を実行するには、`src/train.py` スクリプトに設定ファイルを指定します。

```bash
python src/train.py --config config/exp_d4_n2000_wd0.005.yaml
```

### 2\. 複数の実験を並列で実行する

複数の実験を単一のGPUで同時に実行するには、`run_parallel.sh` スクリプトを使用します。

**注意**: このスクリプトは、指定した数のジョブを同時に実行します。GPUのVRAM（ビデオメモリ）容量に応じて、スクリプト内の `MAX_JOBS` の値を調整してください。VRAMが不足するとエラーで停止します。

```bash
# スクリプトに実行権限を付与
chmod +x run_parallel.sh

# スクリプトを実行
./run_parallel.sh
```

実行したい実験のリストは、`run_parallel.sh` ファイル内の `configs` 配列を編集してください。

-----

## ⚙️ 設定 (Configuration)

すべての実験パラメータは `.yaml` ファイルで管理します。新しい実験を行うには、既存のファイルをコピーするか、新規に作成してください。

**`config/config_example.yaml` の主なパラメータ:**

  * **`depth`**: モデルの隠れ層の数
  * **`n_train`**: トレーニングに使用するデータサンプル数
  * **`weight_decay`**: L2正則化（重み減衰）の強度
  * **`alpha`**: 重み初期化のスケール係数
  * **`max_steps`**: トレーニングの総ステップ数
  * **`log_interval`**: 損失や精度などの基本的なメトリクスをログに記録する間隔
  * **`probe_interval`**: 特徴ランクやプロービング精度など、計算コストの高い分析を実行する間隔

-----

## 📈 結果とロギング (Results & Logging)

実験結果は、コンソールへの出力、チェックポイントファイル、およびWeights & Biases（W\&B）を通じて確認できます。

  * **コンソール出力**: `use_wandb: false` の場合、学習の進捗は標準出力に表示されます。
  * **Weights & Biases**: `use_wandb: true` の場合、すべてのメトリクス（損失、精度、特徴ランク、プローブ精度など）がW\&Bのダッシュボードに送信されます。実験の追跡や比較に便利です。
  * **モデルのチェックポイント**: 学習が完了したモデルの重みは、`.pt` ファイルとして `checkpoints/` ディレクトリに保存されます。

-----

## 📁 ディレクトリ構造 (Directory Structure)

```
.
├── config/                  # 実験設定ファイル (.yaml)
├── data/                    # MNISTデータセット (自動でダウンロードされます)
├── checkpoints/             # 保存されたモデルの重み (.pt)
├── src/
│   ├── data.py              # データローダー
│   ├── model.py             # MLPモデルの定義
│   ├── train.py             # メインの学習スクリプト
│   └── utils.py             # 特徴ランク計算や線形プロービングなどのユーティリティ
├── run_parallel.sh          # 複数実験を並列実行するスクリプト
├── requirements.txt         # 依存ライブラリ
└── README.md                # このファイル
```

-----

## 📄 引用 (Citation)

この研究は以下の論文に基づいています。引用する際はこちらをご利用ください。

```bibtex
@article{fan2024deep,
    title={Deep grokking: Would deep neural networks generalize better?},
    author={Fan, Simin and Pascanu, Razvan and Jaggi, Martin},
    year={2024},
    eprint={2405.19454},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```