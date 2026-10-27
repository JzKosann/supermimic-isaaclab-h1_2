# SuperMimic-H1_2

[English](README.md) | [中文](README.zh-CN.md) | **日本語**

論文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）を、**Isaac Lab** 上で **Unitree H1-2** ヒューマノイド（27 自由度、ハンドなし）向けに再現した**非公式**実装です。

SuperMimic は、シミュレーションのデータから微分可能なロボットの世界モデルを学習し、その中で短いホライズンの想像ロールアウトを行って勾配を逆伝播（BPTT）させることで、モーション追従方策を学習します。PPO は使いません。
本リポジトリはこのパイプライン全体を Isaac Lab 拡張として実装しています。論文の公式コードは公開されていないため、論文本文、公開されている 2 つの G1 再現実装、そして論文に記載のない部分については私たち自身の判断に基づいています（[論文との違い](#論文との違い)を参照）。

> **現状**：コードは完成しており、想像側の解析ヘッドはシミュレータと項目ごとに一致することを確認済みです（[整合性チェック](#3-整合性チェック)を参照）。H1-2 でのフル学習の結果はまだ公開していません。

## 目次

- [機能](#機能)
- [タスク](#タスク)
- [動作環境](#動作環境)
- [インストール](#インストール)
- [使い方](#使い方)
- [Docker](#docker)
- [ディレクトリ構成](#ディレクトリ構成)
- [論文との違い](#論文との違い)
- [結果](#結果)
- [謝辞](#謝辞)
- [ライセンス](#ライセンス)

## 機能

- **世界モデル + BPTT 学習器**（Algorithm A.1）：出力を積分して次状態を得るダイナミクス MLP（3 × 2048）、学習される終了ヘッド、8 ステップのオープンループ学習損失（Eq. 1）。
  ホライズン H = 32 の想像ロールアウト、決定論的方策、GAE（λ = 0.95）ターゲットに当てはめる critic、Isaac Lab から収集するリプレイバッファ。
- **解析ヘッド**：観測、8 項目の追従・正則化報酬、終了特徴量を、微分可能な URDF 順運動学で予測状態から計算するため、報酬から方策まで勾配が流れます。
- **複数クリップのモーションライブラリ**と、リターンに基づくリセットサンプリング（1 秒ごとのビン）。
- **Appendix F の評価**：各クリップを先頭フレームから 1 回通しで再生（ノイズなし、ドメインランダム化なし）。
- **可視化**：参照モーションの半透明の「ゴースト」と、平行移動だけで追従しマウスでの回転・ズームを妨げないカメラ。
- **データパイプライン**：H1-2 にリターゲットした LAFAN1（40 クリップ、2.45 時間。論文 Fig. 4a と同じクリップ集合）を 50 Hz に変換。
- 公式 Isaac Lab イメージをベースにした **Docker** 設定。

## タスク

`import supermimic_h1_2.tasks` で次の gym タスクが登録されます：

| タスク | 方策の観測（1 フレームの次元 × 履歴 10 フレーム） | 目標（1 フレームの次元 × 10 フレーム） | 説明 |
|---|---|---|---|
| `Tracking-Flat-H1_2-SuperMimic-v0` | 87：関節位置（エンコーダバイアス込み）/ 速度、IMU 角速度、重力方向、前回の行動 | 64：未来の参照フレーム | 論文の観測（Table A.1）。ベース位置・並進速度を含まないため状態推定器が不要。 |
| `Tracking-Flat-H1_2-SuperMimic-Anchor-v0` | v0 と同じ | 64 + 9 次元のアンカー相対姿勢 | 論文からの逸脱（SuperMimic-Reproduced の実験 e3）。 |
| `Tracking-Flat-H1_2-SuperMimic-v1` | 87 + 6：ロボットに対する参照ルート姿勢（6D 表現） | 69：軌道形式の目標 | v0 に supermimic_vibed の改良を加えたもの（[論文との違い](#論文との違い)を参照）。 |
| `…-Eval-v0`、`…-Anchor-Eval-v0`、`…-Eval-v1` | | | Appendix F の評価環境。`eval_benchmark.py` / `play.py` がチェックポイントから自動で選びます。 |

どのバリアントもロボット、報酬（Table A.2）、終了条件、ドメインランダム化（Table A.3）は共通です。

## 動作環境

| | 動作確認したバージョン |
|---|---|
| OS | Ubuntu（Linux） |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 + CUDA 12.8（Isaac Sim 同梱） |
| Python | 3.11 |
| GPU | フル設定（環境 4096、想像バッチ 4096）はサーバーの GPU 1 枚で学習。8 GB のノート PC 用 GPU（RTX 5060）ではスモークテスト設定のみ動作。 |

`torch.compile` はデフォルトで有効で、方策更新が約 4.5 倍速くなります。`--compile none` で無効にできます。

## インストール

1. **Isaac Lab 2.3.2 をインストール**：[公式ドキュメント](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)に従ってインストールし、その Python 環境を有効にします。

2. **本リポジトリをクローンして拡張をインストール**（editable モードが必須です。デフォルトのパスはリポジトリの位置から求めます）：

   ```bash
   git clone <このリポジトリの URL> SuperMimic-H1_2
   cd SuperMimic-H1_2
   python -m pip install -e source/supermimic_h1_2
   ```

3. **ロボットモデルとモーションデータをダウンロード**（約 120 MB、Isaac 不要）：

   ```bash
   python scripts/download_assets.py
   ```

   H1-2 の USD を [unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) から `assets/unitree_model/` に、
   40 本の H1-2 CSV を [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) から `datasets/lafan1_csv/h1_2/` に保存します。

4. **クリップを 50 Hz の npz に変換**（Isaac Lab 上で 1 フレームずつ再生。ノート PC 用 RTX 5060 で約 5 分）：

   ```bash
   python scripts/csv_to_npz.py --headless
   ```

   出力先は `datasets/lafan1_h1_2_npz/*.npz` です。

パスは環境変数で上書きできます：`SUPERMIMIC_UNITREE_MODEL_DIR`、`SUPERMIMIC_CSV_DIR`、`SUPERMIMIC_MOTION_DIR`、`SUPERMIMIC_LOG_ROOT`
（`source/supermimic_h1_2/supermimic_h1_2/paths.py` を参照）。

## 使い方

以下のコマンドはすべてリポジトリのルートで実行します。

### 1. 学習

```bash
# 論文の設定（LAFAN1、環境 4096、想像バッチ 4096、H = 32、UTD 0.4、32 万イテレーション）
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v0 --run_name lafan1

# v1 レシピ
python scripts/supermimic/train.py --headless --task Tracking-Flat-H1_2-SuperMimic-v1 --run_name lafan1_v1

# 小さな GPU でのスモークテスト
python scripts/supermimic/train.py --headless --num_envs 64 --small_nets --policy_batch 256 --wm_batch 256 \
    --total_iterations 600 --include walk1

# 再開（ログは同じディレクトリに続けて書き込まれます）
python scripts/supermimic/train.py --headless --task <同じタスク> --resume logs/supermimic/<run>/ckpt_latest.pt
```

ログは `logs/supermimic/<日付>_<run_name>/` に保存されます：`config.json`、`tb/` 以下の TensorBoard ログ、4000 イテレーションごとのチェックポイント（`ckpt_XXXXXXX.pt`）と `ckpt_latest.pt`。
Weights & Biases を使う場合は `--logger wandb --log_project_name <名前>`。`--max_hours` で指定時間に達すると停止します。

よく使うオプション：`--policy_batch`（メモリ不足なら 2048 に下げる）、`--wm_batch`、`--horizon`、`--replay_capacity`、`--motion_dir`、`--include <部分文字列> …`。

マルチ GPU のデータ並列は実装済み（`python -m torch.distributed.run --nproc_per_node=N scripts/supermimic/train.py --headless --distributed`）ですが、**未テスト**です。

### 2. 評価（Appendix F）

```bash
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>/ckpt_0040000.pt
python scripts/supermimic/eval_benchmark.py --headless --ckpt logs/supermimic/<run>   # run 内の全チェックポイントを評価
```

各クリップを先頭フレームから 1 回再生します。**正規化リターン** = Σ 各ステップの報酬 / (0.1 × ステップ数) で、1.0 が完全な追従です。学習時の終了条件（ルート高さ・傾き）が発動したクリップは**失敗**とします。
クリップごとの結果は `<run>/eval/<ckpt>.json` に、チェックポイントごとに 1 行が `<run>/eval/summary.csv` に追記されます。
`--headless` を付けなければ、参照ゴースト付きで評価の様子を描画します。

### 3. 整合性チェック

想像で使う解析ヘッドが Isaac Lab と項目ごとに一致するかを確認します（全 28 リンクの順運動学、方策の観測、すべての報酬項、目標ウィンドウ）。ロボット・観測・報酬を変更したら必ず実行してください：

```bash
python scripts/supermimic/check_consistency.py --headless --num_envs 32 --steps 300 [--task <タスク>]
```

正常なら（v0・v1 で実測）順運動学の位置誤差は 1e-5 m 未満、速度誤差は 1e-4 未満、観測の誤差は 1e-6 程度、各報酬項の誤差は 1e-7 程度です。

### 4. 再生

```bash
python scripts/supermimic/play.py --ckpt logs/supermimic/<run>                  # run の最新チェックポイント
python scripts/supermimic/play.py --ckpt <ckpt.pt> --num_envs 4 --include walk --real_time
```

青い半透明のロボットが参照モーションです。カメラは 1 台のロボットを追従しますが、マウスで自由に回転・ズームできます。キー操作：`[` / `]` で追従する環境を切り替え、`L` で追従のオン・オフ。

## Docker

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
cd docker
docker compose build
docker compose run --rm prepare-data                    # ダウンロード + 変換（初回のみ）
GPU=0 TASK=Tracking-Flat-H1_2-SuperMimic-v1 RUN_NAME=lafan1_v1 docker compose up -d train
GPU=0 CKPT=logs/supermimic/<run> docker compose run --rm eval
```

`logs/`、`datasets/`、`assets/unitree_model/`、`source/`、`scripts/` はホストからマウントするので、コードを変更してもイメージの再ビルドは不要です。
Weights & Biases を使う場合は `LOGGER=wandb WANDB_API_KEY=…`、再開する場合は `RESUME=<ckpt>` を設定します。

## ディレクトリ構成

```
SuperMimic-H1_2/
├── assets/h1_2_description/        微分可能な順運動学で使う H1-2 の URDF（unitree_ros より）
├── docker/                         Dockerfile + docker-compose.yml
├── scripts/
│   ├── download_assets.py          HuggingFace からロボット USD と LAFAN1 H1-2 CSV を取得
│   ├── csv_to_npz.py               CSV（30 fps）→ npz（50 Hz）、Isaac Lab 上で再生して変換
│   └── supermimic/                 train.py、eval_benchmark.py、play.py、check_consistency.py
└── source/supermimic_h1_2/supermimic_h1_2/
    ├── paths.py                    デフォルトのパス（環境変数で上書き可）
    ├── robots.py                   H1-2 のロボット設定
    ├── tasks/                      gym 登録 + 環境設定
    └── mdp/
        ├── commands.py             複数クリップのモーションコマンド、リターンに基づくリセットサンプリング
        ├── actions.py、events.py、observations.py、rewards.py、terminations.py
        ├── kinematics.py           微分可能な URDF 順運動学（torch）
        ├── motion_library.py       モーションライブラリ、参照フレーム、目標ウィンドウ
        ├── state.py                世界モデルの状態表現
        ├── world_model.py          ダイナミクス + 終了ヘッド
        ├── heads.py                解析的な観測・報酬・終了ヘッド
        ├── imagination.py          想像ロールアウト、BPTT、critic
        ├── policy.py、replay.py、trainer.py、distributed.py
        ├── benchmark.py            Appendix F の評価
        └── viewer.py               参照ゴースト + 追従カメラ
```

## 論文との違い

論文に数値の記載がない点、または意図的に変更した点：

| 項目 | 本リポジトリ |
|---|---|
| ロボット | G1 ではなく Unitree H1-2（ハンドなし、27 関節）。アクチュエータは unitree_rl_lab のもの（暗黙的 PD）。 |
| 行動スケール α | 0.25（論文に記載なし）。 |
| 物理エンジン | PhysX（Isaac Lab）。 |
| 参照速度 | URDF 順運動学で求めたリンク原点の速度。npz に保存されている `body_lin_vel_w` は重心速度なので使用しない。 |
| 報酬の再アンカー | 現在ステップのロボットのアンカーを使用（BeyondMimic は 1 ステップ遅れ）。 |
| リプレイ用の終端状態 | `SuperMimicRLEnv` にリセット前フックを追加し、リセットで上書きされる前に s<sub>T+1</sub> を取得（Isaac Lab の RecorderManager は環境 4096 では遅すぎるため）。 |
| 方策 | 論文と同じく決定論的。 |
| `-Anchor-v0` | 目標にロボットに対する参照アンカーの姿勢（9 次元）を追加。論文にはない。 |
| `-v1` | v0 + 観測に参照ルート姿勢、軌道形式の目標、最初の 500 回の更新は世界モデルのみ学習、想像の開始点がクリップ末尾を越えない、解析的なソフト終了、critic の入力に追従誤差。supermimic_vibed から採用しなかったもの：移動平均による正規化、bf16、BeyondMimic の低ゲインアクチュエータ、Newton 物理エンジン。v1 の挙動はすべてスイッチで切り替えられ、デフォルト値は v0 と同じです。 |

## 結果

準備中です。参考までに、論文では G1 で LAFAN1 を H100 1 枚・1.2 時間学習して正規化リターン 0.82 と報告されています。2 つの G1 再現実装の数値は下記のリンク先を参照してください。

## 謝辞

- *SuperMimic*（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）：手法そのもの。
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（Yuxi Liu、MIT）：本リポジトリがモジュール単位で移植した mjlab / G1 の再現実装。
- [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)（Chong Zhang、MIT）：v1 レシピの出典。
- [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)：追従報酬と終了条件。
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)、[unitree_ros](https://github.com/unitreerobotics/unitree_ros)、
  [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model)：ロボットの設定とモデル。
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) と
  [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)：モーションデータ。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)。

## ライセンス

コードは [MIT ライセンス](LICENSE)で公開しています。サードパーティのコード、同梱の URDF、外部データにはそれぞれのライセンスが適用されます。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。モーションデータとロボットの USD は本リポジトリでは再配布していません。
