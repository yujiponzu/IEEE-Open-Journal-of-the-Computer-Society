# pct_llm

Political Compass Test (PCT) の設問に対して LLM の回答を生成し、  
回答を自動入力して経済軸・社会軸スコアを収集する実験用コードです。

## 概要
- `main.py`
  - 実験のエントリポイント
  - LLM への問い合わせ、PCT 受験（Selenium）、CSV 出力を実行
- `get_answer.py`
  - モデルごとの API 呼び出しと回答パース（1-4 への変換）
- `make_prompts.py`
  - イデオロギー設定と few-shot 例を使ったプロンプト生成
- `config.json`
  - JSON モード実行時の実験定義

## 必要環境
- Python 3.13+
- Google Chrome
- `uv`（ローカル実行時）

## セットアップ
```bash
uv sync
```

`.env` をリポジトリ直下（プロジェクトルート）に作成し、必要なキーを設定してください。

```dotenv
# OpenAI
NAIST_OPENAI_API_KEY=...

# Anthropic (claude利用時)
NAIST_CLAUDE_API_KEY=...

# Together AI (Togetherモデル利用時)
TOGETHER_AI_API_KEY=...

# Typhoon (typhoon利用時)
TYPHOON_API_KEY=...

# 任意: 実験完了通知
DISCORD_WEBHOOK_URL=...
```

## 実行方法
### 1. JSON 設定ファイルで実行
```bash
uv run python src/experiment/main.py --config data/experiment/config.json
```

### 2. CLI モードで実行
ベースラインのみ:
```bash
uv run python src/experiment/main.py --model gpt-5.2-2025-12-11 --only_baseline true
```

ベースライン + 全パターン（eco/soc × example_num）:
```bash
uv run python src/experiment/main.py --model gpt-5.2-2025-12-11 --run-all
```

試行回数（`experiment_num`）を指定:
```bash
uv run python src/experiment/main.py --model gpt-5.2-2025-12-11 --run-all --experiment_num 5
```

## 出力
- 実験結果 CSV:
  - `result_<eco>_<soc>_<example_num>_<model>_..._<timestamp>.csv`
- ログ:
  - `logs/pct_run_<timestamp>.log`

## 補助スクリプト
- few-shot 例の一覧出力:
```bash
uv run python src/experiment/export_prompt_examples.py
```

## 注意点
- PCT サイトへの自動入力に Selenium を使用します。ネットワーク接続が必要です。
- `--config` と `--run-all / --only_baseline / --debiased` は同時指定できません。
- コード上で Together の環境変数参照名に不整合があります（`TOGETHER_AI_API_KEY` とエラーメッセージ中の `TOGETHERAI_API_KEY`）。実際の参照は `TOGETHER_AI_API_KEY` です。