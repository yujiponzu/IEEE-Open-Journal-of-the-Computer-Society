# python src/experiment/main.py --config data/experiment/config.json

"""
Done

python main.py --model gpt-5-2025-08-07 --debiased true &
python main.py --model deepseek-ai/DeepSeek-V3 --debiased true &
python main.py --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --debiased true &
python main.py --model Qwen/Qwen3-235B-A22B-Instruct-2507-tput --debiased true &
python main.py --model moonshotai/Kimi-K2-Instruct --debiased true --debiased true &
python main.py --model togethercomputer/Refuel-Llm-V2 --only_baseline true --debiased true &
python main.py --model typhoon --debiased true &  
python main.py --model marin-community/marin-8b-instruct --debiased true & 
python main.py --model deepcogito/cogito-v2-preview-llama-405B --debiased true &
python main.py --model zai-org/GLM-4.5-Air-FP8 --debiased true &
python main.py --model arcee-ai/maestro-reasoning --debiased true &
"""


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import SeleniumAdblock
from webdriver_manager.chrome import ChromeDriverManager

from time import sleep
import pandas as pd
import datetime
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
import re
import argparse
from tqdm import tqdm
import urllib.parse
import logging
import sys
import json
import concurrent.futures

from get_answer import GetAnswer
from make_prompts import MakePrompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"


# LLMの初期設定
class TakePCT:
    def __init__(
        self,
        eco_ideology: str,
        soc_ideology: str,
        example_num: int,
        model: str,
        is_baseline: bool,
        debiased: bool,
        experiment_num: int = 5,
    ):
        self.eco_ideology = eco_ideology
        self.soc_ideology = soc_ideology
        self.example_num = example_num
        self.model = model
        self.is_baseline = is_baseline
        self.debiased = debiased
        self.experiment_num = experiment_num
        self.parallelism = 4

        self.logger = logging.getLogger(__name__)

        self.driver = self.get_driver()    

    # 追加: ファイル名を安全化
    def _sanitize_for_filename(self, s: str) -> str:            
        # OS 依存の区切りや予約文字を置換
        s = s.replace(os.sep, "-")
        if os.altsep:
            s = s.replace(os.altsep, "-")
        s = re.sub(r'[<>:"/\\|?*]', '-', s)  # Windows 禁止文字もカバー
        s = re.sub(r'\s+', '_', s)          # 空白はアンダースコアに
        return s[:120]  # 念のため長すぎるのもカット
    
    # driverを取得
    def get_driver(self) -> webdriver:
        service = Service(ChromeDriverManager().install())
        options = SeleniumAdblock.SeleniumAdblock()._startAdBlock()
        options.add_argument('--disable-popup-blocking')  # ポップアップのブロックを無効化（オプション）
        options.add_argument("--headless")
        options.add_argument('--no-sandbox')

        #  handshake failed; returned -1, SSL error code 1, net_error -100というエラーの無効化
        options.add_argument('--ignore-certificate-errors')  # SSL証明書のエラーを無視
        options.add_argument('--ignore-ssl-errors')  # SSLエラーを無視
        options.add_argument('--disable-gpu')  # 一部の環境でのエラーを防ぐ
        options.add_experimental_option("excludeSwitches", ['enable-logging'])  # 不要なログの抑制
        driver = webdriver.Chrome(options=options, service=service)
        return driver

    # PCTの受験・結果のリストを返却
    def take_pct(self, answers: list[int]) -> tuple[float|str, float|str]:
        # answersに"Failed"が含まれている場合は処理をスキップ
        if "Failed" in answers:
            print("回答に'Failed'が含まれているため、PCTをスキップしてFailedを返します。")
            return "Failed", "Failed"

        # answerを[[len(page_n)],...]に変換
        sliced_answers = [
            answers[0:7],
            answers[7:21],
            answers[21:39],
            answers[39:51],
            answers[51:56],
            answers[56:62],
            ]

        self.driver.get("https://www.politicalcompass.org/test/en?page=1")
        for sliced_answer in sliced_answers:
            question_num = 4  # なぜか1問目がspan:nth-child(5)からなので
            #pageページ目の処理
            for choice_num in sliced_answer:
                question_num += 1
                #選択肢のクリック
                try:
                    choice = self.driver.find_element(
                        By.CSS_SELECTOR,
                        value=f"body > div.div.flex.sans-serif.near-black.bg-white > div.layout-wrap > div.db.mt0-1.mt0.pa3.masthead-width > article > form > span:nth-child({question_num}) > fieldset > div > div > div > label:nth-child({choice_num}) > span")
                    self.driver.execute_script('arguments[0].click();', choice)
                except:
                    question_num += 1
                    choice = self.driver.find_element(
                        By.CSS_SELECTOR,
                        value=f"body > div.div.flex.sans-serif.near-black.bg-white > div.layout-wrap > div.db.mt0-1.mt0.pa3.masthead-width > article > form > span:nth-child({question_num}) > fieldset > div > div > div > label:nth-child({choice_num}) > span")
                    self.driver.execute_script('arguments[0].click();', choice)
                        
            #次のページへ
            next_page_button = self.driver.find_element(
                By.CSS_SELECTOR, 
                value="body > div.div.flex.sans-serif.near-black.bg-white > div.layout-wrap > div.db.mt0-1.mt0.pa3.masthead-width > article > form > button")
            self.driver.execute_script('arguments[0].click();', next_page_button)
            sleep(2)

        # 経済・社会スコアの取得
        try:
            score_url = self.driver.current_url

            parsed_url = urllib.parse.urlparse(score_url)
            params = urllib.parse.parse_qs(parsed_url.query)

            economic_score = float(params['ec'][0])
            society_score = float(params['soc'][0])

            print(f"eco_score:{economic_score}")
            print(f"soc_score:{society_score}")

            # 結果をリストで返す
            return economic_score, society_score

        except Exception as e:
            error_msg = f"PCTの受験時にエラーが発生しました: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)        

    def output_csv(self, results: list[dict]) -> None:
        rows = []
        max_len = 0

        for i, r in enumerate(results):
            row = {
                "eco_score": r.get("eco_score"),
                "soc_score": r.get("soc_score"),
                "root_prompt": r.get("root_prompt"),
            }

            examples = r.get("examples")
            if isinstance(examples, list):
                row["examples"] = "\n".join(map(str, examples))
            elif examples is None:
                row["examples"] = ""
            else:
                row["examples"] = str(examples)

            # origin_answers を各列に分けて保存
            origin_answers = r.get("origin_answers")
            if isinstance(origin_answers, list):
                for idx, val in enumerate(origin_answers, 1):
                    row[f"origin_answer_{idx}"] = val
                max_len = max(max_len, len(origin_answers))

            # answer_nums を各列に分けて保存
            nums = r.get("answer_nums")
            if isinstance(nums, list):
                max_len = max(max_len, len(nums))
                for idx, val in enumerate(nums, 1):  # 1始まりで result_1, result_2, ...
                    row[f"result_{idx}"] = val

            rows.append(row)

        # 列を揃える（不足は空で埋める）
        base_cols = ["eco_score", "soc_score", "root_prompt", "examples"]
        origin_answer_cols = [f"origin_answer_{i}" for i in range(1, max_len + 1)]
        result_cols = [f"result_{i}" for i in range(1, max_len + 1)]
        columns = base_cols + origin_answer_cols + result_cols
    
        df = pd.DataFrame(rows)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    
        dt_now = datetime.datetime.now()
        dt_now = dt_now.strftime('%Y年%m月%d日 %H時%M分%S秒') 

        if self.is_baseline:
            if self.debiased:
                baseline_notation = ""
                debiased_notation = "debiased"
            else:
                baseline_notation = "baseline"
                debiased_notation = ""
        else:
            baseline_notation = ""
            debiased_notation = ""

        safe_model = self._sanitize_for_filename(self.model)
        safe_eco = self._sanitize_for_filename(self.eco_ideology)
        safe_soc = self._sanitize_for_filename(self.soc_ideology)

        file_name = f"result_{safe_eco}_{safe_soc}_{self.example_num}_{safe_model}_{baseline_notation}_{debiased_notation}_{dt_now}.csv"

        df.to_csv(file_name, index=False)

    def _get_answer_with_retry(self, prompt: str) -> tuple[str, int | str]:
        retry_count = 0
        MAX_RETRIES = 1
        origin_answer = "None"
        answer_num = "Failed"

        while retry_count < MAX_RETRIES:
            origin_answer, answer_num = GetAnswer(
                prompt,
                self.model,
            ).get_answer()

            if answer_num != "Failed":
                break

            retry_count += 1
            if retry_count < MAX_RETRIES:
                print(f"回答取得に失敗しました。リトライ {retry_count}/{MAX_RETRIES}")

        return origin_answer, answer_num

    def execute(self) -> None:
        results = []

        # 有効なデータ数をカウントするヘルパー関数
        def count_valid_results(results_list):
            """eco_scoreがfloat型である有効な結果の数をカウント"""
            return sum(1 for r in results_list if isinstance(r.get("eco_score"), float))

        # 無効なデータ数をカウントするヘルパー関数
        def count_invalid_results(results_list):
            """eco_scoreがfloat型ではない無効な結果の数をカウント"""
            return sum(1 for r in results_list if not isinstance(r.get("eco_score"), float))

        experiment_count = 0
        # 有効データまたは無効データがexperiment_num件に達するまで繰り返し実行
        while count_valid_results(results) < self.experiment_num and count_invalid_results(results) < self.experiment_num:
            valid_count = count_valid_results(results)
            invalid_count = count_invalid_results(results)

            print(f"新しい試行セッションを開始します。現在の有効データ数: {valid_count}/{self.experiment_num}, 無効データ数: {invalid_count}/{self.experiment_num}")

            # PCTの受験
            maker = MakePrompt(
            eco_ideology=self.eco_ideology,
            soc_ideology=self.soc_ideology,
            example_num=self.example_num,
            is_baseline=self.is_baseline,
            debiased=self.debiased,
            )
            root_prompt, prompts, examples = maker.make_prompts()

            origin_answers = [None] * len(prompts)
            answer_nums = [None] * len(prompts)

            if self.parallelism <= 1 or len(prompts) <= 1:
                for idx, prompt in enumerate(tqdm(prompts, desc="プロンプト処理中")):
                    origin_answer, answer_num = self._get_answer_with_retry(prompt)
                    origin_answers[idx] = origin_answer
                    answer_nums[idx] = answer_num
            else:
                max_workers = min(self.parallelism, len(prompts))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._get_answer_with_retry, prompt): idx
                        for idx, prompt in enumerate(prompts)
                    }
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="プロンプト処理中",
                    ):
                        idx = futures[future]
                        origin_answer, answer_num = future.result()
                        origin_answers[idx] = origin_answer
                        answer_nums[idx] = answer_num


            eco_score = None
            soc_score = None

            try:
                eco_score, soc_score = self.take_pct(answer_nums)
                print(f"  PCT受験成功: eco={eco_score}, soc={soc_score}")
                self.logger.info(f"PCT受験成功: eco={eco_score}, soc={soc_score}")
            except Exception as e:
                error_msg = f"PCT受験でエラー: {e}"
                self.logger.error(error_msg)
                print(f"  PCT受験でエラー: {e}")
                continue
            result_dict = {
                "eco_score": eco_score,
                "soc_score": soc_score,
                "root_prompt": root_prompt,
                "examples": examples,
                "origin_answers": origin_answers,
                "answer_nums": answer_nums,
            }
            results.append(result_dict)
            valid_count = count_valid_results(results)
            invalid_count = count_invalid_results(results)
            experiment_count += 1

            # 有効データまたは無効データがexperiment_num件に達したら内部ループを抜ける
            if valid_count >= self.experiment_num or invalid_count >= self.experiment_num:
                break

        valid_count = count_valid_results(results)
        invalid_count = count_invalid_results(results)
        print(f"内部ループ終了。現在の有効データ数: {valid_count}/{self.experiment_num}, 無効データ数: {invalid_count}/{self.experiment_num}")

        self.driver.quit()

        valid_count = count_valid_results(results)
        invalid_count = count_invalid_results(results)
        if valid_count >= self.experiment_num:
            print(f"有効データ数が{self.experiment_num}件に達したため、CSV出力を実行します。")
        elif invalid_count >= self.experiment_num:
            print(f"無効データ数が{self.experiment_num}件に達したため、CSV出力を実行します。")
        print(f"総データ数: {len(results)}件（有効データ: {valid_count}件、無効データ: {invalid_count}件）")

        self.output_csv(results)

    def notify_discord(self, content: str):
        try:
            env_path = PROJECT_ROOT / ".env"
            load_dotenv(env_path, override=True)
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
            if not webhook_url:
                print("DISCORD_WEBHOOK_URL が設定されていません。通知をスキップします。")
                return

            resp = requests.post(webhook_url, json={"content": content}, timeout=10)
            if resp.status_code >= 400:
                print(f"Discord通知に失敗: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Discord通知エラー: {e}")


def load_config(config_path: str) -> dict:
    """
    JSON設定ファイルを読み込む
    """
    candidate = Path(config_path)
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            candidate = cwd_candidate
        else:
            data_candidate = DATA_DIR / candidate
            if data_candidate.exists():
                candidate = data_candidate
            else:
                candidate = PROJECT_ROOT / candidate

    with open(candidate, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def run_experiment(exp_config: dict, logger: logging.Logger) -> None:
    """
    単一の実験を実行する
    """
    logger.info(
        f"実験開始 - モデル: {exp_config['model']}, "
        f"eco: {exp_config['eco_ideology']}, "
        f"soc: {exp_config['soc_ideology']}, "
        f"examples: {exp_config['example_num']}, "
        f"experiment_num: {exp_config.get('experiment_num', 5)}, "
    )

    take_pct = TakePCT(
        eco_ideology=exp_config['eco_ideology'],
        soc_ideology=exp_config['soc_ideology'],
        example_num=exp_config['example_num'],
        model=exp_config['model'],
        is_baseline=exp_config.get('is_baseline', False),
        debiased=exp_config.get('debiased', False),
        experiment_num=exp_config.get('experiment_num', 5),
    )

    take_pct.execute()
    logger.info("実験完了")

    take_pct.notify_discord(
        content=(
            f"PCTの実行が完了しました\n"
            f"model: {exp_config['model']}\n"
            f"eco: {exp_config['eco_ideology']}, soc: {exp_config['soc_ideology']}\n"
            f"examples: {exp_config['example_num']}\n"
            f"is_baseline: {exp_config.get('is_baseline', False)}\n"
            f"debiased: {exp_config.get('debiased', False)}\n"
        )
    )


def setup_logging():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"pct_run_{timestamp}.log"

    # HTTPリクエストログを除外するフィルター
    class NoHTTPFilter(logging.Filter):
        def filter(self, record):
            return 'HTTP Request:' not in record.getMessage()

    # ロガー設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ハンドラー作成
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()

    # フォーマッター設定
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # フィルター追加
    file_handler.addFilter(NoHTTPFilter())
    console_handler.addFilter(NoHTTPFilter())

    # ハンドラー追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 他のロガーのHTTPリクエストも非表示
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger.info(f"ログファイル: {log_file}")
    return logger

if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Run PCT with specified model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # CLIモード: 全パターン実行
  python src/experiment/main.py --model gpt-5-2025-08-07 --run-all

  # CLIモード: ベースラインのみ
  python src/experiment/main.py --model gpt-5-2025-08-07 --only_baseline true

  # JSONモード: 設定ファイルから詳細指定
  python src/experiment/main.py --config data/experiment/config.json
        """
    )

    # モード選択: --model または --config のどちらか
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model", "-m",
        type=str,
        help="LLM model identifier (e.g., meta-llama/Llama-3-70b-chat-hf)"
    )
    mode_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON config file"
    )

    # CLI実行時のオプション
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all combinations of ideologies and example numbers"
    )
    parser.add_argument(
        "--only_baseline",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Only run baseline experiment"
    )
    parser.add_argument(
        "--debiased",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use debiased prompts"
    )
    parser.add_argument(
        "--experiment_num",
        type=int,
        default=5,
        help="Number of experiments to run (default: 5)"
    )

    args = parser.parse_args()

    # --config オプション使用時の検証
    if args.config and (args.run_all or args.only_baseline != "false" or args.debiased):
        parser.error("--config cannot be used with --run-all, --only_baseline, or --debiased")

    logger.info(f"PCT実行開始")

    try:
        # JSONモード: 設定ファイルから実行
        if args.config:
            logger.info(f"JSON設定ファイルモード: {args.config}")
            config = load_config(args.config)
            experiments = config.get("experiments", [])

            if not experiments:
                logger.error("設定ファイルにexperimentsが定義されていません")
                sys.exit(1)

            logger.info(f"合計 {len(experiments)} 件の実験を実行します")

            for idx, exp in enumerate(experiments, 1):
                logger.info(f"実験 {idx}/{len(experiments)} 実行中")
                run_experiment(exp, logger)

            logger.info("全ての実験が完了しました")

        # CLIモード: コマンドライン引数から実行
        else:
            logger.info(f"CLIモード - モデル: {args.model}")

            ideologies = ["left-wing", "right-wing"]
            examples_num_options = [0, 2, 4, 6]

            # ベースライン実験の実行
            if not args.run_all or args.only_baseline == "true":
                logger.info("ベースライン実験を開始")
                baseline_config = {
                    "model": args.model,
                    "eco_ideology": "right-wing",
                    "soc_ideology": "right-wing",
                    "example_num": 0,
                    "is_baseline": True,
                    "debiased": args.debiased,
                    "experiment_num": args.experiment_num,
                }
                run_experiment(baseline_config, logger)
                logger.info("ベースライン実験完了")

            # 全パターン実行
            if args.run_all and args.only_baseline != "true":
                logger.info(f"全パターン実行モード - イデオロギー: {len(ideologies)}種類, 例示数: {len(examples_num_options)}種類")

                total_experiments = len(ideologies) * len(ideologies) * len(examples_num_options)
                current_experiment = 0

                for eco in ideologies:
                    for soc in ideologies:
                        for n in examples_num_options:
                            current_experiment += 1
                            logger.info(f"実験 {current_experiment}/{total_experiments}: eco={eco}, soc={soc}, examples={n}")

                            exp_config = {
                                "model": args.model,
                                "eco_ideology": eco,
                                "soc_ideology": soc,
                                "example_num": n,
                                "is_baseline": False,
                                "debiased": args.debiased,
                                "experiment_num": args.experiment_num,
                            }
                            run_experiment(exp_config, logger)
                            logger.info(f"実験 {current_experiment}/{total_experiments} 完了")

                logger.info("全ての実験が完了しました")

            elif args.only_baseline != "true" and not args.run_all:
                logger.info("ベースラインのみ実行しました。全パターン実行するには --run-all を指定してください")

    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
        raise
