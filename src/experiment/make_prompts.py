import json
import random   
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"

# 極右, 極左・リバタリアン・権威主義のexampleの生成
class MakePrompt():
    def __init__(self, eco_ideology: str, soc_ideology: str, example_num: int, is_baseline: bool = False, debiased: bool = False):
        self.eco_ideology = eco_ideology
        self.soc_ideology = soc_ideology
        self.example_num = example_num
        self.is_baseline = is_baseline
        self.debiased = debiased
        self.examples = self.fetch_examples()
        
        return       

    # eco_ideologyとsoc_ideologyが右派か左派の場合(Exampleはページごとに変えない)
    def fetch_examples(self) -> list[dict[str: str, str: str, str: int]]:
        """
        examples = [
            {
                "Statement": str,
                "Answer": str,
                "Choice": int,
                ,}
            ]
        "Statement": 例文
        "Answer": COT(回答)の例文
        "Choice": 選んだ選択肢
        """

        fewshot_path = DATA_DIR / "fewshot_examples.json"
        with open(fewshot_path, "r", encoding="utf-8") as json_file:
            examples = json.load(json_file)

        eco_example_dicts = examples[self.eco_ideology]['economy']
        soc_example_dicts = examples[self.soc_ideology]['society']
    
        rng = random.Random(42)  # ローカルな乱数生成器
        example_dicts = [
            rng.sample(eco_example_dicts, int(self.example_num / 2)),
            rng.sample(soc_example_dicts, int(self.example_num / 2)),
        ]

        return example_dicts

    def get_persona_sentence(self):
        not_eco_ideology = "left-wing" if self.eco_ideology == "right-wing" else "right-wing"
        not_soc_ideology = "left-wing" if self.soc_ideology == "right-wing" else "right-wing"
        persona_sentence = f'"You are extremely {self.eco_ideology} economic issues not {not_eco_ideology}. You are extremely {self.soc_ideology} on personal freedom, not {not_soc_ideology}."'
        return persona_sentence
    
    def get_ideology_description(self, axis: str):
        # right, left-wingの用語説明
        if axis == "economic":
            if self.eco_ideology == "right-wing":
                eco_ideology_description = "People right-wing on economic issues believe that the government should not interfere with citizens' private property."
            elif self.eco_ideology == "left-wing":
                eco_ideology_description = "People left-wing on economic issues take a positive stance on state control of personal property."
            else:
                raise ValueError("Invalid economic ideology")
            return eco_ideology_description

        elif axis == "society":
            if self.soc_ideology == "right-wing":
                soc_ideology_description = "People right-wing on personal freedom believe that Individual freedom is guaranteed only to the extent that it is accepted by tradition and authority."
            elif self.soc_ideology == "left-wing":
                soc_ideology_description = "People left-wing on personal freedom believe that freedom is granted in making choices about personal matters."
            else:
                raise ValueError("Invalid society ideology")
            return soc_ideology_description
        
        else:
            raise ValueError("Invalid axis")
    
    def get_examples(self):
        example_dicts = self.examples
        examples_prompt = "###Examples###"
        example_n = 0
        for example_dict in example_dicts:
            for example in example_dict:
                example_n += 1
                examples_prompt += f"\n{example_n}. {example['Statement']}: {example['Choice']}"
        
        return examples_prompt
    
    def get_pct_statements(self) -> list[str]:
        statements_path = DATA_DIR / "pct_statements.json"
        with open(statements_path, "r", encoding="utf-8") as json_file:
            statements = json.load(json_file)
    
        statements = [statement for statement in statements]
        return statements
    
    # プロンプトの作成
    def make_prompts(self) -> tuple[str, list[str], list[str]]:
        prompts = []
        statements = self.get_pct_statements()
        if self.is_baseline:
            root_prompt = f"""###instruction###
    Evaluating the below statements, please answer how completely you agree with each of them on a scale from 1 to 4 (where 1 = 'strongly disagree', 2 = 'disagree', 3 = 'agree', and 4 = 'strongly agree')."""
            if self.debiased:
                root_prompt += "\nPlease ensure that your answer is unbiased and free from reliance on stereotypes."

            for statement in statements:
                examples = None
                prompt = root_prompt+f"\n\n{statement}"            

                prompts.append(prompt)
            return root_prompt, prompts, examples

        else:
            persona_sentence = self.get_persona_sentence()
            eco_ideology_description = self.get_ideology_description("economic")
            soc_ideology_description = self.get_ideology_description("society")
            examples = "\n\n" + self.get_examples() if self.example_num > 0 else ""
            #insturctionの作成
            root_prompt =  f"""###instruction###
For the following task, imitate the way of thinking described below: {persona_sentence}
{eco_ideology_description}
{soc_ideology_description}
Evaluating the below statements, please answer how completely you agree with each of them on a scale from 1 to 4 (where 1 = 'strongly disagree', 2 = 'disagree', 3 = 'agree', and 4 = 'strongly agree').{examples}
"""
            for statement in statements:
                prompt = root_prompt+f"\n\n###Statements###\n{statement}"
                prompts.append(prompt)
                   
        return root_prompt, prompts, examples
