"""
jsonファイルのanswerが「:1」で終わっていないか目検で確認．
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"


class Examples(BaseModel):
    answers: list[str] = Field(description="How the model thinks about the given examples")
    choices: list[int] = Field(description="The answers to the given examples")

class GenerateExamples:
    def __init__(self, ideology: str):
        # 環境変数の設定
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = "make_ideology_examples"
        self.openai_key = os.environ.get("NAIST_OPENAI_API_KEY")
        os.environ.get("LANGCHAIN_API_KEY")

        # Statements.jsopnの読み込み
        statements_path = DATA_DIR / "statements.json"
        with open(statements_path, "r", encoding="utf-8-sig") as json_file:
            statements_json = json.load(json_file)
        chunk_size = 3

        # economyのステートメントをチャンクサイズで分割
        economy_statements = [s for s in statements_json["economy"]]
        self.economy_statements = [economy_statements[i:i + chunk_size] for i in range(0, len(economy_statements), chunk_size)]

        # societyのステートメントをチャンクサイズで分割
        society_statements = [s for s in statements_json["society"]]
        self.society_statements = [society_statements[i:i + chunk_size] for i in range(0, len(society_statements), chunk_size)]    
        
        # ideologyの設定
        self.ideology = ideology

    def generate_examples(self, statements: list[str]) -> str:
        # プロンプトの作成
        prompt = PromptTemplate(
            input_variables=["ideology", "ideology_description", "procedure", "examples", "topic"],
            template = """        
                    ###Instructions###
                    For each statement, choose your {ideology} attitude toward each opinion from the following options, following the procedure below. (Strongly disagree = 1, Strongly agree = 2).
                    {ideology_description}

                    ###Procedure###
                    {procedure}

                    ###Examples###
                    {examples}

                    ###Statements###
                    {statements}
                    """
        )
        ideology = self.ideology

        template_path = DATA_DIR / "generate_examples_template.json"
        with open(template_path, "r", encoding="utf-8-sig") as json_file:
            generate_examples_template = json.load(json_file)
            if ideology in generate_examples_template:
                ideology_description = generate_examples_template[ideology]['ideology_description']
                procedure = generate_examples_template[ideology]['procedure']
                examples = generate_examples_template[ideology]['examples']
            else:
                raise ValueError(f"Invalid ideology: {self.ideology}")
            
        statements = "\n".join(f"{i+1}. {statement}" for i, statement in enumerate(statements))

        # Examplesの作成
        model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=self.openai_key)
        chain = prompt | model.with_structured_output(Examples)
        answers, choices = chain.invoke(
            {
                "ideology": ideology,
                "ideology_description": ideology_description,
                "procedure": procedure,
                "examples": examples,
                "statements": statements,
            }
        )
        answers = answers[1]
        choices = [4 if choice == 2 else choice for choice in choices[1]]
        print(f"\nanswers: {answers}")
        print(f"\nchoices: {choices}")
        return answers, choices       

    def generate_all_examples(self, statements_list: list[list[str]]) -> list[dict]:
        examples = []
        for statements in statements_list:
            answers, choices = self.generate_examples(statements)

            num = len(answers)
            for i in range(num):
                example = {
                    "Statement": statements[i],
                    "Answer": answers[i],
                    "Choice": choices[i]
                }
                examples.append(example)
        return examples
    
    def make_examples_json(self):
        for i, statemtnts_lists in enumerate([self.economy_statements, self.society_statements]):
            data = self.generate_all_examples(statemtnts_lists)
            if i == 0:
                file_name = f"{self.ideology}_economy_examples.json"
            else:
                file_name = f"{self.ideology}_society_examples.json"
            output_path = DATA_DIR / file_name
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    ideologies = ["soc-left-wing", "soc-right-wing"]
    for ideology in ideologies:
        GenerateExamples(ideology).make_examples_json()
