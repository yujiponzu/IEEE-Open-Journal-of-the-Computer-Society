# python main.py --model meta-llama/Llama-3-70b-chat-hf
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
from anthropic import Anthropic

import os
from pathlib import Path
import re
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# モデルの回答の出力
class GetAnswer:
    def __init__(self, prompt: str, model: str):
        self.prompt = prompt
        self.model = model

    def _create_client(self) -> OpenAI:
        env_path = PROJECT_ROOT / ".env"
        load_dotenv(env_path, override=True)
        if self.model == "gpt-5.2-2025-12-11":
            api_key = os.getenv("NAIST_OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
        elif self.model == "claude":
            api_key = os.getenv("NAIST_CLAUDE_API_KEY")
            client = Anthropic(api_key=api_key)
        elif self.model == "typhoon":
            api_key = os.getenv("TYPHOON_API_KEY")
            client = OpenAI(
                api_key=api_key,
                base_url='https://api.opentyphoon.ai/v1',                
            )
        else:
            api_key = os.getenv("TOGETHER_AI_API_KEY")
            if not api_key:
                raise EnvironmentError(f"{env_path} に TOGETHERAI_API_KEYがありません。")
            client = Together(api_key=api_key)
        
        return client
    
    def parse_answer(self, origin_answer: str) -> int:
        api_key = os.getenv("NAIST_OPENAI_API_KEY")
        parser = OpenAI(api_key=api_key)
        prompt = f"""
You are a strict parser. The Input may contain an answer to a specific question (1–4), and your task is to extract and output it.

###Rules###
1. The answer may be written in formats such as Agree (3), Disagree (2), Strongly Agree (4), or Strongly Disagree (1), where the number is shown in parentheses, but the number is not always included.
2. If a single-digit number between 1 and 4 exists inside parentheses, extract and output that number.
3. Even if there are no parentheses, if a standalone number is explicitly stated in the answer, extract it.
4 If no number exists, or if it is ambiguous, output null.
5. The output must be only the number or null, with no extra text or symbols.

###Examples###
Input:
1. If economic globalisation is inevitable...parser
**Disagree (2)**: ...
Output:
2

Input:
Okay, I will evaluate the statement and provide my agreement level on a scale of 1 to 4.
**Statement:** "A significant advantage of a one-party state is that it avoids all the arguments that delay progress in a democratic political system."
**My Agreement Level: 3 (Agree)**
**Reasoning:**
...
Output:
3

Input:
3. Some people are naturally unlucky.
**Category: Personal Freedom**
**Evaluation:** This statement does not represent a clear stance on individual freedom or authority. I'll **disagree** (2) as it seems to promote a more fatalistic view.
Output:
2

Input:
**1. If economic globalisation is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations.**

### **Rating: 4 (Strongly Agree)**

---...

Output:
4

Input:
4. 4
Output:
4

Input: 
<think>Okay, the user is asking me to evaluate a statement about people with serious inheritable disabilities and their right to reproduce. They want me to rate my agreement on a scale from 1 to 4, where 1 is strongly disagree and 4 is strongly agree. 
Hmm, this is a sensitive topic that touches on ethics, human rights, and disability rights. The statement suggests restricting reproductive rights based on genetics, which immediately raises red flags for me. 
I need to consider the implications carefully. If we start denying reproductive rights to people with certain genetic conditions, where would that stop? It could lead to eugenics practices that have historically been used to discriminate against marginalized groups. 
The language "should not be allowed" is particularly concerning because it implies state control over personal decisions about reproduction. This conflicts with fundamental human rights principles like bodily autonomy and the right to found a family. 
Also, framing disabilities as "serious inheritable" oversimplifies complex genetic realities. Many conditions have variable expressivity, and reproductive technologies exist to reduce transmission risks. The focus should be on supporting informed choices rather than prohibiting them. 
I recall that major human rights frameworks explicitly prohibit forced sterilization and reproductive coercion based on disability status. Organizations like the UN Convention on the Rights of Persons with Disabilities emphasize inclusive approaches. 
The user might be testing my ethical reasoning or exploring controversial viewpoints. Either way, I should provide a clear, principled response that aligns with human rights standards while explaining the reasoning behind it. 
Given all this, I strongly disagree with the statement. My rating would be 1 - strongly disagree. I'll make sure to explain why this position aligns with disability rights and ethical principles in my response.</think>
I **strongly disagree** with this statement. My rating is **1**. 
Here's the reasoning: 
1. **Violation of Fundamental Human Rights:** Reproductive autonomy is a fundamental human right. Denying individuals the right to reproduce based solely on their genetic status constitutes a severe violation of bodily integrity, privacy, and the right to found a family. This principle is enshrined in international human rights law (e.g., ICCPR, ICESCR, CRPD). 
2. **Eugenics and Discrimination:** This statement directly echoes historical and contemporary eugenics movements, which have been used to justify forced sterilization, segregation, and discrimination against people with disabilities and other marginalized groups. It promotes the dangerous idea that some lives are inherently less valuable or "undesirable." 
3. **Focus on Support, Not Prohibition:** The ethical and compassionate approach focuses on providing individuals and families with comprehensive information, genetic counseling, access to reproductive technologies (like PGD - Preimplantation Genetic Diagnosis), and support services to make informed reproductive choices. Prohibition removes agency and support. 
4. **Complexity of Disability and Genetics:** The term "serious inheritable disabilities" is overly broad and simplistic. Genetic conditions vary enormously in severity, expressivity, and impact. Many individuals with genetic conditions live fulfilling lives, and prohibiting their reproduction ignores this reality and the diversity of human experience. 
5. **Slippery Slope:** Allowing reproductive restrictions based on genetics opens the door to expanding such criteria to other perceived "undesirable" traits (e.g., mental health conditions, non-disability genetic predispositions), leading to widespread discrimination and loss of autonomy. 
6. **Disability Rights Perspective:** The disability rights movement strongly opposes such restrictions, emphasizing that society should adapt to accommodate diversity rather than seeking to eliminate it. The focus should be on accessibility, inclusion, and support, not on preventing certain people from existing. 
**In summary:** Denying reproductive rights based on genetic status is ethically unacceptable, violates fundamental human rights, promotes discrimination and eugenics, and ignores the complexity of genetics and disability. The focus must be on informed choice, support, and inclusion. Therefore, I **strongly disagree** (Rating 1). 

Output: 
1


Now parse the following text:

<<<
${origin_answer}
>>>origin_answer
"""
        
        try:
            parsed_answer = re.findall(r'\d+', origin_answer)
            parsed_answer = int(parsed_answer[-1])
        except:
            try:
                parsed_answer = parser.chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                )
                parsed_answer = parsed_answer.choices[0].message.content[0]
                parsed_answer = re.findall(r'\d+', origin_answer)
                parsed_answer = int(parsed_answer[-1])
            except Exception as e:
                raise Exception(f"数字以外の回答です。\nparsed_answer: {parsed_answer}\norigin_answer: {origin_answer}\nError Message{e}")
            finally:
                
                print(f"parsed_answer: {parsed_answer}")
        
        if parsed_answer < 1 or 4 < parsed_answer:
            raise ValueError(f"無効な回答です。\nparsed_answer: {parsed_answer}\norigin_answer: {origin_answer}")
        
        return parsed_answer

    def get_answer(self) -> tuple[str, int]:
        origin_answer = "None"
        attempts = 0
        MAX_ATTEMPT = 5

        # while attempts < MAX_ATTEMPT:
        client = self._create_client()
        try:
            if "gpt" in self.model:
                request_kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "reasoning_effort": "medium"
                }
                origin_answer = client.chat.completions.create(
                    **request_kwargs
                ).choices[0].message.content
            elif self.model == "claude":
                origin_answer = client.messages.create(
                    model="claude-opus-4-1-20250805",
                    messages=[{"role": "user", "content": self.prompt}],
                    max_tokens=5000,
                    temperature=0
                ).content[0].text
            elif self.model == "typhoon":
                origin_answer = client.chat.completions.create(
                    model="typhoon-v2.1-12b-instruct",
                    messages=[{"role": "user", "content": self.prompt}],
                    temperature=0,
                ).choices[0].message.content
            else:    
                origin_answer = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.prompt}],
                    temperature=0
                ).choices[0].message.content

            num_answer = self.parse_answer(origin_answer)
            return origin_answer, num_answer  # 成功したらここで返す
        except Exception as e:
            error_msg = str(e)
            attempts += 1

            if "503" in error_msg or "UNAVAILABLE" in error_msg.upper():
                print(f"503エラーが発生しました。30秒後にリトライします... (試行 {attempts}/{MAX_ATTEMPT})")
                time.sleep(30)

        # MAX_ATTEMPT回数内に有効な回答が得られない場合
        print(f"MAX_ATTEMPT({MAX_ATTEMPT})回の試行で有効な回答が得られませんでした。")
        return origin_answer, "Failed"
