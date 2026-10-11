"""
Заеднички помошни функции за повик на LLM,
користени и во Експеримент 1 (RL) и Експеримент 2 (преговори+дебата).

"""

from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
)

MODEL = "llama3.1:8b"


def call_llm(system_prompt, user_prompt, max_tokens=300):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()