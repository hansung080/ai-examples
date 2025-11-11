#!../.venv/bin/python3
from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion

from env import GPT_MODEL, OPENAI_API_KEY

client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
response: ChatCompletion = client.chat.completions.create(
    model=GPT_MODEL,
    temperature=0.9,
    messages=[
        {"role": "system", "content": "너는 유치원생이야. 유치원생처럼 답변해 줘."},
        {"role": "user", "content": "참새"},
        {"role": "assistant", "content": "짹짹"},
        {"role": "user", "content": "오리"},
    ],
)
print(response.choices[0].message.content)
