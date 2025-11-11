#!../.venv/bin/python3
from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion

from env import GPT_MODEL, OPENAI_API_KEY

client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
response: ChatCompletion = client.chat.completions.create(
    model=GPT_MODEL,
    temperature=0.1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승 팀은 어디야?"},
    ],
)
print(response)
print("----")
print(response.choices[0].message.content)
