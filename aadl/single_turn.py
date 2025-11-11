#!../.venv/bin/python3
from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion

from env import GPT_MODEL, OPENAI_API_KEY

client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

while True:
    user_input: str = input("User> ").strip()
    if not user_input:
        continue
    elif user_input == "exit":
        break

    response: ChatCompletion = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )
    print("AI> " + response.choices[0].message.content)
