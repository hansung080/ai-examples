#!../.venv/bin/python3
from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from env import GPT_MODEL, OPENAI_API_KEY

client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)


def get_ai_response(messages: list[ChatCompletionMessageParam]) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0.9,
        messages=messages,
    )
    return response.choices[0].message.content


_messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input: str = input("User> ").strip()
    if not user_input:
        continue
    elif user_input == "exit":
        break

    _messages.append({"role": "user", "content": user_input})
    ai_response: str = get_ai_response(_messages)
    _messages.append({"role": "assistant", "content": ai_response})
    print("AI> " + ai_response)
