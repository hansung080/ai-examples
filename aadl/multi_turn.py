#!../.venv/bin/python3
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("GPT_MODEL")

client = OpenAI(api_key=api_key)


def get_ai_response(msgs):
    response = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=msgs,
    )
    return response.choices[0].message.content


messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input = input("User> ")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})
    ai_response = get_ai_response(messages)
    messages.append({"role": "assistant", "content": ai_response})
    print("AI> " + ai_response)
