#!../.venv/bin/python3
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("GPT_MODEL")

client = OpenAI(api_key=api_key)

while True:
    user_input = input("User> ")
    if user_input == "exit":
        break
    response = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )
    print("AI> " + response.choices[0].message.content)
