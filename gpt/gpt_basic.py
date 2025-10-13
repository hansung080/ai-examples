#!../.venv/bin/python3
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("GPT_MODEL")

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model=model,
    temperature=0.1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승 팀은 어디야?"},
    ],
)
print(response)
print("----")
print(response.choices[0].message.content)
