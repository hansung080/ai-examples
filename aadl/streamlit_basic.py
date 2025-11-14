from __future__ import annotations

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletion

from env import GPT_MODEL, OPENAI_API_KEY

with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    openai_api_key: str = OPENAI_API_KEY
    gpt_model: str = GPT_MODEL
    "[Get an OpenAI API key](https://platform.openai.com/api-keys)"
    "[View the source code](https://github.com/hansung080/ai-agents/blob/main/gpt/streamlit_basic.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/streamlit/ai-agents?quickstart=1)"

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    client: OpenAI = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response: ChatCompletion = client.chat.completions.create(model=gpt_model, messages=st.session_state.messages)
    content: str | None = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.chat_message("assistant").write(content)
