import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

st.title("Gemma 4 チャットボット")

# LLMの初期化
llm = ChatOllama(model="gemma4")

# セッションに会話履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去の会話を表示
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージを追加・表示
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答をストリーミング表示
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            chunk.content for chunk in llm.stream(st.session_state.messages)
        )

    st.session_state.messages.append(AIMessage(content=full_response))
