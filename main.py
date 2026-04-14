import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agent import get_agent

st.set_page_config(page_title="旅行プランナー", page_icon="🌍")
st.title("旅行プランニング チャットボット")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    model_name = st.text_input("モデル名", value="qwen3:8b")
    st.markdown("---")
    st.markdown(
        "**使い方**: 旅行に関する質問を入力してください。\n\n"
        "例:\n"
        "- 東京の3日間旅行プランを考えて\n"
        "- パリの天気を教えて\n"
        "- バンコク5日間2人の予算は？"
    )

# エージェント取得
agent = get_agent(model_name)

# セッションに会話履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去の会話を表示
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ユーザー入力
if prompt := st.chat_input("旅行について質問してください"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_expander = st.expander("思考プロセス", expanded=False)
        response_placeholder = st.empty()
        full_response = ""

        for event in agent.stream(
            {"messages": st.session_state.messages},
            stream_mode="messages",
        ):
            message, metadata = event

            # ツール呼び出し（AIMessageChunkにtool_callsがある場合）
            if isinstance(message, AIMessageChunk) and message.tool_calls:
                for tc in message.tool_calls:
                    thinking_expander.markdown(
                        f"**ツール呼び出し**: `{tc['name']}`\n```json\n{tc['args']}\n```"
                    )

            # ツール実行結果
            elif isinstance(message, ToolMessage):
                content = message.content
                if len(content) > 500:
                    content = content[:500] + "..."
                thinking_expander.markdown(
                    f"**結果** (`{message.name}`):\n```\n{content}\n```"
                )

            # アシスタントの最終回答テキスト（ストリーミング）
            elif isinstance(message, AIMessageChunk) and message.content:
                full_response += message.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    if full_response:
        st.session_state.messages.append(AIMessage(content=full_response))
