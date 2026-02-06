import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from agents.codebase_analysis_chat_agent import (
    CodebaseAnalysisSessionState,
    CodebaseAnalysisOrchestration,
)
from utils.parsers.env_parser import EnvConfig
import streamlit_tools as st_tools
import re
import io
import json


def render_markdown_tables_with_dataframes(markdown_str):
    st.markdown(markdown_str)
    table_pattern = r"(\|[^\n]+\|\n(\|[:\-]+\|)+\n(?:\|.*\|\n?)+)"
    tables = re.findall(table_pattern, markdown_str)
    for ttuple in tables:
        md_table = ttuple[0]
        try:
            df = pd.read_csv(io.StringIO(md_table), sep='|', engine='python')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            st.dataframe(df, use_container_width=True)
        except Exception as ex:
            st.info(f"(Could not parse markdown table to DataFrame: {ex})")


def extract_content(answer):
    if isinstance(answer, dict) and "content" in answer:
        return answer["content"]
    elif isinstance(answer, str):
        astr = answer.strip()
        if (astr.startswith("{") and astr.endswith("}")) or (astr.startswith('{"')):
            try:
                parsed = json.loads(astr)
                if isinstance(parsed, dict) and "content" in parsed:
                    return parsed["content"]
                else:
                    return answer
            except Exception:
                return answer
        else:
            return answer
    else:
        return str(answer)


env_config = EnvConfig()
STREAMLIT_MODEL = env_config.get("STREAMLIT_MODEL") or "qwen2.5-14b-1m"
PLACEHOLDER = "🤖 _Generating response..._"

if not isinstance(STREAMLIT_MODEL, str) or not STREAMLIT_MODEL.strip():
    st.error("Model configuration invalid or missing.")
    st.stop()

img_path = os.path.join(os.path.dirname(__file__), "qualcomm_logo.png")
if not os.path.isfile(img_path):
    st.warning("Logo image not found at expected path.")

st.set_page_config(
    page_title="CODEBASE METRICS GENIE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

for key, value in [
    ("chat_history_chat", []),
    ("chat_history_chat_summary", ""),
    ("chat_history", []),
    ("all_feedback", []),
    ("feedback_mode", False),
    ("debug_mode", False),
]:
    if key not in st.session_state:
        st.session_state[key] = value


# ------------ AGENT CACHE --------------
@st.cache_resource
def get_orchestrator():
    return CodebaseAnalysisOrchestration()


orchestrator = get_orchestrator()


def show_about():
    st.info("Qualcomm Codebase Metrics & Health Analysis powered by Genie.")
    st.stop()


def show_faq():
    st.markdown("## FAQ")
    FAQS = [
        (
            "What can I ask?",
            "Ask about code health metrics, module dependencies, security, documentation, "
            "test coverage, and refactoring recommendations.",
        ),
        (
            "What data does this use?",
            "It uses precomputed codebase analysis reports, dependency graphs, security metrics, "
            "test coverage data, and other health metrics.",
        ),
    ]
    query = st.text_input("Search FAQs...")
    for q, a in FAQS:
        if query.lower() in q.lower() or query.lower() in a.lower() or not query:
            st.markdown(f"**Q:** {q}  \n  **A:** {a}")
    st.stop()


def show_codebase_chat():
    st.markdown(
        "<h1 style='text-align:center; margin-top: -10px;'>Codebase Metrics Chat</h1>",
        unsafe_allow_html=True,
    )
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "<b>Welcome to <span style='color:#FFD700;'>Codebase Metrics & Health Chat</span>!</b><br>"
            "Ask me questions about your codebase's dependencies, quality, complexity, maintainability, "
            "documentation, test coverage, and security.",
            unsafe_allow_html=True,
        )

    st_tools.feedback_info_if_enabled()

    sample_queries = [
        {
            "title": "Module Details",
            "prompt": (
                "Show me all details about the auth module, including its dependencies, "
                "documentation, security risks, and test coverage."
            ),
        },
        {
            "title": "Overall Health",
            "prompt": (
                "Summarize the overall health of the codebase across security, quality, "
                "complexity, maintainability, and test coverage. Highlight the top 3 issues "
                "I should fix first."
            ),
        },
    ]

    with st.expander("What can I ask?"):
        for q in sample_queries:
            st.markdown(f"- **{q['title']}**: {q['prompt']}")

    if "chat_history_chat" not in st.session_state:
        st.session_state.chat_history_chat = []
    if "chat_history_chat_summary" not in st.session_state:
        st.session_state.chat_history_chat_summary = ""

    if st.session_state.chat_history_chat_summary:
        st.info("Earlier conversation summary: " + st.session_state.chat_history_chat_summary)
        st.download_button(
            "Download Conversation Summary",
            st.session_state.chat_history_chat_summary,
            file_name="codebase_chat_summary.txt",
        )

    # Render existing chat history
    for idx, (speaker, text) in enumerate(st.session_state.chat_history_chat):
        role = "user" if speaker == "You" else "assistant"
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                render_markdown_tables_with_dataframes(extract_content(text))
            else:
                st.markdown(text)

            if role == "assistant":
                user_msg = ""
                if idx > 0 and st.session_state.chat_history_chat[idx - 1][0] == "You":
                    user_msg = st.session_state.chat_history_chat[idx - 1][1]
                feedback = st_tools.feedback_widget(idx, user_msg, text)
                if feedback:
                    st.session_state["all_feedback"].append(feedback)

    # Summarize earlier turns if too long
    max_turns = 25
    history = st.session_state.chat_history_chat
    if len(history) > max_turns:
        old_messages = history[:-max_turns]
        prev_summary = st.session_state.chat_history_chat_summary
        new_summary = st_tools.summarize_chat(old_messages, prev_summary)
        st.session_state.chat_history_chat_summary = new_summary
        st.session_state.chat_history_chat = history[-max_turns:]

    # New user input
    user_input = st.chat_input("Ask about your codebase's health and metrics:")
    if user_input:
        st.session_state.chat_history_chat.append(("You", user_input))
        st.session_state.chat_history_chat.append(("Assistant", PLACEHOLDER))
        st.rerun()

    # If the last assistant message is still the placeholder, call the orchestrator
    if (
        st.session_state.chat_history_chat
        and st.session_state.chat_history_chat[-1] == ("Assistant", PLACEHOLDER)
    ):
        user_message = (
            st.session_state.chat_history_chat[-2][1]
            if len(st.session_state.chat_history_chat) >= 2
            else ""
        )
        with st.spinner("Assistant is thinking..."):
            try:
                state = CodebaseAnalysisSessionState(user_input=user_message)
                state = orchestrator.run_multiturn_chain(state)
                answer = extract_content(state.formatted_response)
            except Exception as e:
                st.error(
                    f"Error: {e}. Try asking about code health metrics, module dependencies, "
                    "security issues, documentation gaps, or refactoring recommendations."
                )
                answer = "No response."

        st.session_state.chat_history_chat[-1] = ("Assistant", answer)
        st.rerun()

    # Download chat history
    if st.session_state.get("chat_history_chat"):
        st.download_button(
            "Download Chat History",
            "\n".join(
                f"{speaker}: {extract_content(text) if speaker != 'You' else text}"
                for speaker, text in st.session_state["chat_history_chat"]
            ),
            file_name="codebase_metrics_chat_history.txt",
        )

    st.stop()


if __name__ == "__main__":
    st_tools.app_css()
    st.markdown(
        "<h1 style='text-align:center; margin-top: -10px; margin-bottom: 16px;'>CODEBASE Metrics Chat</h1>",
        unsafe_allow_html=True,
    )
    menu = st_tools.sidebar(img_path)
    if menu == "About":
        show_about()
    elif menu == "FAQ":
        show_faq()
    elif menu == "Codebase Chatgpt":  # keep key for backward compatibility
        show_codebase_chat()