"""
streamlit_tools.py

Shared UI helpers for the Codebase Metrics / Analysis app.

Author: Pavan R
"""

import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


# -------- App-wide CSS ---------
def app_css() -> None:
    """
    Injects global CSS styling for the app.
    """
    st.markdown(
        """
        <style>
        html, body, table, th, td {
            font-family: "San Francisco", "SF Pro Display", "SF Pro Icons",
                         "Helvetica Neue", Arial, sans-serif !important;
            font-feature-settings: "liga" on, "kern" on;
        }

        /* Table styling */
        table, th, td {
            background-color: #000000 !important;
            color: #F3F6FB !important;
            border-color: #2A2A2A !important;
        }
        thead th {
            background-color: #141414 !important;
            color: #FFD700 !important;
        }
        table, .stDataFrame table, .stChatMessage table {
            border-radius: 18px !important;
            overflow: hidden !important;
            margin-bottom: 1em;
        }

        /* Heading styling */
        h1, h2, h3, h4, h5, h6 {
            color: #80DEEA !important;
            letter-spacing: 0.3px;
            font-weight: 700;
            border: none;
        }

        hr {
            border-top: 2px solid #2A2A2A !important;
            margin-top: 16px;
            margin-bottom: 16px;
        }

        /* Feedback styling */
        .feedback-row {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-top: 8px;
            margin-bottom: 10px;
        }
        .feedback-label {
            font-weight: bold;
            font-size: 16px;
        }
        .feedback-btn {
            font-size: 16px;
            padding: 6px 15px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .feedback-summary {
            background-color: #f2f7ff;
            border-radius: 6px;
            border: 1px solid #b6cfff;
            padding: 7px 12px;
            color: #034694;
            font-size: 15px;
            margin-left: 10px;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_answer(agent_response: Any) -> str:
    """
    Extracts the assistant's answer from a response object.

    Supports:
    - Plain string responses.
    - Dicts with a 'content' field.
    - Objects with a .content attribute.
    - Lists of messages (dict or objects with role/content).

    Returns a human-readable string, or a diagnostic message if extraction fails.
    """
    try:
        # Most common case: LLM orchestration returns a string
        if isinstance(agent_response, str):
            return agent_response

        # Dict with 'content' (LLM wrapper or tool output)
        if isinstance(agent_response, dict) and "content" in agent_response:
            return str(agent_response["content"])

        # Object with 'content' attribute
        if hasattr(agent_response, "content"):
            return str(agent_response.content)

        # Legacy: list of messages or outputs
        if isinstance(agent_response, list) and agent_response:
            assistant_msgs = []
            for msg in agent_response:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_msgs.append(msg)
                elif hasattr(msg, "role") and getattr(msg, "role") == "assistant":
                    assistant_msgs.append(msg)

            if assistant_msgs:
                last = assistant_msgs[-1]
                if isinstance(last, dict):
                    return str(last.get("content", "No response."))
                if hasattr(last, "content"):
                    return str(last.content)
                return str(last)

            # Fallback: use last message in the list
            last = agent_response[-1]
            if isinstance(last, dict):
                return str(last.get("content", last))
            if hasattr(last, "content"):
                return str(last.content)
            return str(last)

        # Explicit None
        if agent_response is None:
            return "No response."

        # Fallback for unexpected types
        return f"Unknown response type: {type(agent_response)}"
    except Exception as e:
        return f"Error extracting answer: {e}"


# -------- Network / Environment helpers ---------
def get_local_ip() -> str:
    """
    Returns a best-effort local IP address for 'how to connect' instructions.

    Falls back to 'localhost' if detection fails.
    """
    ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # This address doesn't need to be reachable; it's used to pick an interface.
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        pass
    return ip


# -------- Sidebar / Navigation ---------
def sidebar(img_path: str) -> str:
    """
    Renders the sidebar with logo, navigation, and About info.

    Returns:
        The currently selected menu item (e.g., 'Codebase Chatgpt', 'About').
    """
    with st.sidebar:
        # --- Logo and App Title ---
        if img_path:
            try:
                st.image(img_path, use_container_width=True)
            except Exception:
                st.warning("Logo image could not be loaded.")
        st.markdown(
            "<h5 style='text-align:center; color:#12325A; margin-top:-10px; "
            "margin-bottom:18px;'>CNSS</h5>",
            unsafe_allow_html=True,
        )

        # --- Navigation Section ---
        st.markdown("### 🗂️ Navigation")
        menu = st.radio(
            "Main Menu",  # Non-empty label avoids Streamlit warnings
            ("Codebase Chatgpt", "About"),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            "<hr style='margin-top: 4px; margin-bottom: 10px;'>",
            unsafe_allow_html=True,
        )

        # --- About Section ---
        if menu == "About":
            net_ip = get_local_ip()
            st.markdown(
                f"""
                <hr>
                <p style='text-align: center; font-size: 15px; color: #F3F6FB;'>
                    <b>Codebase Analysis Dashboard</b><br>
                    Executive analytics for your codebase metrics and health.<br>
                    <b>Contact:</b> sendpavanr@gmail.com
                </p>
                <hr>
                <div style='font-size:13px; color: #FFD700; text-align:center;'>
                    <b>How to access this dashboard:</b><br>
                    On this machine: <a href="http://localhost:8502" target="_blank">
                    http://localhost:8502</a><br>
                    On another device on the same network:
                    <a href="http://{net_ip}:8502" target="_blank">
                    http://{net_ip}:8502</a><br>
                    <i>Note: "0.0.0.0" is a server listening address—not a real URL.<br>
                    Always use "localhost" or your computer's network IP as above.</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return menu


# -------- Chat context helpers ---------
def get_limited_chat_context(
    history: Sequence[Tuple[str, str]],
    summary: str,
    max_turns: int = 25,
) -> List[Dict[str, str]]:
    """
    Creates a list of messages suitable for an LLM chat API.

    - Puts the summarized history (if any) into a single system message.
    - Adds up to the last `max_turns` user + assistant messages.

    Args:
        history: List of (speaker, text) tuples. Speaker is "You" or "Assistant".
        summary: Previous summary text, if any.
        max_turns: Maximum number of recent turns to include.

    Returns:
        A list of dicts with 'role' and 'content' keys.
    """
    context: List[Dict[str, str]] = []
    if summary:
        context.append(
            {
                "role": "system",
                "content": f"Summary of earlier conversation: {summary}",
            }
        )

    for speaker, text in list(history)[-max_turns:]:
        role = "user" if speaker == "You" else "assistant"
        context.append({"role": role, "content": text})

    return context


@st.cache_resource
def process_uploaded_file(uploaded_file: Any) -> Optional[Any]:
    """
    Placeholder for file processing / indexing logic.

    This is intentionally a stub for now; implement your own logic to:
    - Parse the uploaded file.
    - Extract and index its contents for downstream retrieval / QA.

    Returns:
        Processed representation of the uploaded file, if implemented.
    """
    # TODO: Implement actual file processing logic when needed.
    return None


def summarize_chat(
    messages: Sequence[Tuple[str, str]],
    prev_summary: str = "",
) -> str:
    """
    Summarizes a list of (speaker, text) chat tuples.

    Note:
        This is a simple fallback implementation that concatenates text and
        clips to a reasonable length. Replace with an LLM-based summarizer
        via your orchestration layer for better quality.

    Args:
        messages: List of (speaker, text) tuples.
        prev_summary: Existing summary to prepend.

    Returns:
        A concise text summary (may be truncated).
    """
    chat_text = "\n".join(f"{speaker}: {text}" for speaker, text in messages)
    if not chat_text:
        return prev_summary

    full = (prev_summary + "\n" + chat_text).strip()
    return full[:1000]  # Clip to reasonable length


# -------- Feedback helpers ---------
def feedback_toggle_sidebar() -> bool:
    """
    Displays a sidebar toggle for user feedback participation.

    Returns:
        True if the user has opted into feedback; False otherwise.
    """
    # Use st.sidebar.toggle when available (Streamlit ≥ 1.32), else fallback.
    if hasattr(st.sidebar, "toggle"):
        return st.sidebar.toggle(
            "Participate in Feedback? (optional)",
            help=(
                "If enabled, you'll be shown feedback options after each response. "
                "All feedback is voluntary. Feedback and input may be recorded to "
                "improve this tool."
            ),
        )
    return st.sidebar.checkbox(
        "Participate in Feedback? (optional)",
        help=(
            "If enabled, you'll be shown feedback options after each response. "
            "All feedback is voluntary. Feedback and input may be recorded to "
            "improve this tool."
        ),
    )


def feedback_info_if_enabled() -> None:
    """
    Shows an info block if feedback mode is on, or a caption if it's off.
    """
    enabled = st.session_state.get("feedback_mode", False)
    if enabled:
        st.info(
            "📝 **Feedback is optional.**\n\n"
            "**What does 'hallucination' mean?**\n"
            "A hallucination is when the bot says something factually wrong, "
            "makes up data, or invents test results.\n\n"
            "_Notice: If you provide feedback, your input/feedback may be stored "
            "and used to improve this tool._"
        )
    else:
        st.caption("Feedback mode is OFF. No response ratings will be recorded.")


def feedback_widget(
    response_id: int,
    user_message: str,
    bot_response: str,
) -> Optional[Dict[str, Any]]:
    """
    Renders feedback controls (like / dislike / hallucination) for a response.

    Args:
        response_id: Index of the response in the current chat.
        user_message: The corresponding user message.
        bot_response: The assistant's answer.

    Returns:
        A feedback dict if the user provided feedback; otherwise None.
    """
    if not st.session_state.get("feedback_mode", False):
        return None

    col1, col2, col3, col4, col5 = st.columns([4, 3, 4, 4, 10])

    with col1:
        st.markdown(
            "<span style='font-weight:bold; font-size:16px;'>Feedback response:</span>",
            unsafe_allow_html=True,
        )
    with col2:
        liked = st.button("👍 Prefer", key=f"like_{response_id}")
    with col3:
        disliked = st.button("👎 Don't prefer", key=f"dislike_{response_id}")
    with col4:
        halluc = st.checkbox("🤔 Hallucination", key=f"halluc_{response_id}")
    with col5:
        selection: List[str] = []
        if liked:
            selection.append("Prefer")
        if disliked:
            selection.append("Don't prefer")
        if halluc:
            selection.append("Hallucination")

        if selection:
            st.markdown(
                "<span style='background-color:#e5f6fd; border-radius:4px; "
                "padding:4px 8px; color:#0a4a6f; font-size:15px;'>"
                f"<b>Selected:</b> {' | '.join(selection)}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color: #8c8c8c; font-size:14px;'>Selected: None</span>",
                unsafe_allow_html=True,
            )

    if liked or disliked or halluc:
        st.success("Thank you for your feedback! 🙏")
        feedback = {
            "user_message": user_message,
            "bot_response": bot_response,
            "liked": liked,
            "disliked": disliked,
            "hallucination": halluc,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        return feedback

    return None