import html
import os
import re
import textwrap
import uuid

import streamlit as st

from rag import (
    DATA_DIR,
    is_follow_up_query,
    stream_answer,
)


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    textwrap.dedent(
        """
        <style>
        :root {
            --bg: #111214;
            --surface: #18191c;
            --surface-2: #202126;
            --border: #2a2c31;
            --text: #f3f4f6;
            --muted: #979ba5;
            --accent: #8b9cff;
            --accent-soft: rgba(139, 156, 255, 0.12);
            --success: #74d69b;
            --warning: #e5bd72;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #0d0e10;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebarContent"] {
            padding-top: 1.25rem;
        }

        .brand {
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 1.35rem;
        }

        .section-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.64rem;
            font-weight: 700;
            margin: 1.15rem 0 0.5rem;
        }

        .hero {
            max-width: 820px;
            margin: 10vh auto 2.2rem auto;
            text-align: center;
        }

        .hero-mark {
            width: 54px;
            height: 54px;
            border-radius: 16px;
            margin: 0 auto 1.15rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--accent-soft);
            border: 1px solid rgba(139, 156, 255, 0.22);
            color: var(--accent);
            font-weight: 800;
            font-size: 1.15rem;
        }

        .hero-title {
            color: var(--text);
            font-size: 2.15rem;
            font-weight: 760;
            letter-spacing: -0.045em;
            margin-bottom: 0.45rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
        }

        .meta-row {
            display: flex;
            gap: 0.4rem;
            margin-top: 0.6rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--muted);
            font-size: 0.7rem;
        }

        .source-card {
            padding: 0.85rem 0;
            border-bottom: 1px solid var(--border);
        }

        .source-title {
            font-size: 0.82rem;
            font-weight: 650;
            margin-bottom: 0.2rem;
        }

        .source-file {
            color: var(--muted);
            font-size: 0.74rem;
            margin-bottom: 0.45rem;
            word-break: break-word;
        }

        .source-score {
            font-size: 0.7rem;
            margin-bottom: 0.55rem;
        }

        .source-score.high {
            color: var(--success);
        }

        .source-score.medium {
            color: var(--warning);
        }

        .source-score.low {
            color: var(--muted);
        }

        .source-snippet {
            color: #c9ccd3;
            font-size: 0.81rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .empty-example {
            max-width: 820px;
            margin: 0 auto;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: transparent;
        }

        .stButton button {
            border-radius: 10px;
        }

        .stChatInputContainer {
            max-width: 820px;
            margin: 0 auto;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border: 0;
            padding-top: 0.7rem;
            padding-bottom: 0.7rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 0.96rem;
            line-height: 1.75;
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


def init_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = {
            "default": {
                "title": "New chat",
                "messages": [],
            }
        }

    if "active_session" not in st.session_state:
        st.session_state.active_session = "default"

    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    if "editing_message_id" not in st.session_state:
        st.session_state.editing_message_id = None

    if "show_performance" not in st.session_state:
        st.session_state.show_performance = True

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


def new_session():
    session_id = str(uuid.uuid4())

    st.session_state.sessions[session_id] = {
        "title": "New chat",
        "messages": [],
    }

    st.session_state.active_session = session_id
    st.session_state.pending_action = None
    st.session_state.editing_message_id = None


def get_active_session():
    return st.session_state.sessions[
        st.session_state.active_session
    ]


def safe_title(text):
    text = " ".join(text.strip().split())

    if len(text) <= 38:
        return text

    return text[:35] + "..."


def set_pending_prompt(
    instruction,
    retrieval_query=None,
    previous_question=None,
    truncate_index=None,
):
    st.session_state.pending_action = {
        "instruction": instruction,
        "retrieval_query": retrieval_query,
        "previous_question": previous_question,
        "truncate_index": truncate_index,
    }

    st.session_state.editing_message_id = None


def render_answer(text):
    def replace_citation(match):
        number = match.group(1)

        return (
            f'<a href="#source-{number}">'
            f'[{number}]'
            f'</a>'
        )

    rendered = re.sub(
        r"\[(\d+)\]",
        replace_citation,
        text,
    )

    st.markdown(
        rendered,
        unsafe_allow_html=True,
    )


def relevance_label(score):
    if score >= 2.0:
        return "High match", "high"

    if score >= 0.5:
        return "Medium match", "medium"

    return "Low match", "low"


def render_sources(contexts):
    if not contexts:
        return

    with st.expander(
        f"Sources · {len(contexts)}"
    ):
        for context in contexts:
            index = context["index"]
            score = context.get(
                "score",
                0.0,
            )

            label, css_class = relevance_label(
                score
            )

            st.markdown(
                f'<div id="source-{index}" class="source-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Source {index}**"
            )

            st.markdown(
                f"<div class='source-file'>{html.escape(context['source'])}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='source-score {css_class}'>{label} · {score:.3f}</div>",
                unsafe_allow_html=True,
            )

            snippet = context.get(
                "text",
                "",
            )

            st.markdown(
                f"<div class='source-snippet'>{html.escape(snippet[:1000])}{'...' if len(snippet) > 1000 else ''}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "---"
            )


def render_suggestions(
    previous_question,
    message_index,
):
    suggestions = [
        (
            "Can you explain this more simply?",
            "Explain it more simply.",
        ),
        (
            "What are the key points?",
            "Give me the key points.",
        ),
        (
            "Give me an example.",
            "Give me an example.",
        ),
    ]

    cols = st.columns(3)

    for index, (
        label,
        instruction,
    ) in enumerate(suggestions):
        with cols[index]:
            if st.button(
                label,
                key=f"suggestion-{message_index}-{index}",
                use_container_width=True,
            ):
                set_pending_prompt(
                    instruction=instruction,
                    retrieval_query=previous_question,
                    previous_question=previous_question,
                )

                st.rerun()


def render_feedback(
    message_id,
):
    current = st.session_state.feedback.get(
        message_id
    )

    col1, col2, col3 = st.columns(
        [1, 1, 8]
    )

    with col1:
        if st.button(
            "👍",
            key=f"up-{message_id}",
        ):
            st.session_state.feedback[
                message_id
            ] = "positive"

            st.rerun()

    with col2:
        if st.button(
            "👎",
            key=f"down-{message_id}",
        ):
            st.session_state.feedback[
                message_id
            ] = "negative"

            st.rerun()

    with col3:
        if current == "positive":
            st.caption(
                "Thanks for the feedback."
            )

        elif current == "negative":
            st.caption(
                "Thanks — we'll use that feedback."
            )


def render_user_message(
    message,
    message_index,
):
    with st.chat_message(
        "user",
        avatar=None,
    ):
        st.markdown(
            message["content"]
        )

        if st.button(
            "Edit",
            key=f"edit-{message['id']}",
        ):
            st.session_state.editing_message_id = (
                message["id"]
            )

            st.rerun()

        if (
            st.session_state.editing_message_id
            == message["id"]
        ):
            edited = st.text_area(
                "Edit question",
                value=message["content"],
                key=f"edit-text-{message['id']}",
                height=100,
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Submit",
                    key=f"submit-edit-{message['id']}",
                    use_container_width=True,
                ):
                    set_pending_prompt(
                        instruction="",
                        retrieval_query=edited,
                        previous_question=None,
                        truncate_index=message_index,
                    )

                    st.rerun()

            with col2:
                if st.button(
                    "Cancel",
                    key=f"cancel-edit-{message['id']}",
                    use_container_width=True,
                ):
                    st.session_state.editing_message_id = None
                    st.rerun()


def render_assistant_message(
    message,
    message_index,
):
    with st.chat_message(
        "assistant",
        avatar=None,
    ):
        render_answer(
            message["content"]
        )

        if st.session_state.show_performance:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="meta-row">
                        <span class="pill">
                            {message["latency_ms"] / 1000:.1f}s
                        </span>
                        <span class="pill">
                            {message["num_chunks_retrieved"]} chunks
                        </span>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        controls = st.columns(
            [1, 2, 8]
        )

        with controls[0]:
            if st.button(
                "↻",
                key=f"regen-{message['id']}",
                help="Regenerate response",
            ):
                set_pending_prompt(
                    instruction="",
                    retrieval_query=message["question"],
                    previous_question=None,
                    truncate_index=message[
                        "message_index"
                    ],
                )

                st.rerun()

        with controls[1]:
            st.download_button(
                "Copy / download",
                data=message["content"],
                file_name="answer.md",
                mime="text/markdown",
                key=f"export-{message['id']}",
            )

        render_sources(
            message.get(
                "contexts",
                [],
            )
        )

        render_feedback(
            message["id"]
        )

        if message_index == len(
            get_active_session()["messages"]
        ) - 1:
            render_suggestions(
                message["question"],
                message_index,
            )


init_state()

session = get_active_session()


with st.sidebar:
    st.markdown(
        '<div class="brand">RAG Knowledge Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">Search, retrieve, and answer.</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "+ New chat",
        use_container_width=True,
    ):
        new_session()
        st.rerun()

    st.markdown(
        '<div class="section-label">Conversations</div>',
        unsafe_allow_html=True,
    )

    history_query = st.text_input(
        "Search conversations",
        label_visibility="collapsed",
        placeholder="Search conversations...",
    )

    for session_id, item in reversed(
        list(
            st.session_state.sessions.items()
        )
    ):
        if (
            session_id == "default"
            and not item["messages"]
        ):
            continue

        title = item["title"]

        if (
            history_query
            and history_query.lower()
            not in title.lower()
        ):
            continue

        if st.button(
            title,
            key=f"session-{session_id}",
            use_container_width=True,
        ):
            st.session_state.active_session = (
                session_id
            )
            st.rerun()

    st.markdown(
        '<div class="section-label">Add documents</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Add documents",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded_files:
        upload_dir = os.path.join(
            DATA_DIR,
            "uploads",
        )

        os.makedirs(
            upload_dir,
            exist_ok=True,
        )

        for uploaded_file in uploaded_files:
            target = os.path.join(
                upload_dir,
                uploaded_file.name,
            )

            with open(
                target,
                "wb",
            ) as file:
                file.write(
                    uploaded_file.getbuffer()
                )

        st.success(
            "Documents added. Re-index to include them."
        )

    with st.expander("Settings"):
        st.session_state.show_performance = st.checkbox(
            "Show performance details",
            value=st.session_state.show_performance,
        )


for message_index, message in enumerate(
    session["messages"]
):
    if message["role"] == "user":
        render_user_message(
            message,
            message_index,
        )

    elif message["role"] == "assistant":
        render_assistant_message(
            message,
            message_index,
        )


if not session["messages"]:
    st.markdown(
        textwrap.dedent(
            """
            <div class="hero">
                <div class="hero-mark">R</div>
                <div class="hero-title">
                    RAG Knowledge Assistant
                </div>
                <div class="hero-subtitle">
                    Ask any question
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

    examples = [
        "Explain normalization",
        "What is a semaphore?",
        "Explain backpropagation",
    ]

    for index, example in enumerate(
        examples
    ):
        with columns[index]:
            if st.button(
                example,
                key=f"example-{index}",
                use_container_width=True,
            ):
                set_pending_prompt(
                    instruction="",
                    retrieval_query=example,
                    previous_question=None,
                )

                st.rerun()


pending = st.session_state.pending_action

user_input = st.chat_input(
    "Ask any question"
)

prompt_to_process = None
retrieval_query = None
instruction = None
previous_question = None
truncate_index = None


if pending:
    instruction = pending.get(
        "instruction",
        "",
    )

    retrieval_query = pending.get(
        "retrieval_query"
    )

    previous_question = pending.get(
        "previous_question"
    )

    truncate_index = pending.get(
        "truncate_index"
    )

    prompt_to_process = (
        instruction
        if instruction
        else retrieval_query
    )

    st.session_state.pending_action = None

elif user_input:
    prompt_to_process = user_input

    previous_user_question = None

    for message in reversed(
        session["messages"]
    ):
        if message["role"] == "user":
            previous_user_question = message["content"]
            break


    if is_follow_up_query(
        user_input,
        previous_user_question,
    ):
        retrieval_query = previous_user_question
        instruction = user_input
        previous_question = previous_user_question
    else:
        retrieval_query = user_input
        instruction = None
        previous_question = None


if prompt_to_process:
    if truncate_index is not None:
        session["messages"] = session[
            "messages"
        ][:truncate_index]

    if not session["messages"]:
        session["title"] = safe_title(
            retrieval_query or prompt_to_process
        )

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": prompt_to_process,
    }

    session["messages"].append(
        user_message
    )

    with st.chat_message(
        "user",
        avatar=None,
    ):
        st.markdown(
            prompt_to_process
        )

    with st.chat_message(
        "assistant",
        avatar=None,
    ):
        placeholder = st.empty()

        full_answer = ""
        final_result = None

        try:
            for event in stream_answer(
                question=prompt_to_process,
                retrieval_query=retrieval_query,
                instruction=instruction,
                previous_question=previous_question,
            ):
                if event["type"] == "token":
                    full_answer += event["text"]

                    placeholder.markdown(
                        full_answer
                    )

                elif event["type"] == "complete":
                    final_result = event

            if final_result is None:
                raise RuntimeError(
                    "No response was returned."
                )

            placeholder.empty()

            message_id = str(uuid.uuid4())

            assistant_message = {
                "id": message_id,
                "role": "assistant",
                "content": final_result["answer"],
                "question": (
                    retrieval_query
                    or prompt_to_process
                ),
                "contexts": final_result.get(
                    "contexts",
                    [],
                ),
                "latency_ms": final_result[
                    "latency_ms"
                ],
                "num_chunks_retrieved": final_result[
                    "num_chunks_retrieved"
                ],
                "message_index": len(
                    session["messages"]
                ),
            }

            session["messages"].append(
                assistant_message
            )

            render_answer(
                final_result["answer"]
            )

            if st.session_state.show_performance:
                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="meta-row">
                            <span class="pill">
                                {final_result["latency_ms"] / 1000:.1f}s
                            </span>
                            <span class="pill">
                                {final_result["num_chunks_retrieved"]} chunks
                            </span>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

            render_sources(
                final_result.get(
                    "contexts",
                    [],
                )
            )

            render_feedback(
                message_id
            )

            render_suggestions(
                retrieval_query
                or prompt_to_process,
                len(session["messages"]) - 1,
            )

        except Exception as error:
            error_message = f"Error: {error}"

            placeholder.error(
                error_message
            )

            session["messages"].append(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": error_message,
                    "contexts": [],
                    "question": (
                        retrieval_query
                        or prompt_to_process
                    ),
                    "latency_ms": 0,
                    "num_chunks_retrieved": 0,
                    "message_index": len(
                        session["messages"]
                    ),
                }
            )