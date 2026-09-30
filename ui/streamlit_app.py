import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent RAG Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(49, 46, 129, 0.20),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(14, 116, 144, 0.16),
                transparent 35%
            ),
            #090d16;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* ---------- HEADER ---------- */

    .hero {
        text-align: center;
        padding: 20px 0 35px 0;
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
        color: #ffffff;
    }

    .hero-accent {
        color: #60a5fa;
    }

    .hero-line {
        width: 80px;
        height: 4px;
        margin: 18px auto 0 auto;
        border-radius: 10px;
        background: linear-gradient(
            90deg,
            #60a5fa,
            #818cf8
        );
    }

    /* ---------- AGENT CARDS ---------- */

    .agent-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 22px;
        min-height: 175px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.20);
    }

    .agent-icon {
        font-size: 28px;
        margin-bottom: 10px;
    }

    .agent-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .agent-description {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 14px;
    }

    /* ---------- QUESTION BOX ---------- */

    .question-label {
        color: #cbd5e1;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    textarea {
        background-color: #111827 !important;
        color: #f8fafc !important;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        background: linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );
        color: white;
        font-size: 15px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
    }

    /* ---------- RESULT ---------- */

    .result-card {
        background: rgba(15, 23, 42, 0.90);
        border: 1px solid rgba(96, 165, 250, 0.20);
        border-radius: 18px;
        padding: 28px;
        margin-top: 10px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    }

    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 23px;
        font-weight: 750;
        color: #ffffff;
    }

    .status-pass {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #4ade80;
        font-weight: 700;
        font-size: 13px;
    }

    .status-revise {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        font-weight: 700;
        font-size: 13px;
    }

    .answer-box {
        background: #0b1120;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 13px;
        padding: 22px;
        margin-top: 12px;
        color: #e2e8f0;
        line-height: 1.75;
        font-size: 15px;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .feedback-box {
        background: rgba(30, 41, 59, 0.65);
        border-left: 3px solid #60a5fa;
        border-radius: 8px;
        padding: 16px;
        color: #cbd5e1;
        line-height: 1.6;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: #0b1120;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 750;
        margin-top: 5px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 22px;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
        color: #64748b;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🤖</div>
        <div class="hero-title">
            Multi-Agent <span class="hero-accent">RAG</span>
            Research Assistant
        </div>
        <div class="hero-line"></div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AGENT ARCHITECTURE
# ============================================================

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-icon">🔎</div>
            <div class="agent-title">Retriever Agent</div>
            <div class="agent-description">
                Searches the FAISS vector store and retrieves
                the most relevant document chunks for the question.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-icon">📝</div>
            <div class="agent-title">Summarizer Agent</div>
            <div class="agent-description">
                Generates a grounded answer using the information
                retrieved from the document knowledge base.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-icon">🛡️</div>
            <div class="agent-title">Critic Agent</div>
            <div class="agent-description">
                Evaluates the generated answer and can request
                a revision when the answer is insufficient.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Ask a Research Question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "",
    placeholder="Enter your research question...",
    height=110,
    label_visibility="collapsed"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button("🔍  Ask the Research Assistant")


# ============================================================
# API REQUEST
# ============================================================

if ask:

    if not question.strip():

        st.warning("Please enter a research question.")

    else:

        API_URL = "http://127.0.0.1:8000/ask"

        with st.spinner("Running Retriever → Summarizer → Critic..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question.strip()},
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer returned."
                    )

                    decision = result.get(
                        "decision",
                        "UNKNOWN"
                    )

                    revisions = result.get(
                        "revisions",
                        0
                    )

                    feedback = result.get(
                        "feedback",
                        "No critic feedback available."
                    )

                    # --------------------------------------------
                    # RESULT HEADER
                    # --------------------------------------------

                    st.markdown(
                        '<div class="section-title">Research Result</div>',
                        unsafe_allow_html=True
                    )

                    if decision == "PASS":

                        status_html = (
                            '<span class="status-pass">'
                            '✓ PASS'
                            '</span>'
                        )

                    else:

                        status_html = (
                            '<span class="status-revise">'
                            '↻ REVISE'
                            '</span>'
                        )

                   # --------------------------------------------
                   # RESULT STATUS
                   # --------------------------------------------

                    if decision == "PASS":
                        st.success("✓ PASS")
                    else:
                        st.warning("↻ REVISE")

                    # --------------------------------------------
                    # METRICS
                    # --------------------------------------------

                    m1, m2 = st.columns(2)

                    with m1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    Decision
                                </div>
                                <div class="metric-value">
                                    {decision}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with m2:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    Revisions
                                </div>
                                <div class="metric-value">
                                    {revisions}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # --------------------------------------------
                    # ANSWER
                    # --------------------------------------------

                    st.markdown(
                        '<div class="section-title">Answer</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="answer-box">
                            {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # --------------------------------------------
                    # CRITIC FEEDBACK
                    # --------------------------------------------

                    st.markdown(
                        '<div class="section-title">Critic Feedback</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="feedback-box">
                            {feedback}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.error(
                        f"API request failed "
                        f"(HTTP {response.status_code})."
                    )

                    st.code(response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running on port 8000."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Multi-Agent RAG Research Assistant
        · Retriever · Summarizer · Critic · LangGraph
    </div>
    """,
    unsafe_allow_html=True
)