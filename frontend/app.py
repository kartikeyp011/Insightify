"""
Streamlit frontend application for InsightifyAI.

This module provides the user interface for interacting with the InsightifyAI
FastAPI backend. It handles file uploads, user questions, and the challenge/evaluation
lifecycle. State is managed via Streamlit's session_state and a local summary file.

Dependencies:
    - streamlit: For rendering the web UI.
    - requests: For communicating with the FastAPI backend.
    - os: For checking the existence of summary files on the local disk.
"""

import streamlit as st
import requests
import os

# ── App Configuration & Layout ─────────────────────────────────

st.set_page_config(page_title="InsightifyAI", layout="wide")

st.title("📘InsightifyAI")
st.markdown("Upload a document and interact with it using AI — ask questions, get challenged, and receive justifications.")

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["📤 Upload Document", "❓ Ask Anything", "🧠 Challenge Me"])

# ── Session State Management ───────────────────────────────────

if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = ["", "", ""]
if "evaluation" not in st.session_state:
    st.session_state.evaluation = []

# Model / embedding selection state
if "mode" not in st.session_state:
    st.session_state.mode = None          # "Local Models" | "External APIs"
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = None    # Selected LLM name (local mode only)
if "embedding_choice" not in st.session_state:
    st.session_state.embedding_choice = None  # Selected embedding model (local mode only)

# ── Tab 1: Upload Document ─────────────────────────────────────
if tab == "📤 Upload Document":
    st.header("⚙️ Choose Your AI Configuration")

    # Mode selector — radio renders immediately without a submit button
    mode = st.radio(
        "Select inference mode:",
        options=["Local Models", "External APIs"],
        index=0 if st.session_state.mode != "External APIs" else 1,
        horizontal=True,
        key="mode_radio",
    )
    st.session_state.mode = mode

    # Chunking strategy selector available for ALL modes (Local or External)
    chunking_options = [
        "Large Chunking (1200, overlap 200)",
        "Sentence Based",
        "Token Based",
        "Paragraph Based",
        "Semantic Chunking"
    ]
    if "chunking_strategy" not in st.session_state:
        st.session_state.chunking_strategy = chunking_options[0]
        
    st.session_state.chunking_strategy = st.selectbox(
        "🔪 Chunking Strategy",
        options=chunking_options,
        index=chunking_options.index(st.session_state.chunking_strategy) if st.session_state.chunking_strategy in chunking_options else 0,
    )

    if mode == "Local Models":
        col1, col2 = st.columns(2)

        with col1:
            llm_options = ["Phi-3 Mini", "Gemma 2B", "DeepSeek R1", "SmolLM2"]
            llm_default = (
                llm_options.index(st.session_state.llm_choice)
                if st.session_state.llm_choice in llm_options
                else 0
            )
            st.session_state.llm_choice = st.selectbox(
                "🧠 LLM Model",
                options=llm_options,
                index=llm_default,
                key="llm_selectbox",
            )

        with col2:
            emb_options = ["BGE-base", "all-MiniLM-L6-v2", "e5-base"]
            emb_default = (
                emb_options.index(st.session_state.embedding_choice)
                if st.session_state.embedding_choice in emb_options
                else 0
            )
            st.session_state.embedding_choice = st.selectbox(
                "📐 Embedding Model",
                options=emb_options,
                index=emb_default,
                key="embedding_selectbox",
            )

        st.info(
            f"✅ **Configuration saved** — LLM: `{st.session_state.llm_choice}` | "
            f"Embeddings: `{st.session_state.embedding_choice}`"
        )

    else:  # External APIs
        # Clear local model selections when switching to external mode
        st.session_state.llm_choice = None
        st.session_state.embedding_choice = None
        st.info("🌐 **External APIs mode selected.** Model selection is handled server-side.")

    st.divider()

    st.header("📤 Upload PDF or TXT File")
    uploaded_file = st.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("⏳ Uploading and processing..."):
                # Prepare file for multipart/form-data POST request to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.read())}

                # Bundle the model-config selections as additional form fields.
                # None values are sent as empty strings so the field is always
                # present; the backend treats "" the same as absent.
                backend_mode = "local" if st.session_state.mode == "Local Models" else "external"
                data = {
                    "mode": backend_mode,
                    "llm_choice": st.session_state.llm_choice or "",
                    "embedding_choice": st.session_state.embedding_choice or "",
                    "chunking_strategy": st.session_state.chunking_strategy or "",
                }
                response = requests.post(
                    "http://localhost:8000/api/upload",
                    files=files,
                    data=data,
                )

            if response.status_code == 200:
                st.success("✅ File uploaded successfully!")
                summary = response.json().get("summary")

                if summary is not None:
                    st.subheader("📄 Auto Summary")
                    st.markdown(summary)
                else:
                    st.error("❌ No summary returned from backend.")
            else:
                st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")

# ── Tab 2: Ask Anything ────────────────────────────────────────
elif tab == "❓ Ask Anything":
    st.header("❓ Ask Anything About the Document")
    question = st.text_input("Type your question here:")

    if st.button("Ask"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🔍 Getting answer from AI..."):
                response = requests.post("http://localhost:8000/api/ask", json={"question": question})

            if response.status_code == 200:
                st.markdown("### ✅ Answer")
                st.markdown(response.json()["answer"])
            else:
                st.error(f"❌ Error: {response.json()['detail']}")

# ── Tab 3: Challenge Me ────────────────────────────────────────
elif tab == "🧠 Challenge Me":
    st.header("🧠 Challenge Yourself on the Document")

    if not st.session_state.questions:
        if st.button("Generate Challenge Questions"):
            with st.spinner("💡 Generating questions..."):
                response = requests.get("http://localhost:8000/api/challenge")
            
            if response.status_code == 200:
                # Store generated questions and explicitly reset related state lists 
                # so previous answers/evaluations are cleared for the new challenge
                st.session_state.questions = response.json()["questions"]
                st.session_state.answers = ["", "", ""]
                st.session_state.evaluation = []
            else:
                st.error("❌ Failed to generate questions.")

    if st.session_state.questions:
        st.subheader("📋 Answer the Following Questions")

        for i, q in enumerate(st.session_state.questions):
            # Capture user answers dynamically, mapping the keys securely to the index
            st.session_state.answers[i] = st.text_area(
                f"Q{i+1}: {q}",
                value=st.session_state.answers[i],
                key=f"user_answer_{i}"
            )

        if st.button("Submit Answers"):
            # Ensure every text area has been filled before attempting submission
            if all(ans.strip() for ans in st.session_state.answers):
                with st.spinner("📝 Evaluating your answers..."):
                    response = requests.post(
                        "http://localhost:8000/api/evaluate",
                        json={"answers": st.session_state.answers}
                    )

                if response.status_code == 200:
                    st.session_state.evaluation = response.json()["feedback"]
                else:
                    st.error(f"❌ Evaluation failed: {response.json()['detail']}")
            else:
                st.warning("⚠️ Please answer all 3 questions before submitting.")

    if st.session_state.evaluation:
        st.subheader("🧾 Evaluation Results")
        for i, fb in enumerate(st.session_state.evaluation):
            # Fallback to 'N/A'/'?' if keys are mysteriously missing from backend payload
            st.markdown(f"**Q{i+1}:** {fb.get('question', 'N/A')}")
            st.markdown(f"- 🔸 Your Answer: {fb.get('user_answer', 'N/A')}")
            st.markdown(f"- ✅ Ideal Answer: {fb.get('ideal_answer', 'N/A')}")
            st.markdown(f"- 🧠 Score: {fb.get('score', '?')} / 5")
            st.markdown(f"- 💬 Feedback: {fb.get('feedback', 'No feedback available.')}")
            st.markdown("---")