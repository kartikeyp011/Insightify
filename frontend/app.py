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

# ── Constants ──────────────────────────────────────────────────
# NOTE: The summary file persists across app restarts so users don't lose context 
# if they refresh or if the app goes to sleep.
SUMMARY_PATH = "summary.txt"  # File to store the last summary

# ── App Configuration & Layout ─────────────────────────────────

st.set_page_config(page_title="InsightifyAI", layout="wide")

st.title("📘InsightifyAI")
st.markdown("Upload a document and interact with it using AI — ask questions, get challenged, and receive justifications.")

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["📤 Upload Document", "❓ Ask Anything", "🧠 Challenge Me"])

# ── Session State Management ───────────────────────────────────

if "summary" not in st.session_state:
    # Try loading previous summary from disk to persist context across reloads
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            st.session_state.summary = f.read()
    else:
        st.session_state.summary = None

if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = ["", "", ""]
if "evaluation" not in st.session_state:
    st.session_state.evaluation = []

# ── Tab 1: Upload Document ─────────────────────────────────────
if tab == "📤 Upload Document":
    st.header("📤 Upload PDF or TXT File")
    uploaded_file = st.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"])

    if uploaded_file is not None:
        with st.spinner("⏳ Uploading and processing..."):
            # Prepare file for multipart/form-data POST request to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.read())}
            # TODO(dev): Move hardcoded API URLs to an environment variable or config file
            response = requests.post("http://localhost:8000/api/upload", files=files)

        if response.status_code == 200:
            st.success("✅ File uploaded successfully!")
            summary = response.json()["summary"]
            st.session_state.summary = summary

            # Cache the summary to disk for retrieval on subsequent app accesses
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                f.write(summary)

            st.subheader("📄 Auto Summary")
            st.markdown(summary)
        else:
            st.error(f"❌ Upload failed: {response.json()['detail']}")

    elif st.session_state.summary:
        st.subheader("📄 Auto Summary")
        st.markdown(st.session_state.summary)

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