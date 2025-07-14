import streamlit as st
import requests
import os

# ✅ Constants
SUMMARY_PATH = "summary.txt"  # File to store the last summary

# ✅ 1. Configure the Streamlit page
st.set_page_config(page_title="InsightifyAI", layout="wide")

# ✅ 2. Page Header
st.title("📘InsightifyAI")
st.markdown("Upload a document and interact with it using AI — ask questions, get challenged, and receive justifications.")

# ✅ 3. Sidebar Navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["📤 Upload Document", "❓ Ask Anything", "🧠 Challenge Me"])

# ✅ 4. Session state for data persistence
if "summary" not in st.session_state:
    # Try loading previous summary from disk
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

# 🔷 5. Tab 1: Upload Document
if tab == "📤 Upload Document":
    st.header("📤 Upload PDF or TXT File")
    uploaded_file = st.file_uploader("Choose a .pdf or .txt file", type=["pdf", "txt"])

    if uploaded_file is not None:
        with st.spinner("⏳ Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.read())}
            response = requests.post("http://localhost:8000/api/upload", files=files)

        if response.status_code == 200:
            st.success("✅ File uploaded successfully!")
            summary = response.json()["summary"]
            st.session_state.summary = summary

            # Save summary to disk
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                f.write(summary)

            st.subheader("📄 Auto Summary")
            st.markdown(summary)
        else:
            st.error(f"❌ Upload failed: {response.json()['detail']}")

    elif st.session_state.summary:
        # If a summary already exists, show it
        st.subheader("📄 Auto Summary")
        st.markdown(st.session_state.summary)

# 🔷 6. Tab 2: Ask Anything
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

# 🔷 7. Tab 3: Challenge Me
elif tab == "🧠 Challenge Me":
    st.header("🧠 Challenge Yourself on the Document")

    if not st.session_state.questions:
        if st.button("Generate Challenge Questions"):
            with st.spinner("💡 Generating questions..."):
                response = requests.get("http://localhost:8000/api/challenge")
            if response.status_code == 200:
                st.session_state.questions = response.json()["questions"]
                st.session_state.answers = ["", "", ""]
                st.session_state.evaluation = []
            else:
                st.error("❌ Failed to generate questions.")

    if st.session_state.questions:
        st.subheader("📋 Answer the Following Questions")

        for i, q in enumerate(st.session_state.questions):
            st.session_state.answers[i] = st.text_area(
                f"Q{i+1}: {q}",
                value=st.session_state.answers[i],
                key=f"user_answer_{i}"
            )

        if st.button("Submit Answers"):
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
            st.markdown(f"**Q{i+1}:** {fb.get('question', 'N/A')}")
            st.markdown(f"- 🔸 Your Answer: {fb.get('user_answer', 'N/A')}")
            st.markdown(f"- ✅ Ideal Answer: {fb.get('ideal_answer', 'N/A')}")
            st.markdown(f"- 🧠 Score: {fb.get('score', '?')} / 5")
            st.markdown(f"- 💬 Feedback: {fb.get('feedback', 'No feedback available.')}")
            st.markdown("---")