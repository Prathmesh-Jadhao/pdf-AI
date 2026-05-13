import streamlit as st

from utils.pdf_reader import extract_text
from utils.vector_store import create_vector_store
from utils.qa_chain import ask_question


# Page config
st.set_page_config(
    page_title="Doc AI",
    page_icon="🧠"
)

st.title("Doc AI")
st.write("Understand. Compare. Decide.")


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    mode = st.selectbox(
        "Choose Mode",
        [
            "Policy Analyzer",
            "Contract Review",
            "Study Assistant",
            "Resume Matcher",
            "Research Assistant"
        ]
    )

    st.subheader("⚡ Quick Actions")

    quick_actions = {
        "Policy Analyzer": [
            "Summarize this policy",
            "List hidden clauses",
            "Compare benefits and exclusions"
        ],
        "Contract Review": [
            "Highlight risky clauses",
            "Summarize obligations",
            "What should I negotiate?"
        ],
        "Study Assistant": [
            "Summarize this chapter",
            "Generate 10 MCQs",
            "Create revision notes"
        ],
        "Resume Matcher": [
            "Give ATS score",
            "List missing skills",
            "Suggest improvements"
        ],
        "Research Assistant": [
            "Summarize paper",
            "Extract methodology",
            "List key findings"
        ]
    }

    selected_action = st.selectbox(
        "Choose an action",
        ["Custom question"] + quick_actions[mode]
    )

    run_quick_action = st.button("Run Quick Action")

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.rerun()


# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.session_state.vector_db is None:
    try:
        with st.spinner("Processing PDFs..."):
            full_text = ""

            for pdf in uploaded_files:
                text = extract_text(pdf)
                if text:
                    full_text += text + "\n"

            if not full_text.strip():
                st.error("No readable text found in uploaded PDFs.")
            else:
                st.session_state.vector_db = create_vector_store(full_text)
                st.success(
                    f"{len(uploaded_files)} PDF(s) processed successfully!"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")


# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------- CHAT ----------------
if st.session_state.vector_db is None:
    st.info("👆 Upload PDF(s) to get started")

else:
    user_input = st.chat_input("Ask a question")

    query = None

    # quick action runs only once
    if run_quick_action and selected_action != "Custom question":
        query = selected_action

    # custom typed question
    elif user_input:
        query = user_input

    if query:
        # show user msg
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.write(query)

        # get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask_question(
                    st.session_state.vector_db,
                    query,
                    mode
                )

                st.write(answer)

                # source chunks
                with st.expander("📄 View source chunks"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content[:500] + "...")

        # save response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
