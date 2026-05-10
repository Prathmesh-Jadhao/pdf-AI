import streamlit as st

from utils.pdf_reader import extract_text
from utils.vector_store import create_vector_store
from utils.qa_chain import ask_question

st.set_page_config(page_title="Chat with PDF", page_icon="📄")

st.title("📄 Chat with PDF")
st.write("Upload one or more PDFs and ask questions.")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# sidebar reset
with st.sidebar:
    st.title("Settings")

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

if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.vector_db = None
    st.rerun()


# multiple upload
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# only process AFTER files are uploaded
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
                st.success(f"{len(uploaded_files)} PDFs processed successfully!")

    except Exception as e:
        st.error(str(e))


# show old chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# chat
if st.session_state.vector_db is not None:
    user_input = st.chat_input("Ask a question")

    query = None

    if selected_action != "Custom question":
        query = selected_action

    elif user_input:
        query = user_input

    if query:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask_question(
                    st.session_state.vector_db,
                    query,
                    mode
                )
                st.write(answer)

                with st.expander("View source chunks"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content[:500] + "...")


        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )