from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import os

def ask_question(vector_db, query):
    #search relevant chunks
    docs = vector_db.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=10
)

    # combine retrieved text
    context = "\n".join([doc.page_content for doc in docs])

    #prompt
    prompt = f"""
    You are a helpful assistant.
    Answer ONLY using the provided context.
    If the answer is not in the context, say:
    "I could not find that in the uploaded documents."

    Context:
    {context}

    Question:
    {query}
    """

    # load local LLM
    # llm = Ollama(model = "llama3")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # getanswer
    response = llm.invoke(prompt).content

    return response, docs
