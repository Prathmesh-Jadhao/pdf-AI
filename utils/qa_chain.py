from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import os

def ask_question(vector_db, query):
    #search relevant chunks
    docs = vector_db.similarity_search(query, k=3)

    # combine retrieved text
    context = "\n".join([doc.page_content for doc in docs])

    #prompt
    prompt = f"""
    Answer the question using only the content below.

    Context:
    {context}

    Question:
    {query}
    """

    # load local LLM
    # llm = Ollama(model = "llama3")
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # getanswer
    response = llm.invoke(prompt).content

    return response, docs
