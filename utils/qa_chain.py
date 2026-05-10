from langchain_community.llms import Ollama

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
    llm = Ollama(model = "llama3")

    # getanswer
    response = llm.invoke(prompt)

    return response, docs
