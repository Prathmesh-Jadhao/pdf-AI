from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import os

#prompt
PROMPTS = {
    "Policy Analyzer": """
You are an expert policy advisor.

Give:
1. Simple summary
2. Key benefits
3. Hidden clauses
4. Risks/exclusions
5. Final recommendation

Use simple language.

Context:
{context}

Question:
{query}
""",

    "Contract Review": """
You are a legal contract reviewer.

Give:
1. Risk level (High/Medium/Low)
2. Risky clauses
3. Simple explanation
4. Suggested action

Context:
{context}

Question:
{query}
""",

    "Study Assistant": """
You are an expert tutor.

Give:
1. Easy explanation
2. Key points
3. Important questions
4. MCQs
5. Revision tips

Context:
{context}

Question:
{query}
""",

    "Resume Matcher": """
You are an ATS and career coach.

Give:
1. Match score
2. Missing skills
3. Resume improvements
4. Final advice

Context:
{context}

Question:
{query}
""",

    "Research Assistant": """
You are a research analyst.

Give:
1. Summary
2. Methodology
3. Key findings
4. Limitations
5. Comparison if needed

Context:
{context}

Question:
{query}
"""
}

def ask_question(vector_db, query, mode):
    #search relevant chunks
    docs = vector_db.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=15
)

    # combine retrieved text
    context = "\n".join([doc.page_content for doc in docs])

    prompt = PROMPTS[mode].format(
        context=context,
        query=query)

    # load local LLM
    # llm = Ollama(model = "llama3")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # getanswer
    response = llm.invoke(prompt).content

    return response, docs
