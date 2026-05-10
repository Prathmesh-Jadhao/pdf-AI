from utils.pdf_reader import extract_text
from utils.vector_store import create_vector_store
from utils.qa_chain import ask_question

text = extract_text("Prathamesh_Jadhao_Resume_Data_Science.pdf")
db = create_vector_store(text)

query = "what is this PDF about?"
answer = ask_question(db, query)

print(answer)