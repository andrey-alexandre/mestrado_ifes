from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from app.config import settings

def get_retriever(vectorstore):
    return vectorstore.as_retriever()

def get_qa_chain(retriever, model_name: str = "llava:13b"):
    # This is a basic setup, the notebook logic for agents is slightly different (using retriever directly in CriticalReviewAgent)
    # But for general QA purposes:
    if model_name.startswith("gpt"):
        llm = ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY)
    else:
        llm = ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
