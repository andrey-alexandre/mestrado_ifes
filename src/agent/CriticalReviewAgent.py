class CriticalReviewAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def validate_diagnosis(self, state):
        retrieved_docs = self.retriever.invoke(state.validation)
        return {"final_report": f"Confirmação baseada em literatura médica: {retrieved_docs[0].page_content[:200]}..."}