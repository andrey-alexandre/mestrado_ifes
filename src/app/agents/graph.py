import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

from app.agents.schemas import (
    GraphState, ABCDDiagnosticAnswer, SummaryAnswer
)
from app.agents.prompts import ABCD_PROMPT, SUMMARY_PROMPT
from app.agents.llm import get_llm
from app.agents.utils import normalize_answer
from app.segmentation.inference import segment_image
from app.segmentation.loader import load_model

# Agent Classes
class SegmentationAgent:
    def __init__(self):
        self.model = load_model()

    def segment(self, state: GraphState):
        result = segment_image(self.model, state.image_path)
        return {
            "seg_image_data": result["seg_image_data"],
            "image_data": result["image_data"]
        }

class DiagnosticAgent:
    def __init__(self, model_name, output_format, prompt_template, result_key):
        self.model_name = model_name
        # If dummy, make it specific to the agent type so get_llm returns the right mock
        llm_model_name = model_name
        if model_name.startswith("dummy"):
             if "abcd" in result_key: llm_model_name = "dummy:abcd"
             elif "menzies" in result_key: llm_model_name = "dummy:menzies"
             elif "spcl" in result_key: llm_model_name = "dummy:spcl"

        self.llm = get_llm(llm_model_name)
        self.parser = PydanticOutputParser(pydantic_object=output_format)
        self.prompt_template = prompt_template
        self.result_key = result_key


    def analyze(self, state: GraphState):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["image_original", "image_segmentada"], # Assuming prompt uses these keys
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        chain = prompt | self.llm
        if "llava" not in self.model_name:
             chain = chain | self.parser

        img1 = f"data:image/jpeg;base64,{state.image_data}"
        img2 = f"data:image/jpeg;base64,{state.seg_image_data}"

        try:
            output = chain.invoke({"image_original": img1, "image_segmentada": img2})
            if "llava" in self.model_name and hasattr(output, "content"):
                 output = output.content
        except Exception as e:
            # Fallback or error handling
            print(f"Error in {self.result_key}: {e}")
            output = "Error in analysis"

        return {self.result_key: output}

class SummaryAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        llm_model_name = model_name
        if model_name.startswith("dummy"):
            llm_model_name = "dummy:summary"

        self.llm = get_llm(llm_model_name)
        self.parser = PydanticOutputParser(pydantic_object=SummaryAnswer)


    def summarize(self, state: GraphState):
        prompt = PromptTemplate(
            template=SUMMARY_PROMPT,
            input_variables=["diagnosis_abcd1", "diagnosis_abcd2", "diagnosis_abcd3", "image_segmentada"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        chain = prompt | self.llm
        if "llava" not in self.model_name:
            chain = chain | self.parser

        img2 = f"data:image/jpeg;base64,{state.seg_image_data}"

        # Ensure we are passing strings or dicts properly
        d_abcd1 = normalize_answer(state.diagnosis_abcd1)
        d_abcd2 = normalize_answer(state.diagnosis_abcd2)
        d_abcd3 = normalize_answer(state.diagnosis_abcd3)

        try:
            output = chain.invoke({
                "diagnosis_abcd1": d_abcd1,
                "diagnosis_abcd2": d_abcd2,
                "diagnosis_abcd3": d_abcd3,
                "image_segmentada": img2
            })
            if "llava" in self.model_name and hasattr(output, "content"):
                output = output.content
        except Exception as e:
            print(f"Error in summary: {e}")
            output = "Error in summary"

        return {"validation": output}

class CriticalReviewAgent:
    def __init__(self, retriever, model_name):
        self.retriever = retriever
        self.model_name = model_name

    def validate(self, state: GraphState):
        val_res = state.validation
        if not isinstance(val_res, str):
            val_res = val_res.model_dump_json()

        try:
            retrieved_docs = self.retriever.invoke(str(val_res))
            # Basic RAG usage as per notebook
            if retrieved_docs:
                report = f"Confirmação baseada em literatura médica: {retrieved_docs[0].page_content[:200]}..."
            else:
                report = "Sem documentos relevantes encontrados."
        except Exception as e:
            report = f"Could not validate: {e}"

        return {"final_report": report}

def build_graph(model_name: str, retriever):
    graph = StateGraph(GraphState)

    seg_agent = SegmentationAgent()
    abcd_agent1 = DiagnosticAgent(model_name, ABCDDiagnosticAnswer, ABCD_PROMPT, "diagnosis_abcd1")
    abcd_agent2 = DiagnosticAgent(model_name, ABCDDiagnosticAnswer, ABCD_PROMPT, "diagnosis_abcd2")
    abcd_agent3 = DiagnosticAgent(model_name, ABCDDiagnosticAnswer, ABCD_PROMPT, "diagnosis_abcd3")
    summary_agent = SummaryAgent(model_name)
    review_agent = CriticalReviewAgent(retriever, model_name)

    graph.add_node("segmentation", seg_agent.segment)
    graph.add_node("diagnostic_abcd1", abcd_agent1.analyze)
    graph.add_node("diagnostic_abcd2", abcd_agent2.analyze)
    graph.add_node("diagnostic_abcd3", abcd_agent3.analyze)
    graph.add_node("summary", summary_agent.summarize)
    graph.add_node("critical_review", review_agent.validate)

    graph.add_edge(START, "segmentation")
    graph.add_edge("segmentation", "diagnostic_abcd1")
    graph.add_edge("segmentation", "diagnostic_abcd2")
    graph.add_edge("segmentation", "diagnostic_abcd3")
    graph.add_edge(["diagnostic_abcd1", "diagnostic_abcd2", "diagnostic_abcd3"], "summary")
    graph.add_edge("summary", "critical_review")
    graph.add_edge("critical_review", END)

    return graph.compile()
