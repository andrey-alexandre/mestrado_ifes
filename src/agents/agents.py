"""Agentes de diagnóstico dermatoscópico e construção do grafo LangGraph."""
import json
import logging
import os

import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END

from src.agents.schemas import (
    ABCDDiagnosticAnswer,
    MenziesDiagnosticAnswer,
    SPCLDiagnosticAnswer,
    SummaryAnswer,
    GraphState,
)
from src.models.model import (
    convert_to_base64,
    segment_image as _segment_image,
    apply_precomputed_mask,
)

logger = logging.getLogger(__name__)


def _is_ollama(model_name: str) -> bool:
    """Retorna True para modelos Ollama locais (e.g. 'llava:13b', 'llama3:8b')."""
    return ":" in model_name


def _is_gemini(model_name: str) -> bool:
    """Retorna True para modelos Google Gemini (e.g. 'gemini-1.5-pro')."""
    return model_name.startswith("gemini")


def _build_llm(model_name: str, temperature: float):
    """Instancia o LLM correto baseado no model_name.

    Seleção automática por prefixo/padrão:
      - "gemini-*"  → ChatGoogleGenerativeAI  (GOOGLE_API_KEY)
      - "*:*"       → ChatOllama local         (sem API key)
      - demais      → ChatOpenAI               (OPENAI_API_KEY)

    Args:
        model_name: Nome do modelo (e.g. "gpt-4o-mini", "gemini-1.5-pro", "llava:13b").
        temperature: Temperatura de geração.

    Returns:
        Instância de ChatGoogleGenerativeAI, ChatOllama ou ChatOpenAI.
    """
    if _is_ollama(model_name):
        return ChatOllama(model=model_name)
    if _is_gemini(model_name):
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key="GOOGLE_API_KEY",
        )
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key="OPENAI_API_KEY",
    )


def _invoke(llm, output_format, messages: list, model_name: str):
    """Invoca o LLM com saída estruturada (OpenAI/Gemini) ou texto livre (Ollama).

    OpenAI e Gemini suportam with_structured_output() via JSON schema nativo.
    Modelos Ollama retornam o conteúdo textual diretamente (sem schema forçado).

    Args:
        llm: Instância do LLM já construída.
        output_format: Classe Pydantic do schema esperado.
        messages: Lista de mensagens LangChain (HumanMessage, etc.).
        model_name: Nome do modelo (para distinguir Ollama dos demais).

    Returns:
        Objeto Pydantic (OpenAI/Gemini) ou string (Ollama).
    """
    if _is_ollama(model_name):
        return llm.invoke(messages).content
    return llm.with_structured_output(output_format).invoke(messages)


def _image_message(seg_b64: str, text: str) -> HumanMessage:
    """Constrói um HumanMessage multimodal com imagem segmentada e texto.

    A imagem é enviada como image_url (base64 inline), que é o formato
    correto para APIs de visão — não como texto interpolado no prompt.

    Args:
        seg_b64: String base64 da imagem segmentada.
        text: Texto do prompt clínico.

    Returns:
        HumanMessage com conteúdo multimodal.
    """
    return HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{seg_b64}"}},
        {"type": "text", "text": text},
    ])


def _serialize_diagnosis(diagnosis) -> str:
    """Serializa um diagnóstico Pydantic ou string para inclusão em prompt de texto."""
    if hasattr(diagnosis, "model_dump"):
        return json.dumps(diagnosis.model_dump(), ensure_ascii=False, indent=2)
    return str(diagnosis)


# ──────────────────────────────────────────────
# Agente de Segmentação
# ──────────────────────────────────────────────

class SegmentationAgent:
    """Pré-processa a imagem e gera a máscara de segmentação via U-Net."""

    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 0.4,
        image_size: int = 224,
        mode: str = "train",
        augmentation_prob: float = 0.4,
    ):
        """
        Args:
            model: Modelo U-Net pré-carregado.
            threshold: Limiar sigmoid para binarização da máscara.
            image_size: Tamanho de referência da imagem (não usado diretamente).
            mode: Modo de operação ('train' ou 'test').
            augmentation_prob: Probabilidade de aplicar augmentations.
        """
        self.model = model
        self.threshold = threshold
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

    def segment_image(self, state: GraphState) -> dict:
        """Lê a imagem do caminho no estado, segmenta e retorna as versões Base64.

        Args:
            state: Estado do grafo contendo image_path.

        Returns:
            Dict com 'seg_image_data' e 'image_data' em Base64.
        """
        # Delega a lógica de segmentação para src.models.model.segment_image
        # para evitar duplicação de código
        encoded_string, seg_encoded_string = _segment_image(
            image_path=state.image_path,
            model=self.model,
            threshold=self.threshold,
        )
        return {"seg_image_data": seg_encoded_string, "image_data": encoded_string}


# ──────────────────────────────────────────────
# Agente de Segmentação com máscara pré-computada
# ──────────────────────────────────────────────

class PrecomputedSegmentationAgent:
    """Carrega uma segmentação pré-computada (padrão PH2) em vez de rodar o U-Net.

    Requer que state.seg_path esteja preenchido com o caminho da máscara binária.
    Aplica a máscara sobre a imagem original e retorna ambas em Base64.
    """

    def segment_image(self, state: GraphState) -> dict:
        """Lê a imagem e a máscara do estado e retorna as versões Base64.

        Args:
            state: Estado do grafo com image_path e seg_path preenchidos.

        Returns:
            Dict com 'image_data' e 'seg_image_data' em Base64.
        """
        if not state.seg_path:
            raise ValueError(
                "seg_path não definido no estado. "
                "Use --segmentation-mode precomputed apenas com datasets que fornecem máscaras."
            )
        encoded_string, seg_encoded_string = apply_precomputed_mask(
            state.image_path, state.seg_path
        )
        return {"seg_image_data": seg_encoded_string, "image_data": encoded_string}


# ──────────────────────────────────────────────
# Agente ABCD
# ──────────────────────────────────────────────

class ABCDDiagnosticAgent:
    """Analisa a lesão pelo algoritmo ABCD de dermoscopia."""

    def __init__(self, model_name: str, output_format, temperature: float = 0.7):
        """
        Args:
            model_name: Nome do LLM ("gpt-4o-mini" ou "llava:13b").
            output_format: Classe Pydantic para parse estruturado da saída.
            temperature: Temperatura de geração do LLM.
        """
        self.model_name = model_name
        self.output_format = output_format
        self.llm = _build_llm(model_name, temperature)

    def analyze_lesion(self, state: GraphState) -> dict:
        """Executa a análise ABCD sobre a imagem segmentada.

        A imagem segmentada é enviada como image_url dentro do HumanMessage.
        A saída estruturada é garantida por with_structured_output (OpenAI)
        ou retornada como texto para llava.

        Args:
            state: Estado do grafo contendo image_data e seg_image_data.

        Returns:
            Dict com 'diagnosis_abcd' contendo o resultado estruturado.
        """
        text_prompt = """
        <-- Identidade do Agente -->
        Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

        Sua função é identificar e descrever padrões visuais relevantes encontrados nas imagens fornecidas, utilizando critérios técnicos que serão informados abaixo.

        Não forneça um diagnóstico ou julgamento clínico. Apenas descreva o que é visualmente observado com base em critérios objetivos.

        <-- Instruções Gerais -->
        - Ignore variações de cor típicas da pele humana, como tons naturais da derme ou pequenas sombras.
        - Só considere estruturas dermatoscópicas se forem claramente identificáveis, sem dúvida razoável.
        - Não atribua significância clínica a artefatos visuais ou ruídos.
        - Seja conservador ao identificar padrões visuais suspeitos. Se a evidência não for forte, trate como característica benigna ou neutra.

        Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

        <-- Instruções Específicas -->

        Avalie a lesão cutânea na imagem abaixo utilizando o algoritmo ABCD de dermoscopia. Para cada um dos critérios (Assimetria, Bordas, Cor e Estruturas Dermoscópicas), forneça a pontuação de acordo com a seguinte escala:
        Critérios Gerais:
          Considere assimetrias somente aquelas mais acentuadas, caso seja leve, não considere.
          Ao avaliar as cores, desconsidere da contagem a cor da pele do indivíduo para não enviesar a análise
        Assimetria (A): Considere assimetrias somente aquelas acentuadas, caso seja leve, não considere.
          0: Nenhuma assimetria
          1: Assimetria em um eixo
          2: Assimetria em ambos os eixos
        Bordas (B): Divida a imagem em 8 quadrantes de ângulos iguais. Dentro de cada quadrante avalie se a borda tem fim claro delimitado ou não.
          0: Bordas indistintas em todos os quadrantes
          1-8: Bordas nítidas em alguns ou todos os quadrantes (pontuação proporcional)
        Cor (C): Ao avaliar as cores, desconsidere da contagem a cor da pele do indivíduo para não enviesar a análise
          Atribua 1 ponto para cada cor presente (branco/bege, vermelho/rosa, marrom claro, marrom escuro, azul-cinza, preto).
        Estruturas Dermoscópicas (D):
          Atribua 1 ponto para cada estrutura observada (áreas sem estrutura, rede pigmentada, linhas ramificadas, pontos, glóbulos).
        Resultado final:
          Faça a soma ponderada dos resultados de cada critério, com pesos de 1.3, .1, .5 e .5, respectivamente. Mostre o cálculo da soma ponderada passo a passo. Caso o valor seja inferior a 4.75, é uma lesão benigna. Caso seja superior a 4.75, mas inferior a 5.45 é uma lesão suspeita. Caso seja maior que 5.45 é uma lesão maligna.
        """

        messages = [_image_message(state.seg_image_data, text_prompt)]
        output = _invoke(self.llm, self.output_format, messages, self.model_name)
        return {"diagnosis_abcd": output}


# ──────────────────────────────────────────────
# Agente Menzies
# ──────────────────────────────────────────────

class MenziesDiagnosticAgent:
    """Analisa a lesão pelo Método Menzies de dermoscopia."""

    def __init__(self, model_name: str, output_format, temperature: float = 0.7):
        """
        Args:
            model_name: Nome do LLM ("gpt-4o-mini" ou "llava:13b").
            output_format: Classe Pydantic para parse estruturado da saída.
            temperature: Temperatura de geração do LLM.
        """
        self.model_name = model_name
        self.output_format = output_format
        self.llm = _build_llm(model_name, temperature)

    def analyze_lesion(self, state: GraphState) -> dict:
        """Executa a análise pelo Método Menzies sobre a imagem segmentada.

        Args:
            state: Estado do grafo contendo image_data e seg_image_data.

        Returns:
            Dict com 'diagnosis_menzies' contendo o resultado estruturado.
        """
        text_prompt = """
      <-- Identidade do Agente -->
      Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

      Sua função é identificar e descrever padrões visuais relevantes encontrados nas imagens fornecidas, utilizando critérios técnicos que serão informados abaixo.

      Não forneça um diagnóstico ou julgamento clínico. Apenas descreva o que é visualmente observado com base em critérios objetivos.

      <-- Instruções Gerais -->
      - Ignore variações de cor típicas da pele humana, como tons naturais da derme ou pequenas sombras.
      - Só considere estruturas dermatoscópicas se forem claramente identificáveis, sem dúvida razoável.
      - Não atribua significância clínica a artefatos visuais ou ruídos.
      - Seja conservador ao identificar padrões visuais suspeitos. Se a evidência não for forte, trate como característica benigna ou neutra.
      - Considere assimetrias somente aquelas acentuadas, caso seja leve, não considere.

      Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

      <-- Instruções Específicas -->
      Avalie a lesão cutânea na imagem abaixo utilizando o Método Menzies. Aplique os critérios para as Características Positivas e Características Negativas da seguinte forma:

      Características Positivas (pelo menos uma deve estar presente para diagnóstico de melanoma):
        Véu Azul-Branco: Se presente, pontue 1. Caso contrário, pontue 0.
        Múltiplos Pontos Marrons: Se presentes, pontue 1. Caso contrário, pontue 0.
        Pseudópodes: Se presentes, pontue 1. Caso contrário, pontue 0.
        Streaming Radial: Se presente, pontue 1. Caso contrário, pontue 0.
        Despigmentação Tipo Cicatriz: Se presente, pontue 1. Caso contrário, pontue 0.
        Pontos/Glóbulos Pretos Periféricos: Se presentes, pontue 1. Caso contrário, pontue 0.
        Múltiplas Cores (vermelho/rosa, branco/bege, marrom escuro, preto, cinza e azul): Se cinco ou seis cores presentes, pontue 1. Caso contrário, pontue 0.
        Vários pontos/saliências azul-acinzentados: Se presentes, pontue 1. Caso contrário, pontue 0.
        Linhas ramificadas: Se presentes, pontue 1. Caso contrário, pontue 0.
      Características Negativas (ambas devem estar ausentes para diagnóstico de melanoma):
        Cor Única: Se a lesão apresentar apenas uma cor, pontue 1. Caso contrário, pontue 0.
        Simetria do padrão de pigmentação: Se a lesão for simétrica na distribuição de cores, pontue 1. Caso contrário, pontue 0.

      Resultado Final:
        A lesão será classificada como melanoma caso possua 1 característica positiva e as 2 características negativas estiverem ausentes. Se houver apenas 1 característica negativa presente, indica suspeita de melanoma. Não havendo caracterísitica positiva e tendo as 2 características negativas presentes, caracteriza lesão benigna.
      """

        messages = [_image_message(state.seg_image_data, text_prompt)]
        output = _invoke(self.llm, self.output_format, messages, self.model_name)
        return {"diagnosis_menzies": output}


# ──────────────────────────────────────────────
# Agente SPCL (Seven-Point Checklist)
# ──────────────────────────────────────────────

class SPCLDiagnosticAgent:
    """Analisa a lesão pela Checklist de Sete Pontos (Seven-Point Checklist)."""

    def __init__(self, model_name: str, output_format, temperature: float = 0.7):
        """
        Args:
            model_name: Nome do LLM ("gpt-4o-mini" ou "llava:13b").
            output_format: Classe Pydantic para parse estruturado da saída.
            temperature: Temperatura de geração do LLM.
        """
        self.model_name = model_name
        self.output_format = output_format
        self.llm = _build_llm(model_name, temperature)

    def analyze_lesion(self, state: GraphState) -> dict:
        """Executa a análise SPCL sobre a imagem segmentada.

        Args:
            state: Estado do grafo contendo image_data e seg_image_data.

        Returns:
            Dict com 'diagnosis_spcl' contendo o resultado estruturado.
        """
        text_prompt = """
      <-- Identidade do Agente -->
      Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

      Sua função é identificar e descrever padrões visuais relevantes encontrados nas imagens fornecidas, utilizando critérios técnicos que serão informados abaixo.

      Não forneça um diagnóstico ou julgamento clínico. Apenas descreva o que é visualmente observado com base em critérios objetivos.
      <-- Instruções Gerais -->
      - Ignore variações de cor típicas da pele humana, como tons naturais da derme ou pequenas sombras.
      - Só considere estruturas dermatoscópicas se forem claramente identificáveis, sem dúvida razoável.
      - Não atribua significância clínica a artefatos visuais ou ruídos.
      - Seja conservador ao identificar padrões visuais suspeitos. Se a evidência não for forte, trate como característica benigna ou neutra.
      - Considere assimetrias somente aquelas mais acentuadas, caso seja leve, não considere.
      - Não considerar critério de bordas, diâmetro e sangramento na análise.

      Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

      <-- Instruções Específicas -->

      Avalie a lesão cutânea na imagem abaixo utilizando a Checklist de Sete Pontos (Seven-Point Checklist). Aplique os critérios conforme descrito abaixo:
      Critérios Maiores (2 pontos cada):
        Padrão de pigmentação atípico: Se presente, pontue 2. Caso contrário, pontue 0.
        Véu azul-branco irregular: Se presente, pontue 2. Caso contrário, pontue 0.
        Vasos atípicos: Se presentes, pontue 2. Caso contrário, pontue 0.
      Critérios Menores (1 ponto cada):
        Padrão de retículo irregular: Se presente, pontue 1. Caso contrário, pontue 0.
        Padrão de glóbulos irregulares: Se presente, pontue 1. Caso contrário, pontue 0.  Considere irregularidades somente aquelas mais acentuadas, caso seja leve, não considere.
        Hiperpigmentação localizada (pontos escuros localizados): Se presente, pontue 1. Caso contrário, pontue 0.
        Regressão (áreas esbranquiçadas ou azul-acinzentadas): Se presente, pontue 1. Caso contrário, pontue 0.
      Resultado Final:
        A lesão será considerada suspeita para melanoma se a pontuação total for 3 ou mais pontos.
        Pontuações inferiores a 3 indicam menor probabilidade de malignidade, mas não descartam a necessidade de avaliação clínica.
      """

        messages = [_image_message(state.seg_image_data, text_prompt)]
        output = _invoke(self.llm, self.output_format, messages, self.model_name)
        return {"diagnosis_spcl": output}


# ──────────────────────────────────────────────
# Agente de Síntese
# ──────────────────────────────────────────────

class SummaryAgent:
    """Integra os resultados dos três algoritmos (ABCD, Menzies, SPCL) em parecer final."""

    def __init__(self, model_name: str, output_format, temperature: float = 0.7):
        """
        Args:
            model_name: Nome do LLM ("gpt-4o-mini" ou "llava:13b").
            output_format: Classe Pydantic para parse estruturado da saída.
            temperature: Temperatura de geração do LLM.
        """
        self.model_name = model_name
        self.output_format = output_format
        self.llm = _build_llm(model_name, temperature)

    def summarize(self, state: GraphState) -> dict:
        """Consolida os três diagnósticos em um parecer clínico único.

        Os três diagnósticos são serializados e embutidos no texto.
        A imagem segmentada é incluída como image_url para que o modelo
        possa resolver divergências entre algoritmos consultando a imagem.

        Args:
            state: Estado do grafo contendo diagnosis_abcd, diagnosis_menzies,
                   diagnosis_spcl e seg_image_data.

        Returns:
            Dict com 'validation' contendo o parecer consolidado.
        """
        abcd_str = _serialize_diagnosis(state.diagnosis_abcd)
        menzies_str = _serialize_diagnosis(state.diagnosis_menzies)
        spcl_str = _serialize_diagnosis(state.diagnosis_spcl)

        text_prompt = f"""
      <-- Identidade do Agente -->
      Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.
      Não forneça um diagnóstico ou julgamento clínico.

      Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

      <-- Instruções Específicas -->
      Você está atuando como um especialista que recebeu três pareceres técnicos distintos sobre uma mesma lesão cutânea, baseados nos seguintes algoritmos:
      1️⃣ ABCD – {abcd_str}
      2️⃣ Menzies – {menzies_str}
      3️⃣ SPCL (Seven Point Checklist) – {spcl_str}

      Sua função é integrar os três pareceres fornecidos, comparando seus resultados e justificativas, para construir um **diagnóstico consolidado e fundamentado**.

      Considere:
      - O grau de concordância ou conflito entre os algoritmos;
      - A presença de padrões de alto risco recorrentes nos pareceres;
      - A confiabilidade dos critérios observados (por exemplo, múltiplas cores, bordas irregulares e estruturas atípicas);
      - A gravidade potencial com base em critérios combinados.
      - Caso haja divergências significativas entre os pareceres, utilize a imagem disponibilizada para avaliar e decidir qual está mais condizente com a lesão
      """

        messages = [_image_message(state.seg_image_data, text_prompt)]
        output = _invoke(self.llm, self.output_format, messages, self.model_name)
        return {"validation": output}


# ──────────────────────────────────────────────
# Agente Crítico (RAG)
# ──────────────────────────────────────────────

class CriticalReviewAgent:
    """Valida o diagnóstico consolidado contra literatura médica via RAG."""

    def __init__(self, retriever, model_name: str):
        """
        Args:
            retriever: LangChain retriever FAISS com literatura dermatológica.
            model_name: Nome do LLM (usado para determinar o formato do estado).
        """
        self.retriever = retriever
        self.model_name = model_name

    def validate_diagnosis(self, state: GraphState) -> dict:
        """Recupera trechos relevantes da literatura e compõe o relatório final.

        Args:
            state: Estado do grafo contendo 'validation' com o parecer consolidado.

        Returns:
            Dict com 'final_report' contendo confirmação baseada em literatura.
        """
        if _is_ollama(self.model_name):
            val_res = state.validation
        else:
            val_res = state.validation.model_dump()

        retrieved_docs = self.retriever.invoke(str(val_res))
        return {
            "final_report": (
                f"Confirmação baseada em literatura médica: "
                f"{retrieved_docs[0].page_content[:200]}..."
            )
        }


# ──────────────────────────────────────────────
# Construção do Grafo LangGraph
# ──────────────────────────────────────────────

def build_graph(
    retriever,
    model_name: str,
    temperature: float = 0.7,
    seg_threshold: float = 0.4,
    segmentation_mode: str = "model",
    seg_model: torch.nn.Module = None,
):
    """Constrói e compila o grafo de agentes LangGraph.

    Fluxo:
        segmentation → [diagnostic_abcd ‖ diagnostic_menzies ‖ diagnostic_spcl]
                     → summary → critical_review

    Args:
        retriever: Retriever FAISS com literatura médica.
        model_name: Nome do LLM para os agentes diagnósticos.
        temperature: Temperatura de geração do LLM.
        seg_threshold: Limiar sigmoid para o SegmentationAgent (modo "model").
        segmentation_mode: "model" — gera segmentação via U-Net em runtime;
                           "precomputed" — usa máscara pré-existente de state.seg_path.
        seg_model: Modelo U-Net pré-carregado. Obrigatório quando segmentation_mode="model".

    Returns:
        Grafo LangGraph compilado.
    """
    if segmentation_mode == "model" and seg_model is None:
        raise ValueError(
            "seg_model é obrigatório quando segmentation_mode='model'. "
            "Carregue o modelo com load_segmentation_model() antes de chamar build_graph()."
        )

    graph = StateGraph(GraphState)

    # Selecionar agente de segmentação conforme o modo configurado
    if segmentation_mode == "precomputed":
        seg_node = PrecomputedSegmentationAgent().segment_image
    else:
        seg_node = SegmentationAgent(seg_model, threshold=seg_threshold).segment_image

    # Registrar nós
    graph.add_node("segmentation", seg_node)
    graph.add_node(
        "diagnostic_abcd",
        ABCDDiagnosticAgent(model_name, ABCDDiagnosticAnswer, temperature).analyze_lesion,
    )
    graph.add_node(
        "diagnostic_menzies",
        MenziesDiagnosticAgent(model_name, MenziesDiagnosticAnswer, temperature).analyze_lesion,
    )
    graph.add_node(
        "diagnostic_spcl",
        SPCLDiagnosticAgent(model_name, SPCLDiagnosticAnswer, temperature).analyze_lesion,
    )
    graph.add_node(
        "summary",
        SummaryAgent(model_name, SummaryAnswer, temperature).summarize,
    )
    graph.add_node(
        "critical_review",
        CriticalReviewAgent(retriever, model_name).validate_diagnosis,
    )

    # Definir conexões do fluxo
    graph.add_edge(START, "segmentation")
    graph.add_edge("segmentation", "diagnostic_abcd")
    graph.add_edge("segmentation", "diagnostic_menzies")
    graph.add_edge("segmentation", "diagnostic_spcl")
    # Os três agentes de diagnóstico convergem para o agente de síntese
    graph.add_edge(
        ["diagnostic_abcd", "diagnostic_menzies", "diagnostic_spcl"], "summary"
    )
    graph.add_edge("summary", "critical_review")
    graph.add_edge("critical_review", END)

    return graph.compile()
