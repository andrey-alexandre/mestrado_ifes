"""Pydantic output schemas for LLM diagnostic agents and LangGraph state."""
from typing import Optional, Union

from pydantic import BaseModel, Field


class ABCDDiagnosticAnswer(BaseModel):
    """Resultado estruturado do algoritmo ABCD para análise dermatoscópica."""

    assimetria: str = Field(description="Valor de 0 a 2 e motivo.")
    bordas: str = Field(description="Valor de 0 a 8 e motivo.")
    cor: str = Field(description="Valor de 1 a 6 e motivo.")
    estruturas: str = Field(description="Valor de 1 a 5 e motivo.")
    explicacao: str = Field(description="Explicação das pontuações dadas.")


class MenziesPositiveAnswer(BaseModel):
    """Características positivas do Método Menzies (aumentam suspeita de melanoma)."""

    veu_azul_branco: str = Field(description="Valor 0 ou 1. Motivo.")
    multiplos_pontos_marrons: str = Field(description="Valor 0 ou 1. Motivo.")
    pseudopodes: str = Field(description="Valor 0 ou 1. Motivo.")
    streaming_radial: str = Field(description="Valor 0 ou 1. Motivo.")
    despigmentacao_tipo_cicatriz: str = Field(description="Valor 0 ou 1. Motivo.")
    pontos_pretos_perifericos: str = Field(description="Valor 0 ou 1. Motivo.")
    multiplas_cores: str = Field(description="Valor 0 ou 1. Motivo")


class MenziesNegativeAnswer(BaseModel):
    """Características negativas do Método Menzies (reduzem suspeita de melanoma)."""

    simetria: str = Field(description="Valor 0 ou 1. Motivo.")
    cor_unica: str = Field(description="Valor 0 ou 1. Motivo.")


class MenziesDiagnosticAnswer(BaseModel):
    """Resultado estruturado do Método Menzies para análise dermatoscópica."""

    positivas: MenziesPositiveAnswer = Field(
        description="Pontos que positivam o diagnóstico"
    )
    negativas: MenziesNegativeAnswer = Field(
        description="Pontos que negativam o diagnóstico."
    )
    explicacao: str = Field(description="Explicação das pontuações dadas.")


class SPCLDiagnosticAnswer(BaseModel):
    """Resultado estruturado da Checklist de Sete Pontos (Seven-Point Checklist)."""

    padrao_de_pigmentacao_atipico: str = Field(description="Sim ou não e motivo.")
    veu_azul_branco_irregular: str = Field(description="Sim ou não e motivo.")
    vasos_atipicos: str = Field(description="Sim ou não e motivo.")
    padrao_de_reticulo_irregular: str = Field(description="Sim ou não e motivo.")
    padrao_de_globulos_irregulares: str = Field(description="Sim ou não e motivo.")
    hiperpigmentacao_localizada: str = Field(description="Sim ou não e motivo.")
    regressao: str = Field(description="Sim ou não e motivo.")
    explicacao: str = Field(description="Explicação das pontuações dadas.")


class SummaryAnswer(BaseModel):
    """Parecer clínico consolidado integrando ABCD, Menzies e SPCL."""

    diagnostico: str = Field(
        description="Suspeita de Melanoma / Lesão Benigna / Lesão Indeterminada"
    )
    justificativa: str = Field(
        description=(
            "Integração das evidências dos algoritmos ABCD, Menzies e SPCL, "
            "destacando os principais achados."
        )
    )
    recomendacoes: str = Field(
        description=(
            "Sugestões clínicas como biópsia, acompanhamento, ou nenhuma ação imediata."
        )
    )


class GraphState(BaseModel):
    """Estado compartilhado entre os nós do grafo LangGraph."""

    image_path: str
    lesion_size: float
    # Caminho da máscara pré-computada (modo "precomputed"). None no modo "model".
    seg_path: Optional[str] = None
    image_data: Optional[str] = None
    seg_image_data: Optional[str] = None
    diagnosis_abcd: Optional[Union[str, ABCDDiagnosticAnswer]] = None
    diagnosis_menzies: Optional[Union[str, MenziesDiagnosticAnswer]] = None
    diagnosis_spcl: Optional[Union[str, SPCLDiagnosticAnswer]] = None
    validation: Optional[Union[str, SummaryAnswer]] = None
    final_report: Optional[str] = None
