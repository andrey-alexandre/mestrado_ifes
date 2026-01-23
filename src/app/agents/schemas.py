from pydantic import BaseModel, Field
from typing import Union, Optional

class ABCDDiagnosticAnswer(BaseModel):
    assimetria: str = Field(description="Valor de 0 a 2 e motivo.")
    bordas: str = Field(description="Valor de 0 a 8 e motivo.")
    cor: str = Field(description="Valor de 1 a 6 e motivo.")
    estruturas: str = Field(description="Valor de 1 a 5 e motivo.")
    explicacao: str = Field(description="Explicação das pontuações dadas.")
    resultado_final: str = Field(description="Cálculo detalhado do resultado do algoritmo.")

class MenziesPositiveAnswer(BaseModel):
    veu_azul_branco: str = Field(description="Valor 0 ou 1. Motivo.")
    multiplos_pontos_marrons: str = Field(description="Valor 0 ou 1. Motivo.")
    pseudopodes: str = Field(description="Valor 0 ou 1. Motivo.")
    streaming_radial: str = Field(description="Valor 0 ou 1. Motivo.")
    despigmentacao_tipo_cicatriz: str = Field(description="Valor 0 ou 1. Motivo.")
    pontos_pretos_perifericos: str = Field(description="Valor 0 ou 1. Motivo.")
    multiplas_cores: str = Field(description="Valor 0 ou 1. Motivo")

class MenziesNegativeAnswer(BaseModel):
    simetria: str = Field(description="Valor 0 ou 1. Motivo.")
    cor_unica: str = Field(description="Valor 0 ou 1. Motivo.")

class MenziesDiagnosticAnswer(BaseModel):
    positivas: MenziesPositiveAnswer = Field(description="Pontos que positivam o diagnóstico")
    negativas: MenziesNegativeAnswer = Field(description="Pontos que negativam o diagnóstico.")
    explicacao: str = Field(description="Explicação das pontuações dadas.")

class SPCLDiagnosticAnswer(BaseModel):
    padrao_de_pigmentacao_atipico: str = Field(description="Sim ou não e motivo.")
    veu_azul_branco_irregular: str = Field(description="Sim ou não e motivo.")
    vasos_atipicos: str = Field(description="Sim ou não e motivo.")
    padrao_de_reticulo_irregular: str = Field(description="Sim ou não e motivo.")
    padrao_de_globulos_irregulares: str = Field(description="Sim ou não e motivo.")
    hiperpigmentacao_localizada: str = Field(description="Sim ou não e motivo.")
    regressao: str = Field(description="Sim ou não e motivo.")
    explicacao: str = Field(description="Explicação das pontuações dadas.")

class SummaryAnswer(BaseModel):
    diagnostico: str = Field(description="Suspeita de Melanoma / Lesão Benigna / Lesão Indeterminada")
    justificativa: str = Field(description="Integração das evidências dos algoritmos ABCD, Menzies e SPCL, destacando os principais achados.")
    recomendacoes: str = Field(description="Sugestões clínicas como biópsia, acompanhamento, ou nenhuma ação imediata.")

class GraphState(BaseModel):
    image_path: str
    lesion_size: float = 0.0
    image_data: Optional[str] = None
    seg_image_data: Optional[str] = None
    diagnosis_abcd1: Union[str, ABCDDiagnosticAnswer] = None
    diagnosis_abcd2: Union[str, ABCDDiagnosticAnswer] = None
    diagnosis_abcd3: Union[str, ABCDDiagnosticAnswer] = None
    validation: Union[str, SummaryAnswer, None] = None
    final_report: Optional[str] = None
