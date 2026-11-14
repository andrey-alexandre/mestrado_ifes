"""Protótipo EXPERIMENTAL (Etapa 5, achado do doc 22): instruções de
tarefa que pedem ao VLM-direto e ao sumarizador para DECLARAREM
explicitamente sua decisão final, no formato lido por
`src.evaluation.explicit_decision_contract.extract_decisao_final`.

Contexto do achado motivador (ver
`docs/reconstruction/22_final_answer_extraction_preprocessing.md §3`):
o classificador v2 pode capturar "benigno" dentro de uma frase que
ENUMERA pareceres conflitantes como se fosse um veredito negativo, sem
que o sumarizador tenha de fato tomado essa decisão. Em vez de ajustar
a heurística de inferência (fora do escopo desta tarefa, e
explicitamente não feito), este protótipo elimina a necessidade de
INFERIR -- pede que o modelo DECLARE a decisão numa linha dedicada.

Isolamento (garantido por desenho, não só por convenção):
  - Este módulo é NOVO e não é importado por nenhum caminho de
    produção existente. `src/agents/agents.py` (prompts oficiais),
    `src/multiagent/vlm_direct.py`
    (`VLM_DIRECT_TASK_INSTRUCTION`/`run_vlm_direct_call`/
    `run_multiagent_role_call`/`run_synthesis_control`),
    `src/evaluation/case_status_v2.py` e
    `scripts/run_local_vlm_pilot.py` (critério de avanço do Estágio 1)
    NÃO são modificados nem importados para escrita por este módulo.
  - As duas instruções abaixo (VLM-direto e sumarizador) compartilham o
    MESMO bloco de contrato (`_DECISION_CONTRACT_INSTRUCTION_BLOCK`) --
    garante que as duas rotas experimentais usem exatamente a mesma
    sintaxe, lida pelo mesmo parser nos dois casos (ver
    `scripts/verify_explicit_decision_contract.py`).

Diferença deliberada em relação ao prompt oficial do sumarizador
(`SummaryAgent.summarize`, `src/agents/agents.py:441-484`): aqui, se um
parecer (ex.: SPCL) estiver ausente (``None``), ele é OMITIDO da lista
numerada apresentada ao modelo -- o oficial sempre serializa os 3
campos do estado, inclusive quando ``None`` (produzindo um item textual
"– None"). Este é um comportamento NOVO deste protótipo experimental,
não uma correção aplicada ao prompt oficial (que permanece intocado).
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Bloco de contrato compartilhado -- ÚNICA fonte de verdade da sintaxe
# exigida nas duas instruções experimentais abaixo. A sintaxe em si
# (chave/valores/comparação) é documentada em
# `src.evaluation.explicit_decision_contract` -- este bloco só precisa
# ficar CONSISTENTE com aquele módulo, nunca redefinir a sintaxe por
# conta própria.
# ---------------------------------------------------------------------------
_DECISION_CONTRACT_INSTRUCTION_BLOCK = """
<-- Contrato de Decisão Final (EXPERIMENTAL, não oficial) -->
Ao final da sua resposta, inclua uma linha dedicada, separada do resto
do texto, exatamente no formato abaixo (sem texto adicional nessa
linha, sem marcação de negrito/lista em torno dela):

decisao_final: melanoma
(ou)
decisao_final: nao_melanoma
(ou)
decisao_final: indeterminado

Escreva essa linha com EXATAMENTE um desses três valores -- o que
corresponder à SUA decisão final de fato, não uma lista de
possibilidades. Inclua a linha exatamente UMA vez. "indeterminado" é
uma decisão válida e completa: use-a sempre que a evidência disponível
não permitir uma decisão binária -- não force uma escolha entre
melanoma/não melanoma quando a decisão genuína é de indeterminação.
"""


# ---------------------------------------------------------------------------
# VLM-direto experimental -- preserva a mesma tarefa (melanoma versus não
# melanoma a partir só da imagem) do `VLM_DIRECT_TASK_INSTRUCTION` oficial
# em `src/multiagent/vlm_direct.py` (não modificado, não importado aqui).
# Diferença: pede parecer livre + a linha `decisao_final`, em vez das 3
# categorias textuais fixas ("Suspeita de Melanoma"/"Lesão Benigna"/
# "Lesão Indeterminada") que o oficial exige ao final do texto.
# ---------------------------------------------------------------------------
EXPERIMENTAL_VLM_DIRECT_TASK_INSTRUCTION = f"""
<-- Identidade do Agente -->
Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.

<-- Instruções -->
Analise a imagem dermatoscópica fornecida para a tarefa melanoma versus
não melanoma. Dê um parecer breve em português, explicando sua
justificativa com base no que é visualmente observado.
{_DECISION_CONTRACT_INSTRUCTION_BLOCK}"""


def _serialize_for_prompt(diagnosis) -> str:
    """Serializa um diagnóstico Pydantic ou string para embutir no prompt.

    Duplicata INTENCIONAL e trivial de
    `src.agents.agents._serialize_diagnosis` (3 linhas) -- preferida a
    importar um nome privado (`_`-prefixado) de outro módulo através de
    um limite de isolamento que este protótipo deve manter explícito
    (ver docstring do módulo). Nenhuma outra lógica deste módulo depende
    de `src.agents.agents`.
    """
    if hasattr(diagnosis, "model_dump"):
        import json

        return json.dumps(diagnosis.model_dump(), ensure_ascii=False, indent=2)
    return str(diagnosis)


# ---------------------------------------------------------------------------
# Sumarizador experimental -- preserva a tarefa atual (avaliar a imagem
# dermatoscópica e os pareceres disponíveis), igual ao
# `SummaryAgent.summarize` oficial. Cenário com parecer ausente: recebe
# só os pareceres disponíveis (ver docstring do módulo).
# ---------------------------------------------------------------------------
def build_experimental_summary_task_instruction(
    diagnosis_abcd: Optional[object],
    diagnosis_menzies: Optional[object],
    diagnosis_spcl: Optional[object],
) -> str:
    """Monta a instrução de tarefa experimental do sumarizador.

    ``diagnosis_abcd``/``diagnosis_menzies``/``diagnosis_spcl`` --
    mesmos objetos que `GraphState.diagnosis_abcd`/`diagnosis_menzies`/
    `diagnosis_spcl` guardam (``ABCDDiagnosticAnswer``/
    ``MenziesDiagnosticAnswer``/``SPCLDiagnosticAnswer`` ou ``None``,
    nunca um tipo novo introduzido por este protótipo).

    Pareceres com valor ``None`` são OMITIDOS da lista numerada
    apresentada ao modelo (cenário "parecer ausente" recebe só os 2
    disponíveis) -- ver nota de design no docstring do módulo sobre a
    diferença deliberada em relação ao prompt oficial.
    """
    pareceres: list[tuple[str, str]] = []
    if diagnosis_abcd is not None:
        pareceres.append(("ABCD", _serialize_for_prompt(diagnosis_abcd)))
    if diagnosis_menzies is not None:
        pareceres.append(("Menzies", _serialize_for_prompt(diagnosis_menzies)))
    if diagnosis_spcl is not None:
        pareceres.append(
            ("SPCL (Seven Point Checklist)", _serialize_for_prompt(diagnosis_spcl))
        )

    emojis = ("1️⃣", "2️⃣", "3️⃣")
    pareceres_bloco = "\n".join(
        f"{emojis[i]} {nome} – {texto}" for i, (nome, texto) in enumerate(pareceres)
    )

    return f"""
<-- Identidade do Agente -->
Você está atuando como um sistema de apoio à pesquisa em análise de imagens dermatológicas.
Não forneça um diagnóstico ou julgamento clínico.

Você recebe a imagem segmentada da lesão acima, que realça os contornos e características internas da lesão.

<-- Instruções Específicas -->
Você está atuando como um especialista que recebeu pareceres técnicos distintos sobre uma mesma lesão cutânea, baseados nos seguintes algoritmos:
{pareceres_bloco}

Sua função é integrar os pareceres fornecidos, comparando seus resultados e justificativas, para construir um **diagnóstico consolidado e fundamentado**.

Considere:
- O grau de concordância ou conflito entre os algoritmos;
- A presença de padrões de alto risco recorrentes nos pareceres;
- A confiabilidade dos critérios observados (por exemplo, múltiplas cores, bordas irregulares e estruturas atípicas);
- A gravidade potencial com base em critérios combinados.
- Caso haja divergências significativas entre os pareceres, utilize a imagem disponibilizada para avaliar e decidir qual está mais condizente com a lesão
{_DECISION_CONTRACT_INSTRUCTION_BLOCK}"""
