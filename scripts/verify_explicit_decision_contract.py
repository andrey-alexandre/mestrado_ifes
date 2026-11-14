"""Verificação MANUAL (sem inferência, sem chamada a nenhum modelo) do
protótipo de contrato de decisão final explícito --
`src.evaluation.explicit_decision_contract.extract_decisao_final`.

Contexto: ver `docs/reconstruction/22_final_answer_extraction_preprocessing.md
§3` (achado real: v2 pode capturar "benigno" numa enumeração de
pareceres conflitantes como veredito negativo, sem que o sumarizador
tenha tomado essa decisão) e os módulos novos
`src.evaluation.explicit_decision_contract` (parser) e
`src.multiagent.experimental_decision_contract` (instruções de tarefa
experimentais, não usadas por este script -- aqui só o PARSER é
exercitado, com respostas SIMULADAS escritas à mão, para confirmar o
comportamento do parser antes de qualquer execução real).

Este script NÃO:
  - chama nenhum LLM/VLM (Ollama, OpenAI, Gemini);
  - executa o piloto das 12 lesões nem usa a amostra congelada de
    3.000 lesões;
  - modifica `case_status.py`/`case_status_v2.py`/`agents.py`/
    `vlm_direct.py` nem qualquer artefato histórico.

Cada cenário abaixo é um texto bruto SIMULADO (escrito à mão para este
teste, nunca uma resposta real de modelo) com um comentário explicando
o que o cenário exercita e qual resultado é esperado. O script imprime
o resultado obtido ao lado do esperado e um veredito PASS/FAIL por
cenário, e termina com um resumo.

Comando para rodar:

    python scripts/verify_explicit_decision_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.explicit_decision_contract import (  # noqa: E402
    DECISAO_INDETERMINADO,
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISION_STATUS_EXTRACTION_FAILED,
    DECISION_STATUS_FIELD_ABSENT,
    DECISION_STATUS_FIELD_REPEATED,
    DECISION_STATUS_INVALID_VALUE,
    DECISION_STATUS_OK,
    extract_decisao_final,
)

# ---------------------------------------------------------------------------
# Cenário 1 -- justificativa cita "benigno" dentro de uma enumeração de
# pareceres conflitantes, SEM declarar a linha `decisao_final`. Este é o
# texto (reescrito, mesmo espírito do caso real medgemma1.5/conflitante
# do doc 22) que fazia o v2 devolver "negativo" por engano. O parser do
# contrato NUNCA deve inferir nada daqui -- esperado: campo ausente.
# ---------------------------------------------------------------------------
SCENARIO_1_DISAGREEMENT_NO_FIELD = """Integração e Diagnóstico Consolidado:

Os pareceres apresentam resultados significativamente divergentes.

Conclusão: A lesão apresenta características que são discordantemente
interpretadas pelos diferentes algoritmos, com resultados que variam de
benigno (ABCD) a melanoma (Menzies) e borderline (SPCL). A presença de
múltiplas características de alto risco (Menzies) e um achado
borderline (SPCL) é um ponto crítico."""

# ---------------------------------------------------------------------------
# Cenário 2 -- mesmo desacordo do cenário 1, mas agora com a decisão
# `indeterminado` DECLARADA explicitamente. O teste NÃO impõe uma
# categoria binária esperada aqui -- o objetivo é só confirmar que o
# rótulo registrado corresponde EXATAMENTE ao que foi declarado
# (indeterminado é uma decisão válida por si só).
# ---------------------------------------------------------------------------
SCENARIO_2_DISAGREEMENT_EXPLICIT_INDETERMINADO = (
    SCENARIO_1_DISAGREEMENT_NO_FIELD
    + "\n\ndecisao_final: indeterminado"
)

# ---------------------------------------------------------------------------
# Cenário 3 -- decisão válida que DIVERGE de um dos pareceres de entrada
# (ABCD indicou benigno; a decisão final declarada é melanoma, apoiada
# pelos outros dois). De novo, nenhuma categoria é imposta a priori --
# só confirmamos que o campo lido é exatamente o declarado, mesmo quando
# o corpo do texto menciona "benigna" (prova de que o parser NUNCA lê o
# corpo para decidir, só a linha dedicada).
# ---------------------------------------------------------------------------
SCENARIO_3_VALID_DECISION_DIVERGES_FROM_ONE_OPINION = """O critério ABCD indicou lesão benigna. Os critérios Menzies e SPCL,
no entanto, indicam forte suspeita de melanoma, com múltiplos critérios
positivos e nenhum negativo. Considerando a gravidade dos achados de
Menzies e SPCL, e a imagem disponibilizada, a decisão final diverge do
parecer isolado do ABCD.

decisao_final: melanoma"""

# ---------------------------------------------------------------------------
# Cenário 4a -- campo AUSENTE (reuso do cenário 1, nomeado aqui para
# ficar explícito na lista de falhas de formato do item 4 do pedido).
# ---------------------------------------------------------------------------
SCENARIO_4A_FIELD_ABSENT = SCENARIO_1_DISAGREEMENT_NO_FIELD

# ---------------------------------------------------------------------------
# Cenário 4b -- valor INVÁLIDO (fora do vocabulário fechado de 3 tokens).
# ---------------------------------------------------------------------------
SCENARIO_4B_INVALID_VALUE = """Parecer com achados mistos, sem conclusão categórica clara pelos três
algoritmos.

decisao_final: suspeito"""

# ---------------------------------------------------------------------------
# Cenário 4c -- campo REPETIDO com valores DIFERENTES entre si.
# ---------------------------------------------------------------------------
SCENARIO_4C_FIELD_REPEATED_DIFFERENT_VALUES = """Primeira conclusão parcial do raciocínio.

decisao_final: melanoma

Reavaliação adicional após reconsiderar o parecer SPCL.

decisao_final: nao_melanoma"""

# ---------------------------------------------------------------------------
# Cenário 4d -- campo REPETIDO com valores IGUAIS entre si. Mesmo os dois
# valores coincidindo, a repetição em si é uma falha de formato (decisão
# explícita desta tarefa) -- não deve ser tratada como válida "porque
# concordam".
# ---------------------------------------------------------------------------
SCENARIO_4D_FIELD_REPEATED_EQUAL_VALUES = """Primeira conclusão parcial do raciocínio.

decisao_final: melanoma

Confirmação da mesma conclusão após reavaliação.

decisao_final: melanoma"""

# ---------------------------------------------------------------------------
# Cenário 5 -- bloco de template (pensamento interno) contém um rótulo
# DIFERENTE do campo declarado na resposta final de fato. Simula o
# formato real de vazamento de template do medgemma1.5
# (`<unused94>thought ... <unused95>`, ver
# docs/reconstruction/22_final_answer_extraction_preprocessing.md) com um
# rascunho mental que menciona "melanoma" dentro do bloco de pensamento,
# mas a resposta final (depois do delimitador de fechamento) declara
# "nao_melanoma". Esperado: a extração do template roda PRIMEIRO, e o
# parser só vê a linha da resposta final -- o rótulo do bloco de
# pensamento nunca deve ser lido.
# ---------------------------------------------------------------------------
SCENARIO_5_TEMPLATE_BLOCK_DIFFERENT_LABEL_THAN_FINAL_ANSWER = (
    "<unused94>thought\n"
    "Pensando em voz alta: pelos critérios ABCD isolados isto poderia "
    "ser melanoma. decisao_final: melanoma -- mas isto é só um rascunho "
    "mental, não a resposta final, preciso considerar os outros dois "
    "pareceres antes de concluir.<unused95>"
    "Resposta final: considerando o conjunto dos três critérios, a "
    "lesão é benigna e não apresenta padrões suspeitos relevantes.\n\n"
    "decisao_final: nao_melanoma"
)


SCENARIOS: list[dict] = [
    {
        "name": "1_desacordo_benigno_sem_campo",
        "description": (
            "Justificativa cita 'benigno' numa enumeração de pareceres "
            "conflitantes, sem declarar decisao_final."
        ),
        "raw_text": SCENARIO_1_DISAGREEMENT_NO_FIELD,
        "expected_decision_status": DECISION_STATUS_FIELD_ABSENT,
        "expected_decisao_final": None,
    },
    {
        "name": "2_desacordo_indeterminado_explicito",
        "description": "Mesmo desacordo, com decisao_final: indeterminado declarado.",
        "raw_text": SCENARIO_2_DISAGREEMENT_EXPLICIT_INDETERMINADO,
        "expected_decision_status": DECISION_STATUS_OK,
        "expected_decisao_final": DECISAO_INDETERMINADO,
    },
    {
        "name": "3_decisao_valida_diverge_de_um_parecer",
        "description": (
            "Decisão declarada (melanoma) diverge do parecer isolado do "
            "ABCD (benigna), mencionado no corpo do texto."
        ),
        "raw_text": SCENARIO_3_VALID_DECISION_DIVERGES_FROM_ONE_OPINION,
        "expected_decision_status": DECISION_STATUS_OK,
        "expected_decisao_final": DECISAO_MELANOMA,
    },
    {
        "name": "4a_campo_ausente",
        "description": "Nenhuma linha decisao_final no texto.",
        "raw_text": SCENARIO_4A_FIELD_ABSENT,
        "expected_decision_status": DECISION_STATUS_FIELD_ABSENT,
        "expected_decisao_final": None,
    },
    {
        "name": "4b_valor_invalido",
        "description": "decisao_final: suspeito -- fora do vocabulário fechado.",
        "raw_text": SCENARIO_4B_INVALID_VALUE,
        "expected_decision_status": DECISION_STATUS_INVALID_VALUE,
        "expected_decisao_final": None,
    },
    {
        "name": "4c_campo_repetido_valores_diferentes",
        "description": "decisao_final aparece 2x, com valores diferentes.",
        "raw_text": SCENARIO_4C_FIELD_REPEATED_DIFFERENT_VALUES,
        "expected_decision_status": DECISION_STATUS_FIELD_REPEATED,
        "expected_decisao_final": None,
    },
    {
        "name": "4d_campo_repetido_valores_iguais",
        "description": (
            "decisao_final aparece 2x, com o MESMO valor -- ainda assim "
            "é falha de formato (repetição), não deve ser aceito."
        ),
        "raw_text": SCENARIO_4D_FIELD_REPEATED_EQUAL_VALUES,
        "expected_decision_status": DECISION_STATUS_FIELD_REPEATED,
        "expected_decisao_final": None,
    },
    {
        "name": "5_template_com_rotulo_diferente_da_resposta_final",
        "description": (
            "Bloco de pensamento menciona 'melanoma' informalmente; a "
            "resposta final (pós-template) declara decisao_final: "
            "nao_melanoma. Só o rótulo da resposta final deve ser lido."
        ),
        "raw_text": SCENARIO_5_TEMPLATE_BLOCK_DIFFERENT_LABEL_THAN_FINAL_ANSWER,
        "expected_decision_status": DECISION_STATUS_OK,
        "expected_decisao_final": DECISAO_NAO_MELANOMA,
    },
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for scenario in SCENARIOS:
        result = extract_decisao_final(scenario["raw_text"])
        status_ok = result.decision_status == scenario["expected_decision_status"]
        value_ok = result.decisao_final == scenario["expected_decisao_final"]
        verdict = "PASS" if (status_ok and value_ok) else "FAIL"
        n_pass += verdict == "PASS"
        n_fail += verdict == "FAIL"

        print(f"[{verdict}] {scenario['name']}")
        print(f"    descrição: {scenario['description']}")
        print(f"    template_extraction_status: {result.template_extraction_status}")
        print(
            "    decision_status: "
            f"obtido={result.decision_status!r} esperado={scenario['expected_decision_status']!r}"
        )
        print(
            "    decisao_final:   "
            f"obtido={result.decisao_final!r} esperado={scenario['expected_decisao_final']!r}"
        )
        print(f"    matched_raw_values: {result.matched_raw_values!r}")
        print()

    print(f"Resumo: {n_pass} PASS, {n_fail} FAIL, {len(SCENARIOS)} cenários no total.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
