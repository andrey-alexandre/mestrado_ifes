"""Etapa 5 — verificação empírica de que o novo mecanismo de timeout
(`src.agents.agents._invoke_ollama_stream_with_deadline`, ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §9/§10) encerra
de verdade a chamada ao Ollama, em vez de só desistir de esperar.

Configurar um timeout no cliente (`client_kwargs={"timeout": N}`) NÃO
prova por si só que a geração para -- `ChatOllama` sempre streama
internamente mesmo em `.invoke()` (`stream=True` por padrão em
`_chat_params`), então um timeout por LEITURA nunca dispara enquanto o
modelo continuar emitindo tokens, mesmo que lentamente e mesmo que o
tempo TOTAL já tenha passado muito do limite. Este script não assume
nada sobre isso -- mede.

Método (evidência por latência, NUNCA por `ollama.ps()`/estado interno
do servidor -- decisão explícita do pesquisador: não usar `ollama ps`
como indicador de servidor ocioso):

  1. Baseline: envia um prompt trivial e rápido (``--fast-prompt``)
     algumas vezes, SEM prazo (``.invoke()`` comum), mede a latência
     média real deste modelo nesta máquina agora.
  2. Força o corte: envia um prompt deliberadamente longo
     (``--slow-prompt``) através de
     `_invoke_ollama_stream_with_deadline` com um prazo CURTO
     (``--deadline-s``, bem menor que o tempo natural de geração do
     prompt longo) -- espera que `OllamaStreamDeadlineExceeded` seja
     levantada perto de ``--deadline-s`` segundos (confirma que o NOSSO
     código detecta e age no prazo, não só que configuramos um número).
  3. Imediatamente depois, repete o prompt trivial (``--fast-prompt``)
     algumas vezes e mede a latência.
  4. Compara: se a latência pós-corte for parecida com o baseline, é
     evidência de que o servidor foi liberado de verdade (cancelamento
     real). Se for muito maior (ex.: parecida com o tempo que faltava
     para o prompt longo terminar), é evidência de que a geração
     abandonada ainda está ocupando o Ollama -- o cancelamento no nosso
     lado não encerrou nada do lado do servidor.

Isto é evidência circunstancial, não uma prova formal -- o script deixa
isso explícito no resultado (``interpretation_caveat``), nunca imprime
um "PASSOU"/"FALHOU" definitivo sem o contexto completo.

STATUS: implementado, NÃO executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) -- requer Ollama real rodando
localmente com o modelo já baixado.

Uso:
    python scripts/verify_ollama_real_cancellation.py --model-tag medgemma1.5:4b-it-q4_K_M
    python scripts/verify_ollama_real_cancellation.py --model-tag qwen3-vl:8b-instruct-q4_K_M --deadline-s 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover
    from langchain.schema import HumanMessage

from src.agents.agents import (  # noqa: E402
    OllamaStreamDeadlineExceeded,
    _build_llm,
    _invoke_ollama_stream_with_deadline,
)

DEFAULT_FAST_PROMPT = "Responda apenas a palavra: OK"
DEFAULT_SLOW_PROMPT = (
    "Escreva um ensaio longo e detalhado (pelo menos 1500 palavras) em "
    "português sobre a história da dermatoscopia, cobrindo origem, "
    "evolução dos equipamentos, principais critérios diagnósticos "
    "(ABCD, Menzies, Seven Point Checklist) e o estado atual da "
    "pesquisa em diagnóstico assistido por IA. Seja exaustivo e "
    "detalhado em cada seção."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", required=True, help="Tag Ollama, ex.: medgemma1.5:4b-it-q4_K_M")
    parser.add_argument("--deadline-s", type=float, default=8.0, help="Prazo CURTO para forçar o corte (deve ser bem menor que o tempo natural do --slow-prompt)")
    parser.add_argument("--fast-prompt", default=DEFAULT_FAST_PROMPT)
    parser.add_argument("--slow-prompt", default=DEFAULT_SLOW_PROMPT)
    parser.add_argument("--n-baseline-samples", type=int, default=2)
    parser.add_argument("--n-post-timeout-samples", type=int, default=3)
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "experiments" / "multiagent" / "verify_ollama_real_cancellation_results.json"),
    )
    return parser


def _timed_fast_call(llm, prompt: str) -> float:
    """Chamada simples, SEM prazo (``.invoke()`` comum) -- usada só para
    medir latência de referência, nunca para forçar um corte."""
    t0 = time.monotonic()
    llm.invoke([HumanMessage(content=prompt)])
    return time.monotonic() - t0


def main() -> None:
    args = build_arg_parser().parse_args()
    model_name = args.model_tag

    print(f"=== Verificação de cancelamento real -- modelo: {model_name} ===")
    print(f"prazo curto forçado: {args.deadline_s}s (deve ser bem menor que o tempo natural do prompt longo)")

    # LLM sem prazo (client_kwargs vazio) -- usado só para o baseline e
    # para o prompt lento via função de prazo manual (que impõe o prazo
    # por fora, não pelo client_kwargs do httpx).
    llm_no_deadline = _build_llm(model_name, temperature=0)

    print(f"\n--- 1. Baseline ({args.n_baseline_samples}x prompt rápido, sem prazo) ---")
    baseline_samples = []
    for i in range(args.n_baseline_samples):
        elapsed = _timed_fast_call(llm_no_deadline, args.fast_prompt)
        baseline_samples.append(elapsed)
        print(f"  amostra {i + 1}: {elapsed:.2f}s")
    baseline_mean = sum(baseline_samples) / len(baseline_samples)
    print(f"  média baseline: {baseline_mean:.2f}s")

    print(f"\n--- 2. Forçando corte no prompt longo (prazo={args.deadline_s}s) ---")
    t0 = time.monotonic()
    forced_cutoff_fired = False
    forced_elapsed_s = None
    forced_error = None
    try:
        _invoke_ollama_stream_with_deadline(
            llm_no_deadline, [HumanMessage(content=args.slow_prompt)], timeout_s=args.deadline_s
        )
        forced_elapsed_s = time.monotonic() - t0
        print(
            f"  AVISO: o prompt longo terminou sozinho em {forced_elapsed_s:.2f}s, ANTES do prazo "
            f"de {args.deadline_s}s disparar -- aumente --deadline-s ou use um --slow-prompt mais "
            "longo para este teste ser conclusivo (o corte nunca foi exercitado de verdade)."
        )
    except OllamaStreamDeadlineExceeded as exc:
        forced_elapsed_s = time.monotonic() - t0
        forced_cutoff_fired = True
        forced_error = str(exc)
        print(f"  corte disparado em {forced_elapsed_s:.2f}s (prazo pedido: {args.deadline_s}s) -- {forced_error}")
    except Exception as exc:  # noqa: BLE001 -- qualquer outra falha real também é informação útil aqui
        forced_elapsed_s = time.monotonic() - t0
        forced_error = f"{type(exc).__name__}: {exc}"
        print(f"  EXCEÇÃO INESPERADA em {forced_elapsed_s:.2f}s -- {forced_error}")

    print(f"\n--- 3. Prompt rápido imediatamente após o corte ({args.n_post_timeout_samples}x) ---")
    post_timeout_samples = []
    for i in range(args.n_post_timeout_samples):
        elapsed = _timed_fast_call(llm_no_deadline, args.fast_prompt)
        post_timeout_samples.append(elapsed)
        print(f"  amostra {i + 1}: {elapsed:.2f}s")
    post_timeout_mean = sum(post_timeout_samples) / len(post_timeout_samples)
    print(f"  média pós-corte: {post_timeout_mean:.2f}s")

    ratio = (post_timeout_mean / baseline_mean) if baseline_mean > 0 else None
    result: dict[str, Any] = {
        "model_name": model_name,
        "deadline_s_requested": args.deadline_s,
        "baseline_samples_s": baseline_samples,
        "baseline_mean_s": baseline_mean,
        "forced_cutoff_fired": forced_cutoff_fired,
        "forced_cutoff_elapsed_s": forced_elapsed_s,
        "forced_cutoff_error": forced_error,
        "post_timeout_samples_s": post_timeout_samples,
        "post_timeout_mean_s": post_timeout_mean,
        "post_timeout_to_baseline_ratio": ratio,
        "interpretation_caveat": (
            "Evidência CIRCUNSTANCIAL por latência, não uma prova formal. Se "
            "post_timeout_mean_s for próximo de baseline_mean_s (ratio perto de 1), é "
            "evidência de que o servidor foi liberado de verdade pelo nosso corte. Se "
            "for muito maior, é evidência de que a geração abandonada ainda ocupava o "
            "Ollama -- o corte no nosso lado não encerrou nada do lado do servidor. "
            "Nunca use ollama.ps()/estado interno do servidor para decidir isto -- "
            "decisão explícita do pesquisador."
        ),
    }
    if not forced_cutoff_fired:
        result["warning"] = (
            "O corte NUNCA disparou de verdade (prompt longo terminou sozinho antes do "
            "prazo, ou houve uma exceção inesperada) -- este resultado NÃO testa o "
            "cancelamento real. Reexecute com --deadline-s menor e/ou --slow-prompt mais "
            "longo."
        )
        print(f"\n*** {result['warning']} ***")
    elif ratio is not None:
        print(f"\nrazão pós-corte/baseline: {ratio:.2f}x")
        if ratio <= 2.0:
            print("Evidência consistente com cancelamento REAL (latência pós-corte próxima do baseline).")
        else:
            print(
                "Evidência de que o servidor pode NÃO ter sido liberado (latência pós-corte "
                "bem acima do baseline) -- investigar mais antes de confiar no cancelamento."
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nresultado completo gravado em: {output_path}")


if __name__ == "__main__":
    main()
