"""Execução do pipeline multiagente ORIGINAL sobre o piloto (chamadas pagas só com --run-vlm).

Reutiliza, sem modificá-los, `build_graph` (agentes e prompts) e `run_pipeline`. Os
únicos ajustes de ambiente — todos documentados em docs/dataset_validation/audit.md —
são feitos em tempo de execução e não alteram modelos, prompts, temperatura nem
interpretação:

* `Image_Segmentation.code.network` indisponível -> stub inerte (a U-Net não é usada em
  modo `precomputed`);
* crítico RAG -> recuperador nulo (a saída `final_report` não entra na avaliação e a
  construção do RAG exigiria baixar o modelo de embeddings);
* credenciais: `_build_llm` do repositório passa strings literais como chave; aqui os
  construtores `ChatOpenAI`/`ChatGoogleGenerativeAI` recebem a chave das variáveis de
  ambiente OPENAI_API_KEY / GOOGLE_API_KEY.

Cada imagem é processada e persistida individualmente (retomada sem repetir chamadas).
"""
import logging
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.dataset_validation.io_utils import provenance, read_json, sha256_file, write_json

logger = logging.getLogger(__name__)

ImageEntry = tuple  # (image_path, lesion_size_mm, target_label, seg_path)


class InferenceError(RuntimeError):
    """Pré-condição de execução não satisfeita."""


class NullRetriever:
    """Recuperador nulo: preserva a topologia do grafo sem RAG."""

    def invoke(self, query: str):  # noqa: ARG002
        return [types.SimpleNamespace(page_content="[RAG desativado no piloto dataset_validation]")]


def install_unet_stub_if_missing() -> bool:
    """Instala um stub de `Image_Segmentation.code.network` se o pacote não existir."""
    try:
        import Image_Segmentation.code.network  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def _unavailable(name):
        class _Stub:  # pragma: no cover - nunca instanciada em modo precomputed
            def __init__(self, *a, **k):
                raise RuntimeError(f"{name} indisponível: piloto usa segmentação 'precomputed'.")
        _Stub.__name__ = name
        return _Stub

    pkg = types.ModuleType("Image_Segmentation")
    code = types.ModuleType("Image_Segmentation.code")
    net = types.ModuleType("Image_Segmentation.code.network")
    for n in ("U_Net", "R2U_Net", "AttU_Net", "R2AttU_Net"):
        setattr(net, n, _unavailable(n))
    pkg.code, code.network = code, net
    sys.modules.update({"Image_Segmentation": pkg, "Image_Segmentation.code": code,
                        "Image_Segmentation.code.network": net})
    return True


def inject_credentials(agents_module, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Faz os construtores de LLM usarem chaves de ambiente (sem alterar o repositório)."""
    env = os.environ if env is None else env
    wrapped = {}
    for attr, kwarg, var in (("ChatOpenAI", "openai_api_key", "OPENAI_API_KEY"),
                             ("ChatGoogleGenerativeAI", "google_api_key", "GOOGLE_API_KEY")):
        real = getattr(agents_module, attr)

        def factory(*a, _real=real, _kw=kwarg, _var=var, **k):
            key = env.get(_var)
            if not key:
                raise InferenceError(f"Variável de ambiente {_var} ausente (necessária para {_real.__name__}).")
            k[_kw] = key
            return _real(*a, **k)

        setattr(agents_module, attr, factory)
        wrapped[attr] = var
    return wrapped


def safe_name(model_name: str) -> str:
    """Nome de modelo utilizável em caminhos (':' é inválido no Windows)."""
    return model_name.replace(":", "-").replace("/", "-")


def provider_env_var(model_name: str) -> Optional[str]:
    if ":" in model_name:
        return None                      # Ollama local
    if model_name.startswith("gemini"):
        return "GOOGLE_API_KEY"
    return "OPENAI_API_KEY"


def build_entries(manifest: Sequence[Dict], order_seed: int) -> List[ImageEntry]:
    """Entradas no formato de `run_pipeline`, em ordem embaralhada e reprodutível."""
    rows = sorted(manifest, key=lambda r: r["sample_id"])
    random.Random(f"{order_seed}:inference_order").shuffle(rows)
    return [(r["image_path"], 0.0, int(r["binary_label"]), r["mask_path"]) for r in rows]


def estimate_cost(cfg: Dict, n_images: int) -> Dict:
    inf = cfg["inference"]
    a = inf["assumptions"]
    n_diag = inf["calls_per_image"] - 1
    in_tok = n_diag * a["input_tokens_per_diagnostic_call"] + a["input_tokens_summary_call"]
    out_tok = n_diag * a["output_tokens_per_diagnostic_call"] + a["output_tokens_summary_call"]
    per_model = {
        "images": n_images, "llm_calls": n_images * inf["calls_per_image"],
        "input_tokens_est": n_images * in_tok, "output_tokens_est": n_images * out_tok,
    }
    pin, pout = a.get("price_per_1m_input_usd"), a.get("price_per_1m_output_usd")
    models = {}
    for m in inf["models"]:
        entry = dict(per_model)
        entry["cost_usd_est"] = (round(per_model["input_tokens_est"] / 1e6 * pin
                                       + per_model["output_tokens_est"] / 1e6 * pout, 4)
                                 if pin is not None and pout is not None else None)
        entry["credential_env_var"] = provider_env_var(m)
        entry["credential_present"] = bool(os.environ.get(entry["credential_env_var"] or "", ""))
        models[m] = entry
    return {
        "per_model": models,
        "total_llm_calls": per_model["llm_calls"] * len(models),
        "assumptions": a,
        "note": "Estimativa por premissas (tokens por chamada); NÃO é medição. Preço unitário a "
                "preencher em inference.assumptions.price_per_1m_*_usd.",
    }


def default_graph_factory(cfg: Dict, model_name: str) -> Callable:
    def factory():
        install_unet_stub_if_missing()
        import src.agents.agents as agents_module
        inject_credentials(agents_module)
        return agents_module.build_graph(
            retriever=NullRetriever(), model_name=model_name,
            temperature=cfg["inference"]["temperature"],
            segmentation_mode="precomputed", seg_model=None,
        )
    return factory


def run_model(cfg: Dict, manifest: Sequence[Dict], model_name: str, out_dir: Path,
              graph_factory: Optional[Callable] = None, pipeline_runner: Optional[Callable] = None,
              attempts: int = 3, backoff_s: float = 10.0, sleep: Callable = time.sleep,
              limit: Optional[int] = None) -> str:
    """Executa (ou retoma) um modelo sobre o manifesto e consolida o JSON de resultados.

    Cada imagem é persistida individualmente (`raw/<modelo>/`), então reexecutar não repete
    chamadas já concluídas. Falhas transitórias são repetidas (`attempts`); imagens que
    continuam falhando são registradas em `*.meta.json` e não abortam o lote.
    """
    env_var = provider_env_var(model_name)
    if graph_factory is None and pipeline_runner is None and env_var and not os.environ.get(env_var):
        raise InferenceError(f"{env_var} não definida: impossível chamar '{model_name}'.")
    safe = safe_name(model_name)
    raw_dir = out_dir / "raw" / safe
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = build_entries(manifest, cfg["seed"])
    if limit:
        entries = entries[:limit]
    by_path = {r["image_path"]: r for r in manifest}
    graph = None
    failures: Dict[str, str] = {}
    for n, entry in enumerate(entries, start=1):
        row = by_path[entry[0]]
        part = raw_dir / f"{row['dataset']}__{row['image_id']}.json"
        if part.exists():
            continue
        for attempt in range(1, attempts + 1):
            try:
                if pipeline_runner is None:
                    from src.pipeline.pipeline import run_pipeline
                    graph = graph or (graph_factory or default_graph_factory(cfg, model_name))()
                    res_path = run_pipeline(graph, [entry], str(raw_dir / "_tmp"), model_name,
                                            filename_prefix=f"{row['dataset']}__{row['image_id']}",
                                            date_format="")
                    data = read_json(res_path)
                else:
                    data = pipeline_runner([entry])
                write_json(part, data)
                logger.info("[%s] %d/%d %s ok", model_name, n, len(entries), row["sample_id"])
                break
            except InferenceError:
                raise
            except Exception as exc:  # noqa: BLE001 - registrar e seguir
                logger.warning("[%s] %s tentativa %d/%d falhou: %s", model_name, row["sample_id"],
                               attempt, attempts, exc)
                if attempt == attempts:
                    failures[row["sample_id"]] = f"{type(exc).__name__}: {exc}"
                else:
                    sleep(backoff_s * attempt)
    results: Dict = {}
    for entry in entries:
        r = by_path[entry[0]]
        part = raw_dir / f"{r['dataset']}__{r['image_id']}.json"
        if not part.exists():
            continue
        for path, ev in read_json(part).items():
            # identificadores independentes da máquina (os caminhos mudam entre ambientes)
            ev["_meta"] = {"sample_id": r["sample_id"], "dataset": r["dataset"],
                           "image_id": r["image_id"], "model": model_name}
            results[path] = ev
    out_path = out_dir / f"results_{safe}.json"
    write_json(out_path, results)
    root = Path(__file__).resolve().parents[1]
    meta = {
        "model_name": model_name, "temperature": cfg["inference"]["temperature"],
        "segmentation_mode": "precomputed", "n_images_requested": len(entries),
        "n_images_completed": len(results), "failures": failures,
        "results_sha256": sha256_file(out_path), "provenance": provenance(cfg["seed"]),
        "agents_sha256": sha256_file(root / "agents" / "agents.py"),
        "schemas_sha256": sha256_file(root / "agents" / "schemas.py"),
    }
    write_json(out_dir / f"results_{safe}.meta.json", meta)
    return str(out_path)
