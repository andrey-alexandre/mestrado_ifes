"""Interpretação das respostas das VLMs — a MESMA regra para todos os datasets.

A resposta original (`validation.diagnostico`, texto livre) é sempre preservada. A
conversão é feita em duas camadas, ambas determinísticas e registradas por imagem:

1. `legacy_exact`: casamento por substring idêntico ao de `src/evaluation/evaluate.py`
   (`suspeita de melanoma` | `lesão indeterminada` | `lesão benigna`, nessa ordem).
2. `extended_keywords`: usada SOMENTE quando a camada 1 não reconhece o texto (ex.: o
   GPT-5.4-nano responde em texto livre). Ver `_extended`. O tier fica registrado.

Depois, a categoria é binarizada por uma política explícita:
  - `indeterminate_as_positive` (padrão do repositório: "indeterminada" -> melanoma);
  - `indeterminate_as_negative` (apenas "suspeita de melanoma" é positivo).
Respostas não avaliáveis/ambíguas NÃO são forçadas a uma classe: ficam como
`NAO_AVALIAVEL` e são contadas à parte (nas métricas, excluídas e, em sensibilidade,
tratadas como erro).
"""
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

from src.evaluation.evaluate import _DIAGNOSTICO_TO_LABEL  # regra existente do repositório

SUSPEITA, INDETERMINADA, BENIGNA, NAO_AVALIAVEL = "SUSPEITA", "INDETERMINADA", "BENIGNA", "NAO_AVALIAVEL"
CATEGORIES = (SUSPEITA, INDETERMINADA, BENIGNA, NAO_AVALIAVEL)
POLICIES = ("indeterminate_as_positive", "indeterminate_as_negative")

# chave da regra legada -> categoria (as chaves são validadas contra evaluate.py nos testes)
LEGACY_KEY_TO_CATEGORY = {
    "suspeita de melanoma": SUSPEITA,
    "lesão indeterminada": INDETERMINADA,
    "lesão benigna": BENIGNA,
}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[*_`#]+", "", t)


_NEGATED_SUSPICION = re.compile(r"\b(sem|nao|ausencia de|nenhum[a]?)\b[^.;:()]{0,30}(suspeit|suspeic|malign)")


def _extended(text: str) -> Tuple[str, str]:
    t = _norm(text)
    if "indetermin" in t:
        return INDETERMINADA, "extended_keywords"
    susp = any(k in t for k in ("suspeit", "suspeic", "malign"))
    ben = "benign" in t
    if _NEGATED_SUSPICION.search(t):
        return NAO_AVALIAVEL, "extended_keywords:negated_suspicion"
    if susp and not ben:
        return SUSPEITA, "extended_keywords"
    if ben and not susp:
        return BENIGNA, "extended_keywords"
    if susp and ben:
        return NAO_AVALIAVEL, "extended_keywords:ambiguous"
    return NAO_AVALIAVEL, "unrecognized"


def classify_diagnosis(text: Optional[str]) -> Tuple[str, str]:
    """(categoria, tier) para o texto de `validation.diagnostico`."""
    if text is None or not str(text).strip():
        return NAO_AVALIAVEL, "empty"
    low = str(text).lower()
    for key in _DIAGNOSTICO_TO_LABEL:
        if key in low:
            return LEGACY_KEY_TO_CATEGORY[key], "legacy_exact"
    return _extended(str(text))


def binarize(category: str, policy: str) -> Optional[int]:
    if policy not in POLICIES:
        raise ValueError(f"Política desconhecida: {policy}")
    if category == SUSPEITA:
        return 1
    if category == BENIGNA:
        return 0
    if category == INDETERMINADA:
        return 1 if policy == "indeterminate_as_positive" else 0
    return None


def diagnosis_text(event: Dict) -> Optional[str]:
    val = event.get("validation")
    if isinstance(val, dict):
        return val.get("diagnostico")
    return None if val is None else str(val)


# ── Critérios mencionados pelos agentes (recomputados do JSON, sem alterar a resposta) ──

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def lead_number(value: Any) -> Tuple[Optional[float], bool]:
    """Primeiro número do campo textual; `imprecise` se vier como razão/intervalo ("7/8")."""
    if value is None:
        return None, False
    s = str(value)
    m = _NUM.search(s)
    if not m:
        return None, False
    imprecise = bool(re.match(r"\s*-?\d+(?:[.,]\d+)?\s*(/|-|a|até)\s*\d", s))
    return float(m.group(0).replace(",", ".")), imprecise


def lead_yes_no(value: Any) -> Optional[int]:
    if value is None:
        return None
    t = _norm(str(value)).strip(" .:-")
    if re.match(r"^(sim|presente|yes)\b", t):
        return 1
    if re.match(r"^(nao|ausente|no)\b", t):
        return 0
    return None


ABCD_WEIGHTS = {"assimetria": 1.3, "bordas": 0.1, "cor": 0.5, "estruturas": 0.5}
ABCD_BENIGN_MAX, ABCD_SUSPECT_MAX = 4.75, 5.45          # limiares declarados no prompt
MENZIES_POS = ("veu_azul_branco", "multiplos_pontos_marrons", "pseudopodes", "streaming_radial",
               "despigmentacao_tipo_cicatriz", "pontos_pretos_perifericos", "multiplas_cores")
MENZIES_NEG = ("simetria", "cor_unica")
SPCL_MAJOR = ("padrao_de_pigmentacao_atipico", "veu_azul_branco_irregular", "vasos_atipicos")
SPCL_MINOR = ("padrao_de_reticulo_irregular", "padrao_de_globulos_irregulares",
              "hiperpigmentacao_localizada", "regressao")
SPCL_THRESHOLD = 3                                       # >= 3 pontos: suspeita


def agent_features(event: Dict) -> Dict[str, Any]:
    """Pontuações e decisões declaradas pelos agentes, recomputadas deterministicamente."""
    f: Dict[str, Any] = {}
    abcd = event.get("diagnosis_abcd")
    if isinstance(abcd, dict):
        total, ok = 0.0, True
        for k, w in ABCD_WEIGHTS.items():
            v, imp = lead_number(abcd.get(k))
            f[f"abcd_{k}"] = v
            f[f"abcd_{k}_imprecise"] = imp
            if v is None:
                ok = False
            else:
                total += w * v
        f["abcd_total"] = round(total, 4) if ok else None
        if ok:
            f["abcd_class"] = ("benigna" if total < ABCD_BENIGN_MAX else
                               "suspeita" if total <= ABCD_SUSPECT_MAX else "maligna")
    men = event.get("diagnosis_menzies")
    if isinstance(men, dict) and isinstance(men.get("positivas"), dict) and isinstance(men.get("negativas"), dict):
        pos = [lead_number(men["positivas"].get(k))[0] for k in MENZIES_POS]
        neg = [lead_number(men["negativas"].get(k))[0] for k in MENZIES_NEG]
        for k, v in zip(MENZIES_POS, pos):
            f[f"menzies_{k}"] = v
        for k, v in zip(MENZIES_NEG, neg):
            f[f"menzies_{k}"] = v
        if None not in pos and None not in neg:
            p, n = int(sum(1 for v in pos if v >= 1)), int(sum(1 for v in neg if v >= 1))
            f["menzies_positive_count"], f["menzies_negative_count"] = p, n
            f["menzies_decision"] = ("melanoma" if p >= 1 and n == 0 else
                                     "suspeita" if p >= 1 and n == 1 else
                                     "benigna" if p == 0 and n == 2 else "indeterminada")
    spcl = event.get("diagnosis_spcl")
    if isinstance(spcl, dict):
        vals = {k: lead_yes_no(spcl.get(k)) for k in SPCL_MAJOR + SPCL_MINOR}
        for k, v in vals.items():
            f[f"spcl_{k}"] = v
        if None not in vals.values():
            score = 2 * sum(vals[k] for k in SPCL_MAJOR) + sum(vals[k] for k in SPCL_MINOR)
            f["spcl_score"] = score
            f["spcl_decision"] = "suspeita" if score >= SPCL_THRESHOLD else "benigna"
    high = 0
    high += f.get("abcd_class") in ("suspeita", "maligna")
    high += f.get("menzies_decision") in ("melanoma", "suspeita")
    high += f.get("spcl_decision") == "suspeita"
    f["n_methods_high_risk"] = int(high) if any(
        k in f for k in ("abcd_class", "menzies_decision", "spcl_decision")) else None
    return f
