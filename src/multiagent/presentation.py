"""Preparacao das tres condicoes de apresentacao de imagem (S0/S1/S2) da
matriz experimental multiagente (``docs/reconstruction/
15_multiagent_experimental_matrix.md``), usadas depois por VLM-direto,
multiagente e CNN.

Esta e a "Etapa 5, primeira entrega": SEM nenhuma chamada real a modelo
(VLM ou segmentador). O que este modulo faz:

  1. Verifica o SHA-256 do manifesto congelado de 3.000 lesoes antes de
     qualquer uso (mesmo padrao de
     ``scripts/run_s2_mask_generation.py::_verify_manifest_or_die`` --
     abortar alto e ruidoso, nunca prosseguir silenciosamente com um
     manifesto nao verificado).
  2. Resolve por convencao de nome (nunca por coluna do manifesto, que
     nao tem essa coluna) os 3 caminhos de cada caso: imagem original
     HAM10000, mascara de referencia (Tschandl) e entrada do JSON de
     geracao de S2.
  3. Produz um relatorio de pareamento que NUNCA remove nem substitui
     nada da amostra -- apenas registra incompatibilidades.
  4. Implementa UMA unica funcao de contorno (``draw_contour``), usada
     de forma identica para desenhar S1 (mascara de referencia) e S2
     (mascara predita) -- a unica diferenca entre as duas condicoes e
     qual mascara entra como argumento.
  5. Decide, por caso, se a condicao S2 esta DISPONIVEL ou INDISPONIVEL
     (e por qual motivo estrutural exato) -- nunca um booleano implicito,
     e nunca com fallback para S0/S1 quando indisponivel.
  6. Um stub de payload "dubla de VLM" que mostra exatamente o que um
     agente visual receberia por caso+condicao, provando por construcao
     que nenhum rotulo de referencia (``dx``, ``binary_target``, ou a
     propria identidade da mascara como "referencia" vs "predita")
     atravessa para o payload.

STATUS: implementada, aguardando execucao do pesquisador. Nenhuma funcao
deste modulo foi executada pelo assistente que a escreveu -- apenas
lida/revisada (regra `feedback_no_autonomous_execution`).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constantes do manifesto congelado (mesmos valores de
# scripts/run_s2_mask_generation.py -- nao redefinir com outro valor em
# nenhum outro lugar).
# ---------------------------------------------------------------------------
REGISTERED_MANIFEST_SHA256 = "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"
EXPECTED_N_CASES = 3000

# ---------------------------------------------------------------------------
# Parametros EXATOS e UNICOS do desenho de contorno -- usados identicamente
# para S1 e S2. Documentados aqui, nao espalhados em defaults de funcao.
# ---------------------------------------------------------------------------
CONTOUR_COLOR_RGB = (255, 0, 0)  # vermelho puro, canal (R,G,B) nessa ordem
CONTOUR_THICKNESS_PX = 3
CONTOUR_RETRIEVAL_MODE = cv2.RETR_EXTERNAL  # so contorno externo de cada
# blob -- trata naturalmente multiplos contornos desconexos na mesma
# mascara (um por blob), sem considerar buracos internos.
CONTOUR_APPROX_METHOD = cv2.CHAIN_APPROX_SIMPLE
CONTOUR_LINE_TYPE = cv2.LINE_8  # sem anti-aliasing (LINE_AA deliberadamente
# NAO usado) -- a borda desenhada deve ter pixels exatos, sem blending,
# para nao introduzir um artefato visual diferente entre S1 e S2 além do
# contorno em si.

# Motivos de indisponibilidade de S2 (categoria estrutural, distinta de
# qualquer recusa/indeterminacao de VLM -- nao ha VLM nesta entrega).
REASON_MASK_EMPTY = "mascara_predita_vazia"
REASON_FILE_MISSING_LOCALLY = "mascara_predita_arquivo_ausente_localmente"
REASON_FAILURE = "mascara_predita_falha"
REASON_SHAPE_MISMATCH = "mascara_predita_shape_incompativel"
REASON_NO_S2_ENTRY = "mascara_predita_sem_entrada_no_json"

# Motivos de indisponibilidade de S1 (mascara de referencia).
REASON_REF_MASK_MISSING = "mascara_referencia_ausente_ou_ilegivel"
REASON_REF_SHAPE_MISMATCH = "mascara_referencia_shape_incompativel"

# Motivo de indisponibilidade de S0 (imagem original).
REASON_IMAGE_MISSING = "imagem_original_ausente_ou_ilegivel"


# ---------------------------------------------------------------------------
# 1. Verificacao de hash do manifesto
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_or_die(manifest_path: Path) -> None:
    """Aborta (``SystemExit``) se o manifesto nao existir ou se seu
    SHA-256 nao corresponder ao registrado em
    ``docs/reconstruction/10_frozen_sample_registry.md``. Mesma politica
    de ``scripts/run_s2_mask_generation.py::_verify_manifest_or_die`` --
    nunca um "aviso" silencioso.
    """
    if not manifest_path.exists():
        raise SystemExit(f"ERRO: manifesto nao encontrado em {manifest_path}")
    actual = sha256_of_file(manifest_path)
    if actual != REGISTERED_MANIFEST_SHA256:
        raise SystemExit(
            f"ERRO FATAL: SHA-256 do manifesto NAO corresponde ao registrado.\n"
            f"  arquivo: {manifest_path}\n"
            f"  esperado: {REGISTERED_MANIFEST_SHA256}\n"
            f"  encontrado: {actual}\n"
            "Execucao abortada -- nao usar um manifesto nao verificado para "
            "preparar S0/S1/S2."
        )


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------------------
# 2. Resolucao de caminhos por convencao de nome (nunca por coluna do CSV)
# ---------------------------------------------------------------------------
def resolve_original_image_path(images_dir: Path, image_id: str) -> Path:
    return Path(images_dir) / f"{image_id}.jpg"


def resolve_reference_mask_path(ref_masks_dir: Path, image_id: str) -> Path:
    # Mesma convencao de src/data_audit/sampling.py:308-318.
    return Path(ref_masks_dir) / f"{image_id}_segmentation.png"


def index_s2_cases_by_image_id(s2_results_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``s2_results_json`` e o dict carregado de
    ``s2_mask_generation_results.json`` (chaves ``summary``/``cases``).
    Devolve um indice ``image_id -> case`` -- nao assume unicidade;
    chamadores que precisem detectar duplicatas devem usar
    ``find_duplicate_image_ids`` sobre a lista bruta de casos antes de
    indexar, se quiserem o relatorio de duplicatas do proprio JSON de S2.
    """
    cases = s2_results_json.get("cases", [])
    index: dict[str, dict[str, Any]] = {}
    for case in cases:
        index[case["image_id"]] = case
    return index


# ---------------------------------------------------------------------------
# Deteccao de duplicatas (sem remover nada)
# ---------------------------------------------------------------------------
def find_duplicate_ids(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    """Devolve ``{valor: contagem}`` apenas para valores de ``key`` que
    aparecem mais de uma vez em ``rows``. Nao modifica ``rows``.
    """
    counts: dict[str, int] = {}
    for row in rows:
        val = row[key]
        counts[val] = counts.get(val, 0) + 1
    return {val: n for val, n in counts.items() if n > 1}


# ---------------------------------------------------------------------------
# Leitura/validacao de mascara binaria
# ---------------------------------------------------------------------------
def load_binary_mask(path: Path) -> Optional[np.ndarray]:
    """Abre ``path`` como imagem e devolve uma mascara booleana (H, W)
    (``True`` = foreground, limiar > 127 em escala de cinza). Devolve
    ``None`` se o arquivo nao existir ou nao abrir como imagem valida --
    NUNCA levanta excecao para esse caso (quem chama decide o motivo
    estrutural a registrar).
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        img = Image.open(path)
        img.load()  # forca decodificacao completa agora, nao so o header
        gray = np.array(img.convert("L"))
    except Exception:
        return None
    return gray > 127


def load_original_image(path: Path) -> Optional[Image.Image]:
    """Abre a imagem original em RGB. Devolve ``None`` se ausente/ilegivel."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Funcao UNICA de contorno -- usada identicamente para S1 e S2
# ---------------------------------------------------------------------------
def draw_contour(
    image: Image.Image,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = CONTOUR_COLOR_RGB,
    thickness: int = CONTOUR_THICKNESS_PX,
) -> Image.Image:
    """Desenha o(s) contorno(s) externo(s) de ``binary_mask`` sobre uma
    copia de ``image`` e devolve a copia (``image`` original nao e
    alterada). ``binary_mask`` deve ter exatamente o mesmo ``(H, W)`` de
    ``image`` -- quem chama deve verificar isso antes (ver
    ``shapes_match``); esta funcao nao redimensiona nada.

    Parametros fixos (documentados em ``CONTOUR_*`` no topo do modulo):
    cor vermelha pura ``(255, 0, 0)`` em RGB, espessura 3 px, apenas
    contornos externos (``cv2.RETR_EXTERNAL`` -- trata multiplos blobs
    desconexos na mesma mascara desenhando um contorno por blob,
    ignorando buracos internos), sem anti-aliasing (``cv2.LINE_8``). Esta
    e a UNICA funcao de desenho de contorno do projeto; S1 e S2 SEMPRE
    passam por aqui com os mesmos parametros -- a unica diferenca entre
    as duas condicoes e qual ``binary_mask`` e passada (referencia vs
    predita).
    """
    if binary_mask.shape[:2] != (image.height, image.width):
        raise ValueError(
            f"shape da mascara {binary_mask.shape[:2]} != shape da imagem "
            f"{(image.height, image.width)} -- chamador deveria ter "
            "verificado isso antes de chamar draw_contour."
        )
    rgb_array = np.array(image.convert("RGB"))
    out_array = rgb_array.copy()
    mask_uint8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, CONTOUR_RETRIEVAL_MODE, CONTOUR_APPROX_METHOD)
    # cv2 trabalha em BGR por convencao, mas como desenhamos diretamente
    # sobre um array RGB com uma tupla de cor em RGB, o resultado sai
    # correto em RGB (cv2.drawContours nao sabe nem precisa saber qual e
    # a ordem de canal -- so escreve os 3 valores na ordem dada).
    cv2.drawContours(out_array, contours, -1, color, thickness, lineType=CONTOUR_LINE_TYPE)
    return Image.fromarray(out_array, mode="RGB")


def shapes_match(image: Image.Image, binary_mask: np.ndarray) -> bool:
    return binary_mask.shape[:2] == (image.height, image.width)


# ---------------------------------------------------------------------------
# 4/5. Pareamento por caso + decisao de disponibilidade por condicao
# ---------------------------------------------------------------------------
@dataclass
class CasePairing:
    lesion_id: str
    image_id: str
    dx: str
    binary_target: str

    original_image_path: Path
    original_image_available: bool = False
    reason_s0_unavailable: str = ""

    reference_mask_path: Path = field(default=None)  # type: ignore[assignment]
    reference_mask_available: bool = False
    reason_s1_unavailable: str = ""

    s2_json_entry_found: bool = False
    s2_status_in_json: str = ""
    s2_mask_path_in_json: str = ""
    s2_mask_empty_in_json: Optional[bool] = None
    s2_mask_available_locally: bool = False
    reason_s2_unavailable: str = ""

    duplicate_lesion_id: bool = False
    duplicate_image_id: bool = False


def build_pairing_report(
    manifest_rows: list[dict[str, str]],
    s2_cases_by_image_id: dict[str, dict[str, Any]],
    images_dir: Path,
    ref_masks_dir: Path,
    s2_masks_dir: Path,
) -> tuple[list[CasePairing], dict[str, Any]]:
    """Para cada linha do manifesto, resolve os 3 caminhos por convencao
    e registra disponibilidade/motivo por condicao. NAO remove nem
    substitui nenhuma linha -- ``manifest_rows`` e usado apenas para
    leitura.

    Devolve ``(pairings, extras)`` -- ``extras`` inclui
    ``duplicate_lesion_ids``, ``duplicate_image_ids`` e
    ``image_ids_without_s2_entry``/``s2_entries_without_manifest_row``
    (vice-versa explicitamente verificado, nao assumido vazio).
    """
    dup_lesion = find_duplicate_ids(manifest_rows, "lesion_id")
    dup_image = find_duplicate_ids(manifest_rows, "image_id")

    manifest_image_ids = {row["image_id"] for row in manifest_rows}
    s2_image_ids = set(s2_cases_by_image_id.keys())
    image_ids_without_s2_entry = sorted(manifest_image_ids - s2_image_ids)
    s2_entries_without_manifest_row = sorted(s2_image_ids - manifest_image_ids)

    pairings: list[CasePairing] = []
    for row in manifest_rows:
        image_id = row["image_id"]
        lesion_id = row["lesion_id"]

        original_path = resolve_original_image_path(images_dir, image_id)
        ref_mask_path = resolve_reference_mask_path(ref_masks_dir, image_id)

        pairing = CasePairing(
            lesion_id=lesion_id,
            image_id=image_id,
            dx=row.get("dx", ""),
            binary_target=row.get("binary_target", ""),
            original_image_path=original_path,
            reference_mask_path=ref_mask_path,
            duplicate_lesion_id=lesion_id in dup_lesion,
            duplicate_image_id=image_id in dup_image,
        )

        # --- S0: imagem original ---
        image = load_original_image(original_path)
        pairing.original_image_available = image is not None
        if image is None:
            pairing.reason_s0_unavailable = REASON_IMAGE_MISSING

        # --- S1: mascara de referencia ---
        ref_mask = load_binary_mask(ref_mask_path)
        if ref_mask is None:
            pairing.reference_mask_available = False
            pairing.reason_s1_unavailable = REASON_REF_MASK_MISSING
        elif image is not None and not shapes_match(image, ref_mask):
            pairing.reference_mask_available = False
            pairing.reason_s1_unavailable = REASON_REF_SHAPE_MISMATCH
        elif image is None:
            # Sem imagem nao ha como confirmar shape -- S1 tambem fica
            # indisponivel, mas o motivo "verdadeiro" e a ausencia da
            # imagem (ja registrado em reason_s0_unavailable); aqui so
            # marcamos a mascara como nao utilizavel por essa razao.
            pairing.reference_mask_available = False
            pairing.reason_s1_unavailable = REASON_IMAGE_MISSING
        else:
            pairing.reference_mask_available = True

        # --- S2: entrada do JSON + mascara predita local ---
        s2_case = s2_cases_by_image_id.get(image_id)
        if s2_case is None:
            pairing.s2_json_entry_found = False
            pairing.reason_s2_unavailable = REASON_NO_S2_ENTRY
        else:
            pairing.s2_json_entry_found = True
            pairing.s2_status_in_json = s2_case.get("status", "")
            pairing.s2_mask_path_in_json = s2_case.get("mask_path", "")
            pairing.s2_mask_empty_in_json = s2_case.get("mask_empty")

            if pairing.s2_status_in_json != "ok":
                pairing.reason_s2_unavailable = REASON_FAILURE
            elif pairing.s2_mask_empty_in_json:
                pairing.reason_s2_unavailable = REASON_MASK_EMPTY
            else:
                # status == "ok" e mask_empty == False no JSON -- mas o
                # JSON reflete a execucao no Mac do pesquisador, NUNCA
                # confiar ciegamente: reabrir o arquivo real nesta
                # maquina, no diretorio local informado via CLI (nao no
                # caminho absoluto do Mac gravado no JSON).
                local_mask_path = Path(s2_masks_dir) / f"{image_id}_s2_mask.png"
                s2_mask = load_binary_mask(local_mask_path)
                if s2_mask is None:
                    pairing.s2_mask_available_locally = False
                    pairing.reason_s2_unavailable = REASON_FILE_MISSING_LOCALLY
                elif image is not None and not shapes_match(image, s2_mask):
                    pairing.s2_mask_available_locally = False
                    pairing.reason_s2_unavailable = REASON_SHAPE_MISMATCH
                elif image is None:
                    pairing.s2_mask_available_locally = False
                    pairing.reason_s2_unavailable = REASON_IMAGE_MISSING
                else:
                    pairing.s2_mask_available_locally = True
                    pairing.reason_s2_unavailable = ""

        pairings.append(pairing)

    extras = {
        "duplicate_lesion_ids": dup_lesion,
        "duplicate_image_ids": dup_image,
        "image_ids_without_s2_entry": image_ids_without_s2_entry,
        "s2_entries_without_manifest_row": s2_entries_without_manifest_row,
    }
    return pairings, extras


def resolve_s2_mask_path_locally(s2_masks_dir: Path, image_id: str) -> Path:
    return Path(s2_masks_dir) / f"{image_id}_s2_mask.png"


# ---------------------------------------------------------------------------
# 6. Stub de payload "dubla de VLM" -- confirma por construcao a ausencia
#    de qualquer rotulo de referencia.
# ---------------------------------------------------------------------------
GENERIC_TASK_INSTRUCTION_PLACEHOLDER = (
    "[placeholder -- texto real da instrucao de tarefa ainda nao definido "
    "neste modulo; nenhuma chamada de VLM ocorre nesta entrega]"
)


def build_vlm_payload_stub(
    image_path: Path,
    condition: str,
    task_instruction: str = GENERIC_TASK_INSTRUCTION_PLACEHOLDER,
) -> dict[str, Any]:
    """Simula o que um agente visual (VLM) receberia para um caso+condicao
    especifico. Note que a assinatura desta funcao e deliberadamente
    incapaz de aceitar ``dx``/``binary_target``/qual-mascara-e-referencia
    como argumento -- essas informacoes nao existem no escopo desta
    funcao, entao nao ha como elas vazarem para o payload devolvido. A
    unica coisa especifica do caso e o caminho do arquivo de imagem
    daquela condicao (``image_path``, que para S0 e a imagem original,
    para S1/S2 e a imagem ja com o contorno desenhado -- em nenhum caso
    um arquivo que entregue o rotulo de origem da mascara).
    """
    return {
        "image_path": str(image_path),
        "condition": condition,
        "instruction": task_instruction,
    }


FORBIDDEN_PAYLOAD_KEYS = {"dx", "binary_target", "label", "ground_truth", "mask_type", "is_reference_mask"}


def verify_payload_has_no_labels(payload: dict[str, Any]) -> bool:
    """Confere que nenhuma chave proibida (rotulo de referencia) esta
    presente no payload dublê, e que nenhum valor textual do payload
    contem os literais ``dx``/``binary_target``/``melanoma``/``nao_melanoma``
    (defesa adicional contra vazamento acidental via string livre).
    Devolve ``True`` se o payload estiver limpo.
    """
    keys_lower = {str(k).lower() for k in payload.keys()}
    if keys_lower & FORBIDDEN_PAYLOAD_KEYS:
        return False
    forbidden_substrings = ("binary_target", "melanoma", "nao_melanoma")
    for value in payload.values():
        text = str(value).lower()
        for substr in forbidden_substrings:
            if substr in text:
                return False
    return True
