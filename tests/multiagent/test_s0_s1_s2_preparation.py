"""Testes com dados sinteticos (nenhuma dependencia dos datasets reais)
para `src/multiagent/presentation.py` -- preparacao de S0/S1/S2 (Etapa 5,
primeira entrega, sem chamada a VLM).

STATUS: implementado, aguardando execucao do pesquisador
(`python -m pytest tests/multiagent/test_s0_s1_s2_preparation.py -v`).
Nao executado pelo assistente que escreveu este arquivo.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from src.multiagent.presentation import (
    REASON_FAILURE,
    REASON_FILE_MISSING_LOCALLY,
    REASON_MASK_EMPTY,
    REASON_REF_SHAPE_MISMATCH,
    REASON_SHAPE_MISMATCH,
    S2PayloadUnavailableError,
    build_pairing_report,
    build_real_vlm_payload,
    build_vlm_payload_stub,
    decode_payload_image,
    draw_contour,
    find_duplicate_ids,
    images_are_clearly_different,
    images_are_effectively_equal,
    load_binary_mask,
    payload_text_has_no_labels,
    require_s2_payload_available,
    sha256_of_file,
    verify_manifest_or_die,
    verify_payload_has_no_labels,
)


# ---------------------------------------------------------------------------
# Helpers de dados sinteticos
# ---------------------------------------------------------------------------
def _make_image(w: int = 20, h: int = 16, color=(100, 150, 200)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _make_mask(w: int, h: int, box) -> np.ndarray:
    """Mascara booleana (H, W) com um retangulo True definido por
    ``box = (x0, y0, x1, y1)``.
    """
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    arr = (mask.astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# draw_contour: S1 e S2 usam a MESMA funcao -- resultado deve ser
# byte-a-byte identico dado o mesmo input.
# ---------------------------------------------------------------------------
def test_draw_contour_identical_for_s1_and_s2_given_same_mask():
    image = _make_image()
    mask = _make_mask(20, 16, (4, 3, 12, 10))

    result_as_if_s1 = draw_contour(image, mask)
    result_as_if_s2 = draw_contour(image, mask)

    assert np.array_equal(np.array(result_as_if_s1), np.array(result_as_if_s2))
    # e deve ter de fato desenhado algo (nao e so a imagem original intacta)
    assert not np.array_equal(np.array(result_as_if_s1), np.array(image))


def test_draw_contour_does_not_mutate_input_image():
    image = _make_image()
    mask = _make_mask(20, 16, (4, 3, 12, 10))
    original_bytes = np.array(image).copy()

    draw_contour(image, mask)

    assert np.array_equal(np.array(image), original_bytes)


def test_draw_contour_rejects_shape_mismatch():
    image = _make_image(20, 16)
    wrong_shape_mask = _make_mask(10, 10, (1, 1, 5, 5))
    with pytest.raises(ValueError):
        draw_contour(image, wrong_shape_mask)


def test_draw_contour_handles_multiple_disconnected_blobs():
    image = _make_image(30, 30)
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:6, 2:6] = True
    mask[20:25, 20:25] = True  # segundo blob desconexo

    result = draw_contour(image, mask)
    assert not np.array_equal(np.array(result), np.array(image))


# ---------------------------------------------------------------------------
# Hash/verificacao do manifesto
# ---------------------------------------------------------------------------
def test_verify_manifest_or_die_aborts_on_hash_mismatch(tmp_path):
    manifest_path = tmp_path / "fake_manifest.csv"
    manifest_path.write_text("lesion_id,image_id,dx,binary_target\nHAM_1,ISIC_1,nv,nao_melanoma\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        verify_manifest_or_die(manifest_path)


def test_verify_manifest_or_die_aborts_on_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        verify_manifest_or_die(tmp_path / "does_not_exist.csv")


def test_sha256_of_file_matches_hashlib_reference(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("conteudo qualquer", encoding="utf-8")
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha256_of_file(p) == expected


# ---------------------------------------------------------------------------
# Duplicatas no manifesto -- deteccao sem remocao
# ---------------------------------------------------------------------------
def test_find_duplicate_ids_detects_without_removing():
    rows = [
        {"lesion_id": "HAM_1", "image_id": "ISIC_1"},
        {"lesion_id": "HAM_2", "image_id": "ISIC_2"},
        {"lesion_id": "HAM_1", "image_id": "ISIC_3"},  # lesion_id duplicado
    ]
    dups = find_duplicate_ids(rows, "lesion_id")
    assert dups == {"HAM_1": 2}
    # a lista original nao foi alterada
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# load_binary_mask: arquivo ausente/ilegivel -> None, nunca excecao
# ---------------------------------------------------------------------------
def test_load_binary_mask_returns_none_for_missing_file(tmp_path):
    assert load_binary_mask(tmp_path / "nope.png") is None


def test_load_binary_mask_returns_none_for_corrupt_file(tmp_path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not a real png file")
    assert load_binary_mask(bad) is None


def test_load_binary_mask_reads_valid_mask(tmp_path):
    mask = _make_mask(10, 8, (1, 1, 5, 5))
    path = tmp_path / "mask.png"
    _save_mask_png(mask, path)
    loaded = load_binary_mask(path)
    assert loaded is not None
    assert np.array_equal(loaded, mask)


# ---------------------------------------------------------------------------
# build_pairing_report: cenarios reais exigidos pelo pesquisador, com
# dados 100% sinteticos.
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_project(tmp_path):
    images_dir = tmp_path / "images"
    ref_masks_dir = tmp_path / "ref_masks"
    s2_masks_dir = tmp_path / "s2_masks"
    for d in (images_dir, ref_masks_dir, s2_masks_dir):
        d.mkdir()

    w, h = 20, 16

    # Caso 1: tudo disponivel (S0, S1, S2 ok)
    img1 = _make_image(w, h)
    img1.save(images_dir / "ISIC_0000001.jpg")
    mask1 = _make_mask(w, h, (2, 2, 10, 10))
    _save_mask_png(mask1, ref_masks_dir / "ISIC_0000001_segmentation.png")
    _save_mask_png(mask1, s2_masks_dir / "ISIC_0000001_s2_mask.png")

    # Caso 2: S2 com mascara vazia (mask_empty=True no JSON) -- S0/S1 ok
    img2 = _make_image(w, h)
    img2.save(images_dir / "ISIC_0000002.jpg")
    mask2 = _make_mask(w, h, (1, 1, 6, 6))
    _save_mask_png(mask2, ref_masks_dir / "ISIC_0000002_segmentation.png")
    # nao cria arquivo de s2 para o caso 2 -- irrelevante, json ja diz vazia

    # Caso 3: S2 "ok" no JSON mas arquivo ausente localmente
    img3 = _make_image(w, h)
    img3.save(images_dir / "ISIC_0000003.jpg")
    mask3 = _make_mask(w, h, (1, 1, 6, 6))
    _save_mask_png(mask3, ref_masks_dir / "ISIC_0000003_segmentation.png")
    # arquivo de s2 deliberadamente NAO criado, apesar do JSON dizer "ok"

    # Caso 4: S2 com shape incompativel (mascara de outra resolucao)
    img4 = _make_image(w, h)
    img4.save(images_dir / "ISIC_0000004.jpg")
    mask4 = _make_mask(w, h, (1, 1, 6, 6))
    _save_mask_png(mask4, ref_masks_dir / "ISIC_0000004_segmentation.png")
    wrong_shape_mask = _make_mask(w + 5, h + 5, (1, 1, 6, 6))
    _save_mask_png(wrong_shape_mask, s2_masks_dir / "ISIC_0000004_s2_mask.png")

    # Caso 5: status "falha" registrado no JSON de S2
    img5 = _make_image(w, h)
    img5.save(images_dir / "ISIC_0000005.jpg")
    mask5 = _make_mask(w, h, (1, 1, 6, 6))
    _save_mask_png(mask5, ref_masks_dir / "ISIC_0000005_segmentation.png")

    # Caso 6: mascara de referencia ausente (S1 indisponivel), S2 ok
    img6 = _make_image(w, h)
    img6.save(images_dir / "ISIC_0000006.jpg")
    _save_mask_png(mask1, s2_masks_dir / "ISIC_0000006_s2_mask.png")

    manifest_rows = [
        {"lesion_id": "HAM_0000001", "image_id": "ISIC_0000001", "dx": "nv", "binary_target": "nao_melanoma"},
        {"lesion_id": "HAM_0000002", "image_id": "ISIC_0000002", "dx": "mel", "binary_target": "melanoma"},
        {"lesion_id": "HAM_0000003", "image_id": "ISIC_0000003", "dx": "nv", "binary_target": "nao_melanoma"},
        {"lesion_id": "HAM_0000004", "image_id": "ISIC_0000004", "dx": "nv", "binary_target": "nao_melanoma"},
        {"lesion_id": "HAM_0000005", "image_id": "ISIC_0000005", "dx": "nv", "binary_target": "nao_melanoma"},
        {"lesion_id": "HAM_0000006", "image_id": "ISIC_0000006", "dx": "nv", "binary_target": "nao_melanoma"},
        # Caso 7: image_id duplicado (mesmo image_id de HAM_0000001 reaproveitado)
        {"lesion_id": "HAM_0000007", "image_id": "ISIC_0000001", "dx": "nv", "binary_target": "nao_melanoma"},
    ]

    s2_cases_by_image_id = {
        "ISIC_0000001": {"status": "ok", "mask_path": "/mac/path/x1.png", "mask_empty": False},
        "ISIC_0000002": {"status": "ok", "mask_path": "/mac/path/x2.png", "mask_empty": True},
        "ISIC_0000003": {"status": "ok", "mask_path": "/mac/path/x3.png", "mask_empty": False},
        "ISIC_0000004": {"status": "ok", "mask_path": "/mac/path/x4.png", "mask_empty": False},
        "ISIC_0000005": {"status": "falha", "mask_path": "", "mask_empty": None},
        "ISIC_0000006": {"status": "ok", "mask_path": "/mac/path/x6.png", "mask_empty": False},
    }

    return {
        "images_dir": images_dir,
        "ref_masks_dir": ref_masks_dir,
        "s2_masks_dir": s2_masks_dir,
        "manifest_rows": manifest_rows,
        "s2_cases_by_image_id": s2_cases_by_image_id,
    }


def test_pairing_report_mask_empty_case_blocks_s2_only(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case2 = next(p for p in pairings if p.image_id == "ISIC_0000002")
    assert case2.original_image_available is True
    assert case2.reference_mask_available is True
    assert case2.s2_mask_available_locally is False
    assert case2.reason_s2_unavailable == REASON_MASK_EMPTY


def test_pairing_report_s2_ok_in_json_but_missing_locally(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case3 = next(p for p in pairings if p.image_id == "ISIC_0000003")
    assert case3.s2_mask_available_locally is False
    assert case3.reason_s2_unavailable == REASON_FILE_MISSING_LOCALLY
    # nao deve ser confundido com mascara_predita_vazia
    assert case3.reason_s2_unavailable != REASON_MASK_EMPTY


def test_pairing_report_s2_shape_mismatch(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case4 = next(p for p in pairings if p.image_id == "ISIC_0000004")
    assert case4.reason_s2_unavailable == REASON_SHAPE_MISMATCH


def test_pairing_report_s2_failure_status(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case5 = next(p for p in pairings if p.image_id == "ISIC_0000005")
    assert case5.reason_s2_unavailable == REASON_FAILURE
    # S0/S1 do caso 5 devem permanecer disponiveis -- falha de S2 nao
    # contamina as outras condicoes.
    assert case5.original_image_available is True
    assert case5.reference_mask_available is True


def test_pairing_report_reference_mask_missing_marks_s1_unavailable(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case6 = next(p for p in pairings if p.image_id == "ISIC_0000006")
    assert case6.reference_mask_available is False
    assert case6.s2_mask_available_locally is True  # S2 nao e afetado pela ausencia de S1


def test_pairing_report_reference_mask_shape_mismatch(synthetic_project):
    # sobrescreve a mascara de referencia do caso 1 com uma de shape errada
    wrong_shape_mask = _make_mask(25, 21, (1, 1, 6, 6))
    _save_mask_png(wrong_shape_mask, synthetic_project["ref_masks_dir"] / "ISIC_0000001_segmentation.png")

    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case1 = next(p for p in pairings if p.image_id == "ISIC_0000001" and p.lesion_id == "HAM_0000001")
    assert case1.reference_mask_available is False
    assert case1.reason_s1_unavailable == REASON_REF_SHAPE_MISMATCH
    # S2 do mesmo caso nao deve ser afetado pelo problema em S1
    assert case1.s2_mask_available_locally is True


def test_pairing_report_detects_duplicate_image_id_without_dropping_rows(synthetic_project):
    pairings, extras = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    assert extras["duplicate_image_ids"] == {"ISIC_0000001": 2}
    # nenhuma linha foi descartada: 7 linhas no manifesto -> 7 pairings
    assert len(pairings) == len(synthetic_project["manifest_rows"]) == 7


def test_pairing_report_detects_manifest_image_id_without_s2_entry(synthetic_project):
    rows = synthetic_project["manifest_rows"] + [
        {"lesion_id": "HAM_0000099", "image_id": "ISIC_0000099", "dx": "nv", "binary_target": "nao_melanoma"}
    ]
    _, extras = build_pairing_report(
        rows,
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    assert "ISIC_0000099" in extras["image_ids_without_s2_entry"]


def test_pairing_report_detects_extra_s2_entry_without_manifest_row(synthetic_project):
    s2_cases = dict(synthetic_project["s2_cases_by_image_id"])
    s2_cases["ISIC_9999999"] = {"status": "ok", "mask_path": "/mac/extra.png", "mask_empty": False}
    _, extras = build_pairing_report(
        synthetic_project["manifest_rows"],
        s2_cases,
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    assert "ISIC_9999999" in extras["s2_entries_without_manifest_row"]


# ---------------------------------------------------------------------------
# Stub de payload dublê -- nenhum rotulo de referencia deve aparecer.
# ---------------------------------------------------------------------------
def test_payload_stub_contains_no_reference_labels(tmp_path):
    fake_image_path = tmp_path / "ISIC_0000001_s2.png"
    fake_image_path.write_bytes(b"fake")
    payload = build_vlm_payload_stub(fake_image_path, condition="S2")

    assert verify_payload_has_no_labels(payload) is True
    assert "dx" not in payload
    assert "binary_target" not in payload
    assert set(payload.keys()) == {"image_path", "condition", "instruction"}


def test_payload_stub_signature_cannot_accept_labels():
    import inspect

    sig = inspect.signature(build_vlm_payload_stub)
    assert "dx" not in sig.parameters
    assert "binary_target" not in sig.parameters


def test_verify_payload_has_no_labels_catches_leaked_key():
    leaked_payload = {"image_path": "x.png", "condition": "S1", "dx": "mel"}
    assert verify_payload_has_no_labels(leaked_payload) is False


def test_verify_payload_has_no_labels_catches_leaked_value_substring():
    leaked_payload = {"image_path": "x.png", "condition": "S1", "instruction": "this case is binary_target=melanoma"}
    assert verify_payload_has_no_labels(leaked_payload) is False


# ---------------------------------------------------------------------------
# draw_contour: preservacao explicita de resolucao/enquadramento.
# ---------------------------------------------------------------------------
def test_draw_contour_preserves_resolution_exactly():
    image = _make_image(37, 23)
    mask = _make_mask(37, 23, (5, 4, 20, 15))
    result = draw_contour(image, mask)
    assert result.size == image.size


# ---------------------------------------------------------------------------
# Regressao: contorno fino numa imagem grande nao pode ser um falso negativo
# em images_are_clearly_different (bug real encontrado em execucao no Mac
# com imagens HAM10000 450x600 -- a media de diferenca sobre a imagem
# INTEIRA ficava bem abaixo do limiar de tolerancia a ruido de JPEG mesmo
# com o contorno claramente desenhado, porque o contorno altera uma fracao
# minuscula do total de pixels).
# ---------------------------------------------------------------------------
def test_images_are_clearly_different_detects_thin_contour_on_large_image():
    w, h = 450, 600
    image = _make_image(w, h, color=(120, 90, 80))
    # mascara pequena em relacao ao total (lesao tipica), produz um
    # contorno de poucos pixels de espessura -- a mudanca de conteudo e
    # real e visivel, mas afeta uma fracao minuscula da imagem inteira.
    mask = _make_mask(w, h, (200, 250, 260, 310))

    contoured = draw_contour(image, mask)

    # a media de diferenca sobre a imagem inteira cai sob o limiar de
    # tolerancia a JPEG (prova de que a negacao de "effectively_equal"
    # seria um falso negativo aqui).
    assert images_are_effectively_equal(image, contoured) is True
    # mas a deteccao por fracao de pixels localmente diferentes confirma
    # corretamente que o contorno esta presente.
    assert images_are_clearly_different(image, contoured) is True


def test_images_are_clearly_different_false_for_truly_identical_images():
    image = _make_image(40, 30)
    identical_copy = image.copy()
    assert images_are_clearly_different(image, identical_copy) is False


# ---------------------------------------------------------------------------
# Payload REAL (_image_message/convert_to_base64) -- decodifica de volta
# a pixels e confirma que carrega a imagem certa, sem rotulo no texto.
# ---------------------------------------------------------------------------
def test_build_real_vlm_payload_roundtrip_matches_source_image():
    image = _make_image(24, 18, color=(10, 200, 30))
    payload = build_real_vlm_payload(image, condition="S0")
    decoded = decode_payload_image(payload)
    assert images_are_effectively_equal(decoded, image)


def test_build_real_vlm_payload_contour_image_differs_from_original():
    image = _make_image(24, 18)
    mask = _make_mask(24, 18, (2, 2, 15, 12))
    contoured = draw_contour(image, mask)

    payload_original = build_real_vlm_payload(image, condition="S0")
    payload_contoured = build_real_vlm_payload(contoured, condition="S1")

    decoded_original = decode_payload_image(payload_original)
    decoded_contoured = decode_payload_image(payload_contoured)
    assert images_are_clearly_different(decoded_original, decoded_contoured)


def test_payload_text_has_no_labels_passes_for_clean_instruction():
    image = _make_image(10, 10)
    payload = build_real_vlm_payload(image, condition="S2", task_instruction="descreva a lesao")
    assert payload_text_has_no_labels(payload) is True


def test_payload_text_has_no_labels_catches_leaked_diagnosis_text():
    image = _make_image(10, 10)
    payload = build_real_vlm_payload(
        image, condition="S2", task_instruction="este caso tem binary_target=melanoma"
    )
    assert payload_text_has_no_labels(payload) is False


# ---------------------------------------------------------------------------
# require_s2_payload_available: bloqueio explicito, nunca fallback.
# ---------------------------------------------------------------------------
def test_require_s2_payload_available_raises_for_unavailable_case(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case2 = next(p for p in pairings if p.image_id == "ISIC_0000002")  # mascara_predita_vazia
    with pytest.raises(S2PayloadUnavailableError):
        require_s2_payload_available(case2)


def test_require_s2_payload_available_does_not_raise_for_available_case(synthetic_project):
    pairings, _ = build_pairing_report(
        synthetic_project["manifest_rows"],
        synthetic_project["s2_cases_by_image_id"],
        synthetic_project["images_dir"],
        synthetic_project["ref_masks_dir"],
        synthetic_project["s2_masks_dir"],
    )
    case1 = next(p for p in pairings if p.image_id == "ISIC_0000001" and p.lesion_id == "HAM_0000001")
    require_s2_payload_available(case1)  # nao deve levantar
