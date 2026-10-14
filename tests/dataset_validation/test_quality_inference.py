import types
from pathlib import Path

import numpy as np
import pytest

from src.dataset_validation import inference as INF
from src.dataset_validation import quality as Q
from src.dataset_validation.io_utils import read_json

QCFG = {"lesion_interior_erosion_px": 3, "boundary_band_px": 3, "skin_ring_dilation_px": 5,
        "hair_blackhat_size": 5, "hair_threshold": 15, "reference_width_px": 64, "corner_fraction": 0.1}


def _scene(blur=False, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((96, 128, 3), 200, np.float32)
    mask = np.zeros((96, 128), bool)
    mask[30:70, 40:90] = True
    img[mask] = (90, 50, 40)
    if blur:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=(2.5, 2.5, 0))
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def test_indicators_cover_all_names_and_geometry():
    img, mask = _scene()
    r = Q.compute_indicators(img, mask, QCFG, file_bytes=1000)
    assert set(Q.INDICATORS) <= set(r)
    assert r["width_px"] == 128 and r["height_px"] == 96
    assert r["lesion_area_ratio"] == pytest.approx(mask.mean())
    assert r["contrast_lesion_skin_dE"] > 20 and r["lesion_touches_border"] == 0.0


def test_sharpness_drops_with_blur_and_noise_estimate_rises_with_noise():
    sharp, mask = _scene()
    blur, _ = _scene(blur=True)
    noisy, _ = _scene(noise=6.0)
    a, b, c = (Q.compute_indicators(x, mask, QCFG) for x in (sharp, blur, noisy))
    assert a["lap_var_global"] > b["lap_var_global"]
    assert a["edge_strength_boundary_ref"] > b["edge_strength_boundary_ref"]
    assert c["noise_sigma_skin"] > a["noise_sigma_skin"] + 3


def test_empty_mask_yields_nan_not_error():
    img, _ = _scene()
    r = Q.compute_indicators(img, np.zeros(img.shape[:2], bool), QCFG)
    assert np.isnan(r["contrast_lesion_skin_dE"]) and r["lesion_area_px"] == 0


def test_build_entries_deterministic_and_complete():
    m = [{"sample_id": f"a:{i}", "image_path": f"/i{i}", "mask_path": f"/m{i}", "binary_label": i % 2}
         for i in range(10)]
    e1, e2 = INF.build_entries(m, 42), INF.build_entries(list(reversed(m)), 42)
    assert e1 == e2 and len(e1) == 10 and e1 != INF.build_entries(m, 7)
    assert all(e[1] == 0.0 and e[3].startswith("/m") for e in e1)


def test_estimate_and_credential_gate(cfg, monkeypatch, tmp_path):
    for v in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    est = INF.estimate_cost(cfg, 80)
    assert est["per_model"]["gpt-5.4-nano"]["llm_calls"] == 320 and est["total_llm_calls"] == 640
    assert est["per_model"]["gpt-5.4-nano"]["credential_present"] is False
    with pytest.raises(INF.InferenceError):
        INF.run_model(cfg, [], "gpt-5.4-nano", tmp_path / "x")


def test_run_model_resumes_without_repeating_calls(cfg, tmp_path):
    manifest = [{"sample_id": f"ph2:{i}", "dataset": "ph2", "image_id": str(i), "image_path": f"/i{i}",
                 "mask_path": f"/m{i}", "binary_label": i % 2} for i in range(4)]
    calls = []

    def runner(entries):
        path = entries[0][0]
        calls.append(path)
        return {path: {"image_path": path, "real_value": entries[0][2],
                       "validation": {"diagnostico": "Lesão Benigna"}}}

    out = tmp_path / "inf"
    p = INF.run_model(cfg, manifest, "llava:13b", out, pipeline_runner=runner)
    assert len(calls) == 4 and len(read_json(p)) == 4
    INF.run_model(cfg, manifest, "llava:13b", out, pipeline_runner=runner)
    assert len(calls) == 4                                     # nenhuma chamada nova
    assert read_json(Path(p).with_suffix(".meta.json"))["segmentation_mode"] == "precomputed"


def test_inject_credentials_uses_env_without_touching_other_kwargs():
    seen = {}

    class Fake:
        def __init__(self, **k):
            seen.update(k)

    mod = types.SimpleNamespace(ChatOpenAI=Fake, ChatGoogleGenerativeAI=Fake)
    INF.inject_credentials(mod, env={"OPENAI_API_KEY": "k1", "GOOGLE_API_KEY": "k2"})
    mod.ChatOpenAI(model="m", temperature=0, openai_api_key="OPENAI_API_KEY")
    assert seen == {"model": "m", "temperature": 0, "openai_api_key": "k1"}
    mod2 = types.SimpleNamespace(ChatOpenAI=Fake, ChatGoogleGenerativeAI=Fake)
    INF.inject_credentials(mod2, env={})
    with pytest.raises(INF.InferenceError):
        mod2.ChatOpenAI(model="m")


def test_run_model_retries_then_records_failure_and_tags_sample_id(cfg, tmp_path):
    manifest = [{"sample_id": f"ph2:{i}", "dataset": "ph2", "image_id": str(i), "image_path": f"/i{i}",
                 "mask_path": f"/m{i}", "binary_label": i % 2} for i in range(3)]
    attempts = {}

    def runner(entries):
        path = entries[0][0]
        attempts[path] = attempts.get(path, 0) + 1
        if path == "/i1" or (path == "/i2" and attempts[path] < 2):
            raise ConnectionError("timeout")
        return {path: {"image_path": path, "real_value": entries[0][2], "validation": {"diagnostico": "x"}}}

    out = tmp_path / "inf"
    p = INF.run_model(cfg, manifest, "gpt-5.4-nano", out, pipeline_runner=runner, attempts=3,
                      sleep=lambda s: None)
    res, meta = read_json(p), read_json(Path(p).with_suffix(".meta.json"))
    assert attempts["/i1"] == 3 and attempts["/i2"] == 2          # repetiu até o limite / recuperou
    assert list(meta["failures"]) == ["ph2:1"] and meta["n_images_completed"] == 2
    assert {ev["_meta"]["sample_id"] for ev in res.values()} == {"ph2:0", "ph2:2"}
    # retomada: só a imagem faltante é chamada de novo
    before = dict(attempts)
    INF.run_model(cfg, manifest, "gpt-5.4-nano", out, pipeline_runner=lambda e: {e[0][0]: {"real_value": 0}},
                  attempts=1, sleep=lambda s: None)
    assert attempts == before


def test_interpret_results_matches_by_sample_id_across_machines():
    from src.dataset_validation import analysis as A
    manifest = [{"sample_id": "ph2:IMD001", "dataset": "ph2", "class_label": "melanoma", "binary_label": 1,
                 "image_id": "IMD001", "acquisition_source": "x", "dx_type": "", "ph2_nevus_subtype": "",
                 "image_path": "C:/maquina_a/IMD001.bmp"}]
    results = {"/data/maquina_b/IMD001.bmp": {"validation": {"diagnostico": "Suspeita de Melanoma"},
                                              "_meta": {"sample_id": "ph2:IMD001"}}}
    row = A.interpret_results(manifest, results)[0]
    assert row["status"] == "ok" and row["category"] == "SUSPEITA"
