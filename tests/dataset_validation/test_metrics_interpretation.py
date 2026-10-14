import math

import pytest

from src.dataset_validation import interpretation as I
from src.dataset_validation import metrics as M


def test_confusion_and_point_metrics():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    c = M.confusion(y, p)
    assert c == {"tp": 3, "tn": 4, "fp": 2, "fn": 1}
    m = M.point_metrics(c)
    assert m["sensitivity"] == 0.75 and m["specificity"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 4 / 6) / 2)
    assert m["accuracy"] == pytest.approx(0.7)


def test_wilson_known_values():
    lo, hi = M.wilson_ci(15, 20, 0.95)            # valor de referência de livro-texto: [0.531, 0.888]
    assert lo == pytest.approx(0.531, abs=2e-3) and hi == pytest.approx(0.888, abs=2e-3)
    assert M.wilson_ci(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert M.wilson_ci(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(M.wilson_ci(0, 0)[0])


def test_metrics_with_ci_reproduces_thesis_table():
    # Gemini (dissertação): TP=17 FN=8 FP=4 TN=21 -> acc .76, prec .81, sens .68, F1 .74
    y = [1] * 25 + [0] * 25
    p = [1] * 17 + [0] * 8 + [1] * 4 + [0] * 21
    r = M.metrics_with_ci(y, p, n_resamples=500, seed=1)
    assert round(r["accuracy"]["estimate"], 2) == 0.76 and round(r["precision"]["estimate"], 2) == 0.81
    assert round(r["sensitivity"]["estimate"], 2) == 0.68 and round(r["f1"]["estimate"], 2) == 0.74
    assert r["confusion_matrix"] == [[21, 4], [8, 17]]
    assert r["f1"]["ci_low"] <= r["f1"]["estimate"] <= r["f1"]["ci_high"]
    assert r["roc_auc"]["estimate"] is None
    again = M.metrics_with_ci(y, p, n_resamples=500, seed=1)
    assert again["f1"]["ci_low"] == r["f1"]["ci_low"]       # semente fixa


def test_degenerate_metrics_are_nan_not_errors():
    r = M.metrics_with_ci([1, 1], [1, 1], n_resamples=50)
    assert math.isnan(r["specificity"]["estimate"]) and r["sensitivity"]["estimate"] == 1.0


def test_compare_proportions_holm_and_cliffs():
    r = M.compare_proportions(18, 20, 8, 20)
    assert r["difference"] == pytest.approx(0.5) and r["ci_low"] < 0.5 < r["ci_high"]
    assert r["fisher_exact_p_two_sided"] < 0.01
    assert M.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert M.cliffs_delta([3, 4, 5], [1, 2]) == 1.0


def test_interpretation_keys_match_repo_rule():
    from src.evaluation.evaluate import _DIAGNOSTICO_TO_LABEL
    assert set(I.LEGACY_KEY_TO_CATEGORY) == set(_DIAGNOSTICO_TO_LABEL)
    assert I.binarize(I.INDETERMINADA, "indeterminate_as_positive") == _DIAGNOSTICO_TO_LABEL["lesão indeterminada"]
    assert I.binarize(I.SUSPEITA, "indeterminate_as_positive") == _DIAGNOSTICO_TO_LABEL["suspeita de melanoma"]
    assert I.binarize(I.BENIGNA, "indeterminate_as_positive") == _DIAGNOSTICO_TO_LABEL["lesão benigna"]


@pytest.mark.parametrize("text,cat,tier", [
    ("Suspeita de Melanoma", I.SUSPEITA, "legacy_exact"),
    ("Lesão Benigna", I.BENIGNA, "legacy_exact"),
    ("Lesão Indeterminada com necessidade de correlação histopatológica", I.INDETERMINADA, "legacy_exact"),
    ("Lesão cutânea com **alto grau de suspeição para melanoma**", I.SUSPEITA, "extended_keywords"),
    ("Lesão cutânea indeterminada com suspeita elevada", I.INDETERMINADA, "extended_keywords"),
    ("Lesão cutânea benigna", I.BENIGNA, "extended_keywords"),
    ("Sem suspeita de malignidade evidente", I.NAO_AVALIAVEL, "extended_keywords:negated_suspicion"),
    ("", I.NAO_AVALIAVEL, "empty"),
    ("Não foi possível analisar", I.NAO_AVALIAVEL, "unrecognized"),
])
def test_classify_diagnosis(text, cat, tier):
    assert I.classify_diagnosis(text) == (cat, tier)


def test_binarize_policies_and_non_evaluable_kept():
    assert I.binarize(I.NAO_AVALIAVEL, "indeterminate_as_positive") is None
    assert I.binarize(I.INDETERMINADA, "indeterminate_as_negative") == 0
    assert I.binarize(I.SUSPEITA, "indeterminate_as_negative") == 1
    with pytest.raises(ValueError):
        I.binarize(I.SUSPEITA, "x")


def test_agent_features_recompute_declared_rules():
    pos = {k: "0. não" for k in I.MENZIES_POS}
    pos[I.MENZIES_POS[0]] = "1. sim"
    spcl = {k: "Não." for k in I.SPCL_MAJOR + I.SPCL_MINOR}
    spcl[I.SPCL_MAJOR[0]] = "Sim, presente"
    ev = {
        "diagnosis_abcd": {"assimetria": "1 - leve", "bordas": "5/8 - parciais", "cor": "3", "estruturas": "3"},
        "diagnosis_menzies": {"positivas": pos, "negativas": {"simetria": "0. assimétrica", "cor_unica": "0. várias"}},
        "diagnosis_spcl": spcl,
    }
    f = I.agent_features(ev)
    assert f["abcd_total"] == pytest.approx(1.3 + 0.5 + 1.5 + 1.5)         # 4.8, igual ao CSV legado
    assert f["abcd_class"] == "suspeita" and f["abcd_bordas_imprecise"] is True
    assert f["menzies_decision"] == "melanoma"
    assert f["spcl_score"] == 2 and f["spcl_decision"] == "benigna"
    assert f["n_methods_high_risk"] == 2
    assert I.agent_features({"diagnosis_abcd": "texto livre"}) == {"n_methods_high_risk": None}
