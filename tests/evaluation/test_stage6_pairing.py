from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_pairing import (  # noqa: E402
    build_manifest_index,
    derive_image_id_from_path,
    pair_results_by_lesion,
)

MANIFEST_ROWS = [
    {"lesion_id": "HAM_0000001", "image_id": "ISIC_0000001", "dx": "nv", "binary_target": "nao_melanoma"},
    {"lesion_id": "HAM_0000002", "image_id": "ISIC_0000002", "dx": "mel", "binary_target": "melanoma"},
    {"lesion_id": "HAM_0000003", "image_id": "ISIC_0000003", "dx": "nv", "binary_target": "nao_melanoma"},
]


def test_derive_image_id_from_path():
    assert derive_image_id_from_path("/x/y/ISIC_0024579.jpg") == "ISIC_0024579"
    assert derive_image_id_from_path("ISIC_0024579.jpg") == "ISIC_0024579"


def test_build_manifest_index_basic():
    idx = build_manifest_index(MANIFEST_ROWS)
    assert idx.lesion_ids == ["HAM_0000001", "HAM_0000002", "HAM_0000003"]
    assert idx.image_id_to_lesion_id["ISIC_0000002"] == "HAM_0000002"
    assert idx.lesion_id_to_binary_target["HAM_0000002"] == "melanoma"
    assert sorted(idx.lesion_ids_by_category["nv"]) == ["HAM_0000001", "HAM_0000003"]
    assert idx.lesion_ids_by_category["mel"] == ["HAM_0000002"]


def test_pairing_all_present_no_issues():
    idx = build_manifest_index(MANIFEST_ROWS)
    results = {
        "/a/ISIC_0000001.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
        "/a/ISIC_0000002.jpg": {"validation": {"diagnostico": "Suspeita de Melanoma"}},
        "/a/ISIC_0000003.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
    }
    report = pair_results_by_lesion(idx, results)
    assert report.missing_lesion_ids == []
    assert report.duplicate_lesion_ids == {}
    assert report.orphan_paths == []
    assert set(report.lesion_id_to_event.keys()) == {"HAM_0000001", "HAM_0000002", "HAM_0000003"}


def test_pairing_flags_missing_lesion():
    idx = build_manifest_index(MANIFEST_ROWS)
    results = {"/a/ISIC_0000001.jpg": {"validation": {"diagnostico": "Lesão Benigna"}}}
    report = pair_results_by_lesion(idx, results)
    assert report.missing_lesion_ids == ["HAM_0000002", "HAM_0000003"]


def test_pairing_flags_duplicate_lesion():
    idx = build_manifest_index(MANIFEST_ROWS)
    results = {
        "/a/ISIC_0000001.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
        "/b/ISIC_0000001.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
        "/a/ISIC_0000002.jpg": {"validation": {"diagnostico": "Suspeita de Melanoma"}},
        "/a/ISIC_0000003.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
    }
    report = pair_results_by_lesion(idx, results)
    assert "HAM_0000001" in report.duplicate_lesion_ids
    assert sorted(report.duplicate_lesion_ids["HAM_0000001"]) == [
        "/a/ISIC_0000001.jpg", "/b/ISIC_0000001.jpg",
    ]
    # Mesmo duplicado, ainda expoe um evento usavel (nao descartado):
    assert "HAM_0000001" in report.lesion_id_to_event


def test_pairing_flags_orphan_path():
    idx = build_manifest_index(MANIFEST_ROWS)
    results = {
        "/a/ISIC_0000001.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
        "/a/ISIC_0000002.jpg": {"validation": {"diagnostico": "Suspeita de Melanoma"}},
        "/a/ISIC_0000003.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
        "/a/ISIC_9999999.jpg": {"validation": {"diagnostico": "Lesão Benigna"}},
    }
    report = pair_results_by_lesion(idx, results)
    assert report.orphan_paths == ["/a/ISIC_9999999.jpg"]
    assert report.missing_lesion_ids == []
