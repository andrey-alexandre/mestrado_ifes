from src.data_audit import overlap


def test_compare_by_hash_finds_duplicate():
    manifest_a = [
        {"path": "a/one.jpg", "size": 3, "sha256": "aaa"},
        {"path": "a/two.jpg", "size": 4, "sha256": "bbb"},
    ]
    manifest_b = [
        {"path": "b/copy_of_one.jpg", "size": 3, "sha256": "aaa"},
        {"path": "b/three.jpg", "size": 5, "sha256": "ccc"},
    ]

    result = overlap.compare_by_hash(manifest_a, manifest_b)

    assert result["duplicate_count"] == 1
    assert result["duplicates"]["aaa"] == {
        "paths_a": ["a/one.jpg"],
        "paths_b": ["b/copy_of_one.jpg"],
    }
    assert "bbb" not in result["duplicates"]
    assert "ccc" not in result["duplicates"]


def test_compare_by_hash_no_duplicate():
    manifest_a = [{"path": "a/one.jpg", "size": 3, "sha256": "aaa"}]
    manifest_b = [{"path": "b/two.jpg", "size": 4, "sha256": "bbb"}]

    result = overlap.compare_by_hash(manifest_a, manifest_b)

    assert result["duplicate_count"] == 0
    assert result["duplicates"] == {}


def test_compare_by_hash_multiple_paths_sharing_one_hash():
    manifest_a = [
        {"path": "a/one.jpg", "size": 3, "sha256": "aaa"},
        {"path": "a/one_copy.jpg", "size": 3, "sha256": "aaa"},
    ]
    manifest_b = [{"path": "b/one.jpg", "size": 3, "sha256": "aaa"}]

    result = overlap.compare_by_hash(manifest_a, manifest_b)

    assert result["duplicates"]["aaa"]["paths_a"] == ["a/one.jpg", "a/one_copy.jpg"]
    assert result["duplicates"]["aaa"]["paths_b"] == ["b/one.jpg"]


def test_compare_by_id_wraps_pair_by_id():
    result = overlap.compare_by_id(["ISIC_0000000", "ISIC_0000001"], ["ISIC_0000001", "ISIC_0000002"])

    assert result["matched"] == ["ISIC_0000001"]
    assert result["only_in_a"] == ["ISIC_0000000"]
    assert result["only_in_b"] == ["ISIC_0000002"]
