import hashlib
import zipfile

from src.data_audit import content_hash


def test_hash_isic2018_task1_images_hashes_only_jpg_entries(tmp_path):
    images_zip = tmp_path / "images.zip"
    with zipfile.ZipFile(images_zip, "w") as zf:
        zf.writestr("ISIC_0000000.jpg", b"AAA")
        zf.writestr("ISIC_0000001.jpg", b"BBBB")
        zf.writestr("LICENSE.txt", b"not an image")

    result = content_hash.hash_isic2018_task1_images(images_zip)

    assert result["expected"] == 2
    assert result["processed"] == 2
    assert result["missing"] == 0
    assert result["unreadable"] == []
    by_path = {m["path"]: m for m in result["manifest"]}
    assert by_path["ISIC_0000000.jpg"]["sha256"] == hashlib.sha256(b"AAA").hexdigest()
    assert "LICENSE.txt" not in by_path


def test_hash_bcn20000_images_materializes_and_hashes_each_split(tmp_path):
    train_content = _build_inner_zip({"a.jpg": b"A1", "b.jpg": b"B22"})
    test_content = _build_inner_zip({"c.jpg": b"C333"})

    outer_zip = tmp_path / "BCN20000.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.writestr("BCN_20k_train.zip", train_content)
        zf.writestr("BCN_20k_test.zip", test_content)

    extracted_dir = tmp_path / "_extracted"

    result = content_hash.hash_bcn20000_images(
        outer_zip=outer_zip,
        train_zip_entry="BCN_20k_train.zip",
        test_zip_entry="BCN_20k_test.zip",
        extracted_dir=extracted_dir,
        train_csv_row_count=2,
        test_csv_row_count=1,
    )

    assert (extracted_dir / "BCN_20k_train.zip").read_bytes() == train_content
    assert (extracted_dir / "BCN_20k_test.zip").read_bytes() == test_content

    train = result["train"]
    assert train["expected"] == 2
    assert train["processed"] == 2
    assert train["missing"] == 0
    assert train["csv_row_count"] == 2
    assert train["csv_vs_zip_count_matches"] is True
    assert train["materialization"]["materialized"] is True

    test = result["test"]
    assert test["expected"] == 1
    assert test["processed"] == 1
    assert test["missing"] == 0
    assert test["csv_row_count"] == 1
    assert test["csv_vs_zip_count_matches"] is True


def test_hash_bcn20000_images_registers_csv_vs_zip_count_divergence(tmp_path):
    train_content = _build_inner_zip({"a.jpg": b"A1"})
    outer_zip = tmp_path / "BCN20000.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.writestr("BCN_20k_train.zip", train_content)
        zf.writestr("BCN_20k_test.zip", _build_inner_zip({}))

    result = content_hash.hash_bcn20000_images(
        outer_zip=outer_zip,
        train_zip_entry="BCN_20k_train.zip",
        test_zip_entry="BCN_20k_test.zip",
        extracted_dir=tmp_path / "_extracted",
        train_csv_row_count=5,  # csv claims 5 rows, zip only lists 1 .jpg
        test_csv_row_count=0,
    )

    train = result["train"]
    assert train["expected"] == 1
    assert train["csv_row_count"] == 5
    assert train["csv_vs_zip_count_matches"] is False
    assert train["missing"] == 4


def _build_inner_zip(entries: dict[str, bytes]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()
