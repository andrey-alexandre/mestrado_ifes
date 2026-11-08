import csv
import io
import zipfile

from PIL import Image

from src.cnn_baseline import bcn20000_dataset as ds


def _make_fake_zip(tmp_path, image_names):
    zip_path = tmp_path / "fake_bcn.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in image_names:
            img = Image.new("RGB", (32, 32), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            zf.writestr(f"{ds.ZIP_ENTRY_PREFIX}{name}", buf.getvalue())
    return zip_path


def _manifest_rows():
    return [
        {
            "lesion_id": "L1", "bcn_filename": "a.jpg", "source_partition": "train",
            "diagnosis": "MEL", "binary_target": "melanoma", "split": "train",
        },
        {
            "lesion_id": "L2", "bcn_filename": "b.jpg", "source_partition": "train",
            "diagnosis": "NV", "binary_target": "nao_melanoma", "split": "train",
        },
        {
            "lesion_id": "L3", "bcn_filename": "c.jpg", "source_partition": "train",
            "diagnosis": "NV", "binary_target": "nao_melanoma", "split": "val",
        },
    ]


def test_filter_rows_by_split():
    rows = _manifest_rows()
    train_rows = ds.filter_rows_by_split(rows, "train")
    val_rows = ds.filter_rows_by_split(rows, "val")
    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert val_rows[0]["lesion_id"] == "L3"


def test_read_split_manifest_round_trips(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    rows = _manifest_rows()
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    read_back = ds.read_split_manifest(manifest_path)
    assert read_back == rows


def test_dataset_loads_real_image_from_zip_and_maps_label(tmp_path):
    rows = _manifest_rows()[:2]
    zip_path = _make_fake_zip(tmp_path, ["a.jpg", "b.jpg"])
    dataset = ds.BCN20000ZipDataset(rows, zip_path, transform=None, augment=None)

    assert len(dataset) == 2
    image, label, lesion_id, filename = dataset[0]
    assert image.size == (32, 32)
    assert label == 1  # melanoma
    assert lesion_id == "L1"
    assert filename == "a.jpg"

    _image2, label2, _lid2, _fn2 = dataset[1]
    assert label2 == 0  # nao_melanoma


def test_dataset_applies_transform_when_given(tmp_path):
    rows = _manifest_rows()[:1]
    zip_path = _make_fake_zip(tmp_path, ["a.jpg"])

    def fake_transform(img):
        return "TRANSFORMED"

    dataset = ds.BCN20000ZipDataset(rows, zip_path, transform=fake_transform, augment=None)
    image, *_ = dataset[0]
    assert image == "TRANSFORMED"


def test_dataset_applies_augment_only_when_given(tmp_path):
    calls = []

    def fake_augment(img):
        calls.append(1)
        return img

    rows = _manifest_rows()[:1]
    zip_path = _make_fake_zip(tmp_path, ["a.jpg"])
    dataset = ds.BCN20000ZipDataset(rows, zip_path, transform=None, augment=fake_augment)
    dataset[0]
    assert len(calls) == 1


def test_dataset_does_not_open_zip_at_construction(tmp_path):
    rows = _manifest_rows()[:1]
    zip_path = _make_fake_zip(tmp_path, ["a.jpg"])
    dataset = ds.BCN20000ZipDataset(rows, zip_path)
    assert dataset._zf is None
    dataset[0]
    assert dataset._zf is not None


def test_binary_target_to_label_mapping_values():
    assert ds.BINARY_TARGET_TO_LABEL["melanoma"] == 1
    assert ds.BINARY_TARGET_TO_LABEL["nao_melanoma"] == 0
    assert ds.LABEL_TO_BINARY_TARGET[1] == "melanoma"
    assert ds.LABEL_TO_BINARY_TARGET[0] == "nao_melanoma"
