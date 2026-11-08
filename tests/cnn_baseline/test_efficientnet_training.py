import pytest
import torch

from src.cnn_baseline import efficientnet_training as et


def test_select_best_epoch_picks_highest_val_auc():
    history = [
        {"epoch": 1, "val_auc": 0.70},
        {"epoch": 2, "val_auc": 0.85},
        {"epoch": 3, "val_auc": 0.80},
    ]
    result = et.select_best_epoch(history)
    assert result["selected_epoch"] == 2
    assert result["selected_val_auc"] == 0.85
    assert result["rule"] == et.CHECKPOINT_SELECTION_RULE


def test_select_best_epoch_ties_prefer_earliest():
    history = [
        {"epoch": 1, "val_auc": 0.80},
        {"epoch": 2, "val_auc": 0.90},
        {"epoch": 3, "val_auc": 0.90},
    ]
    result = et.select_best_epoch(history)
    assert result["selected_epoch"] == 2


def test_select_best_epoch_raises_on_empty_history():
    with pytest.raises(ValueError):
        et.select_best_epoch([])


def test_binary_target_to_label_mapping_reexported():
    assert et.BINARY_TARGET_TO_LABEL["melanoma"] == 1
    assert et.BINARY_TARGET_TO_LABEL["nao_melanoma"] == 0
    assert et.LABEL_TO_BINARY_TARGET[1] == "melanoma"
    assert et.LABEL_TO_BINARY_TARGET[0] == "nao_melanoma"


def test_build_val_scores_rows_shape_and_content():
    rows = et.build_val_scores_rows(
        lesion_ids=["L1", "L2"],
        bcn_filenames=["f1.jpg", "f2.jpg"],
        true_labels=[1, 0],
        predicted_scores=[0.91, 0.12],
    )
    assert len(rows) == 2
    assert rows[0] == {
        "lesion_id": "L1",
        "bcn_filename": "f1.jpg",
        "binary_target": "melanoma",
        "true_label": 1,
        "predicted_score": 0.91,
    }
    assert rows[1]["binary_target"] == "nao_melanoma"


def test_build_val_scores_rows_rejects_score_out_of_range():
    with pytest.raises(ValueError):
        et.build_val_scores_rows(["L1"], ["f1.jpg"], [1], [1.5])
    with pytest.raises(ValueError):
        et.build_val_scores_rows(["L1"], ["f1.jpg"], [1], [-0.1])


def test_build_val_scores_rows_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        et.build_val_scores_rows(["L1", "L2"], ["f1.jpg"], [1, 0], [0.5, 0.5])


def test_build_val_scores_rows_rejects_unknown_label():
    with pytest.raises(ValueError):
        et.build_val_scores_rows(["L1"], ["f1.jpg"], [2], [0.5])


def test_build_model_returns_single_logit_output():
    model, weights = et.build_model(pretrained=False)
    assert weights is None
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)
