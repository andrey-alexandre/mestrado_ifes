from __future__ import annotations

import pytest

from src.cnn_baseline.threshold_selection import select_youden_threshold


def test_perfect_separation_picks_a_threshold_between_the_classes():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = select_youden_threshold(labels, scores)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["youden_j"] == 1.0
    assert 0.3 < result["threshold"] <= 0.7


def test_known_handcomputed_example():
    # scores sorted desc with labels: 1,0,1,0,0 -> thresholds 0.9,0.8,0.6,0.4,0.2 (+inf)
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.6, 0.4, 0.2]
    # at threshold=0.6: predicted positive={0.9,0.8,0.6} -> tp=2 (labels 1 at .9,.6), fp=1 (.8)
    # tpr = 2/2 = 1.0, fpr = 1/3 = 0.333 -> J = 0.667
    # at threshold=0.8: predicted positive={0.9,0.8} -> tp=1, fp=1 -> tpr=0.5, fpr=0.333 -> J=0.167
    # at threshold=0.9: predicted positive={0.9} -> tp=1, fp=0 -> tpr=0.5, fpr=0 -> J=0.5
    result = select_youden_threshold(labels, scores)
    assert result["threshold"] == pytest.approx(0.6)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["youden_j"] == pytest.approx(2 / 3)


def test_single_class_raises():
    with pytest.raises(ValueError):
        select_youden_threshold([1, 1, 1], [0.1, 0.5, 0.9])


def test_clean_separation_picks_the_unique_best_threshold():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.4, 0.5]
    result = select_youden_threshold(labels, scores)
    assert result["youden_j"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.4)
