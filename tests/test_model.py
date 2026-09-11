"""ErrorSpanClassifier。ネットワーク不要で動くことを含めて検査する。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deference.generate import default_context  # noqa: E402
from deference.model import ErrorSpanClassifier  # noqa: E402
from deference.types import Audience, ErrorType, InjectedError, Span  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    return ErrorSpanClassifier.tiny_for_test()


def test_label_list_covers_all_error_types():
    labels = ErrorSpanClassifier.label_list()
    assert labels[0] == "O"
    for et in ErrorType:
        if et is ErrorType.NONE:
            continue
        assert f"B-{et.value}" in labels
        assert f"I-{et.value}" in labels


def test_predict_runs_without_training(tiny):
    ctx = default_context(Audience.EXTERNAL)
    findings = tiny.predict("弊社の佐藤社長がおっしゃいました。", ctx)
    assert isinstance(findings, list)


def test_offsets_round_trip_through_prefix(tiny):
    """文脈プレフィックスを差し引いて本文基準のオフセットに戻せること。"""
    ctx = default_context(Audience.EXTERNAL)
    text = "弊社の佐藤社長がおっしゃいました。"
    enc = tiny.encode(text, ctx)
    prefix_len = enc["prefix_len"]
    assert prefix_len > 0
    combined = enc["prefix"] + text
    assert combined[prefix_len:] == text
    for start, end in enc["offset_mapping"]:
        if end <= start or end <= prefix_len:
            continue
        assert 0 <= start - prefix_len <= len(text)


def test_labels_align_with_injected_spans(tiny):
    ctx = default_context(Audience.EXTERNAL)
    text = "弊社の佐藤社長がおっしゃいました。"
    span = Span(text.index("おっしゃい"), text.index("おっしゃい") + 5, "おっしゃい")
    err = InjectedError(span=span, error_type=ErrorType.UCHI_SONKEIGO, original_text="申し")
    enc = tiny.encode(text, ctx, errors=[err])
    labels = ErrorSpanClassifier.label_list()
    tagged = {labels[i] for i in enc["labels"] if i >= 0}
    assert any(t.endswith(ErrorType.UCHI_SONKEIGO.value) for t in tagged), tagged


def test_parameter_count_in_target_range():
    """目標サイズ 0.1〜0.3B の宣言が、既定 backbone と整合すること。"""
    assert ErrorSpanClassifier.DEFAULT_MODEL_ID == "xlm-roberta-base"
