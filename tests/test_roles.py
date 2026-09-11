"""RoleTagger の立場推定。

実証する主張: 「向きの誤り検出」。〈誰の行為か〉を読み違えると向きの判定が
成り立たない。代表的な手掛かりが効いていることを固定する。
"""

from __future__ import annotations

import pytest

from deference.generate import default_context
from deference.roles import RoleTagger
from deference.types import Audience, KeigoClass, Party, Person


@pytest.fixture(scope="module")
def tagger():
    return RoleTagger()


@pytest.fixture(scope="module")
def ctx():
    base = default_context(Audience.EXTERNAL)
    return base.with_persons(
        [
            Person("佐藤", Party.SELF_GROUP, "社長", "弊社"),
            Person("田中", Party.ADDRESSEE, "様", "貴社"),
            Person("鈴木", Party.ADDRESSEE_GROUP, "様", "貴社"),
        ]
    )


@pytest.mark.parametrize(
    "text,expected_actor,expected_class",
    [
        ("弊社の田中がご説明いたします。", Party.SELF_GROUP, None),
        ("田中様がおっしゃいました。", Party.ADDRESSEE, KeigoClass.SONKEIGO),
        ("明日、貴社に伺います。", Party.SELF, KeigoClass.KENJOUGO_1),
        ("担当者に伺ってください。", Party.ADDRESSEE, KeigoClass.KENJOUGO_1),
        ("資料をお送りいただき、ありがとうございました。", Party.SELF, KeigoClass.KENJOUGO_1),
        ("弊社の佐藤社長がおっしゃっておりました。", Party.SELF_GROUP, KeigoClass.SONKEIGO),
        ("私が資料をお持ちします。", Party.SELF, KeigoClass.KENJOUGO_1),
        ("鈴木様がご覧になりました。", Party.ADDRESSEE_GROUP, KeigoClass.SONKEIGO),
    ],
)
def test_actor_and_class(tagger, ctx, text, expected_actor, expected_class):
    roles = tagger.tag(text, ctx)
    assert roles, f"述語が同定できませんでした: {text}"
    r = roles[0]
    assert r.actor is expected_actor, (
        f"{text}: actor={r.actor.value} (期待 {expected_actor.value}) "
        f"evidence={r.evidence}"
    )
    if expected_class is not None:
        assert r.keigo_class is expected_class, (
            f"{text}: class={r.keigo_class.value} (期待 {expected_class.value})"
        )


def test_kudasaru_makes_actor_other_side(tagger, ctx):
    r = tagger.tag("資料をお送りくださいました。", ctx)[0]
    assert r.actor.is_other_side, r.evidence


def test_itadaku_makes_actor_self_side(tagger, ctx):
    r = tagger.tag("資料をお送りいただきました。", ctx)[0]
    assert r.actor.is_self_side, r.evidence


def test_low_confidence_when_subject_is_absent(tagger, ctx):
    """主語が無い文では確信度を下げること（過剰指摘を避けるため）。"""
    roles = tagger.tag("本日、ご連絡いたします。", ctx)
    assert roles
    assert roles[0].meta.get("actor_confidence", 1.0) <= 0.8


def test_sentence_split_offsets(tagger):
    text = "一つ目です。二つ目です。\n三つ目です。"
    spans = tagger.split_sentences(text)
    assert len(spans) == 3
    for s in spans:
        assert text[s.start : s.end].strip()
