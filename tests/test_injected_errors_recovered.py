"""注入した誤りの位置と種別が復元できること。

実証する主張: 「向きの誤り検出」。注入台帳（位置・種別）と検出結果を突き合わせ、
種別ごとの復元率を出す。特に文脈が必要な種別で拾えているかが主張の中心である。
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from deference.types import ErrorType, Verdict


def _reportable(result):
    return [f for f in result.findings if f.verdict is Verdict.NORM_DIVERGENCE]


def test_injected_spans_and_types_are_recovered(corpus, injector, deference):
    total = defaultdict(int)
    located = defaultdict(int)
    typed = defaultdict(int)

    for sentence in corpus:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None:
                continue
            gold = sample.errors[0]
            total[error_type] += 1
            found = _reportable(deference.check(sample.text, sample.context))
            overlapping = [f for f in found if f.span.overlaps(gold.span)]
            if overlapping:
                located[error_type] += 1
            if any(f.error_type is error_type for f in overlapping):
                typed[error_type] += 1

    assert total, "注入できた誤りが1件もありません"
    lines = []
    for et in sorted(total, key=lambda e: -total[e]):
        n = total[et]
        lines.append(
            f"  {et.value:26} n={n:5d} 位置={located[et]/n*100:5.1f}% "
            f"種別={typed[et]/n*100:5.1f}%"
            + ("  [文脈が必要]" if et.requires_context else "")
        )
    report = "\n".join(lines)
    print("\n誤り種別ごとの復元率:\n" + report)

    overall = sum(typed.values()) / sum(total.values())
    assert overall >= 0.6, f"全体の種別復元率が {overall*100:.1f}% しかありません\n{report}"

    # 文脈が必要な種別は、この製品の主張そのものなので個別に下限を置く
    ctx_total = sum(total[e] for e in total if e.requires_context)
    ctx_typed = sum(typed[e] for e in total if e.requires_context)
    assert ctx_total > 0, "文脈が必要な誤りが1件も注入されていません"
    ctx_rate = ctx_typed / ctx_total
    print(f"文脈が必要な種別のみ: {ctx_rate*100:.1f}% (n={ctx_total})")
    assert ctx_rate >= 0.5, (
        f"文脈が必要な誤りの種別復元率が {ctx_rate*100:.1f}% しかありません。\n{report}"
    )


def test_canonical_shishin_examples(injector, deference):
    """「敬語の指針」が実際に挙げている事例を検出できること。"""
    samples = injector.canonical_examples()
    assert samples, "教科書例が構成できていません"
    missed = []
    for sample in samples:
        found = _reportable(deference.check(sample.text, sample.context))
        for gold in sample.errors:
            if not any(f.span.overlaps(gold.span) for f in found):
                missed.append((sample.text, gold))
    rate = 1 - len(missed) / sum(len(s.errors) for s in samples)
    print(f"\n指針の教科書例の検出率: {rate*100:.1f}%")
    assert rate >= 0.7, "指針の事例の検出率が低すぎます:\n" + "\n".join(
        f"  [{g.error_type.value}] {t}" for t, g in missed
    )


@pytest.mark.parametrize("error_type", [e for e in ErrorType if e is not ErrorType.NONE])
def test_every_error_type_is_constructible(error_type, corpus, injector):
    """全ての誤り種別が、少なくとも1件は規則で構成できること。"""
    from deference.types import Audience
    from deference.generate import default_context

    if error_type is ErrorType.DEFERENCE_INVERSION:
        sample = injector.inject_deference_inversion(default_context(Audience.EXTERNAL))
        assert sample is not None
        return
    if error_type in (ErrorType.STYLE_MIXING, ErrorType.SASETE_ITADAKU_OVERUSE):
        pytest.skip("本文単位でのみ注入する種別（一文では定義できない）")
    for sentence in corpus:
        if injector.inject(sentence, error_type) is not None:
            return
    pytest.fail(f"{error_type.value} を構成できる文がコーパスにありません")
