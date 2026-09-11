"""注入台帳の健全性。

実証する主張: 「向きの誤り検出」の評価可能性。スパンが本文と一致し、
gold で置き換えれば元の正しい本文に戻ることを保証する。
"""

from __future__ import annotations

from deference.types import ErrorType


def test_all_injected_samples_verify(corpus, injector):
    checked = 0
    for sentence in corpus:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None:
                continue
            sample.verify()
            checked += 1
    assert checked > 100, f"検査できた注入が {checked} 件しかありません"
    print(f"\n検証した注入サンプル: {checked} 件")


def test_gold_restores_the_original_text(corpus, injector):
    """gold_suggestions のどれかで置き換えると、注入前の本文に戻ること。"""
    failures = []
    tried = 0
    for sentence in corpus[:300]:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None or len(sample.errors) != 1:
                continue
            err = sample.errors[0]
            tried += 1
            restored = {
                sample.text[: err.span.start] + g + sample.text[err.span.end :]
                for g in err.gold_suggestions
            }
            if sample.source_text not in restored:
                failures.append((sample.source_text, sample.text, err))
    assert tried > 0
    rate = 1 - len(failures) / tried
    print(f"\ngold で原文に戻せた割合: {rate*100:.1f}% ({tried} 件中)")
    assert rate >= 0.95, "gold が原文を復元できない事例:\n" + "\n".join(
        f"  原文={a!r}\n  注入後={b!r}\n  gold={e.gold_suggestions}"
        for a, b, e in failures[:6]
    )


def test_established_double_keigo_is_never_injected(corpus, injector):
    """指針が定着を認めた二重敬語を、誤りとして注入しないこと。"""
    from deference import norms

    for sentence in corpus:
        sample = injector.inject(sentence, ErrorType.DOUBLE_KEIGO)
        if sample is None:
            continue
        for form in norms.ESTABLISHED_DOUBLE_KEIGO:
            assert form not in sample.errors[0].span.text, (
                f"定着形 {form!r} を誤りとして注入しています: {sample.text!r}"
            )


def test_injection_is_deterministic(generator):
    from deference.inject import ErrorInjector

    a = ErrorInjector(seed=0, generator=generator)
    b = ErrorInjector(seed=0, generator=generator)
    corpus = generator.generate_corpus(60)
    for s in corpus:
        x = a.inject_random(s)
        y = b.inject_random(s)
        assert (x is None) == (y is None)
        if x is not None:
            assert x.text == y.text
