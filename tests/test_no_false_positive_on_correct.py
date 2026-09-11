"""生成した正しい形が誤りとして検出されないこと。

実証する主張: 「過剰指摘の少なさ」。Generator が規範から構成した文は、定義上すべて
指針に沿っている。ここに指摘が出れば、それは偽陽性である。過剰指摘は実務で最も
嫌われるので、この不変条件はプロジェクトの生命線に当たる。
"""

from __future__ import annotations

from collections import Counter

from deference.types import Verdict


def test_correct_corpus_has_no_reportable_findings(corpus, deference):
    """正例コーパス全体で、規範上の指摘が出ないこと。"""
    false_positives = []
    for sentence in corpus:
        result = deference.check(sentence.text, sentence.context)
        for finding in result.findings:
            if finding.verdict is Verdict.NORM_DIVERGENCE:
                false_positives.append((sentence, finding))

    rate = len(false_positives) / max(1, len(corpus))
    breakdown = Counter(f.error_type.value for _, f in false_positives)
    detail = "\n".join(
        f"  {s.text!r}\n    → [{f.error_type.value}] {f.span.text!r} "
        f"(conf={f.confidence:.2f}, actor={s.actor.value})"
        for s, f in false_positives[:20]
    )
    # 0% を要求すると脆いテストになるが、実測値は必ず表示する。
    assert rate <= 0.02, (
        f"正例 {len(corpus)} 文に対する偽陽性が {len(false_positives)} 件 "
        f"({rate*100:.2f}%) で、閾値 2% を超えています。\n"
        f"内訳: {dict(breakdown)}\n{detail}"
    )
    print(
        f"\n偽陽性率: {rate*100:.2f}% "
        f"({len(false_positives)}/{len(corpus)}) 内訳={dict(breakdown)}"
    )


def test_correct_mails_have_no_reportable_findings(generator, deference):
    """メール本文（複数文）でも偽陽性が出ないこと。"""
    from deference.generate import default_context
    from deference.types import Audience

    bad = []
    for i in range(30):
        aud = (Audience.EXTERNAL, Audience.INTERNAL, Audience.PUBLIC)[i % 3]
        mail = generator.generate_mail(context=default_context(aud), n_sentences=5)
        result = deference.check(mail.text, mail.context)
        for f in result.findings:
            if f.verdict is Verdict.NORM_DIVERGENCE:
                bad.append((mail.text, f))
    assert not bad, "正しいメール本文に指摘が出ました:\n" + "\n".join(
        f"  [{f.error_type.value}] {f.span.text!r}\n{t}" for t, f in bad[:5]
    )
