"""揺れラベルの事例で指摘が出ないこと。

実証する主張: 「過剰指摘の少なさ」。指針が「習慣として定着している」「どちらも
使える」「個人差がある」と述べている表現を赤くすると、ツールとして信頼されない。
このテストがその一線を守る。
"""

from __future__ import annotations

from deference.types import Verdict


def test_no_reportable_finding_on_variation_focus(variation_set, deference):
    flagged = []
    cases = variation_set.cases()
    assert len(cases) >= 40, f"揺れの事例が {len(cases)} 件しかありません"

    for case in cases:
        result = deference.check(case.text, case.context)
        for f in result.findings:
            if f.verdict is Verdict.NORM_DIVERGENCE and f.span.overlaps(case.focus):
                flagged.append((case, f))

    rate = 1 - len(flagged) / len(cases)
    detail = "\n".join(
        f"  {c.text!r}\n    焦点={c.focus.text!r} → [{f.error_type.value}] {f.span.text!r}"
        f"\n    理由: {c.reason}"
        for c, f in flagged[:10]
    )
    print(f"\n非過剰指摘率: {rate*100:.1f}% ({len(cases)-len(flagged)}/{len(cases)})")
    assert not flagged, (
        f"揺れとして扱うべき表現に指摘が出ました（{len(flagged)}/{len(cases)} 件）:\n{detail}"
    )


def test_established_double_keigo_is_never_an_error(variation_set, deference):
    """指針が定着を認めた二重敬語（お伺いする等）を誤りにしないこと。"""
    from deference import norms
    from deference.generate import default_context
    from deference.types import Audience

    ctx = default_context(Audience.EXTERNAL)
    for form in norms.ESTABLISHED_DOUBLE_KEIGO:
        text = f"明日、{form.replace('する','します').replace('いたす','いたします').replace('申し上げる','申し上げます').replace('なる','なります')}。"
        result = deference.check(text, ctx)
        bad = [
            f
            for f in result.findings
            if f.verdict is Verdict.NORM_DIVERGENCE and form[:4] in f.span.text
        ]
        assert not bad, (
            f"指針が「習慣として定着している」と明記した二重敬語に指摘が出ました: "
            f"{form!r} / {text!r} → {[f.error_type.value for f in bad]}\n"
            f"根拠: {norms.cite('double_keigo.established').render()}"
        )


def test_variation_findings_are_kept_but_not_reported(variation_set, deference):
    """揺れは捨てずに Verdict.VARIATION として保持されること（UI が使う）。"""
    seen_variation = False
    for case in variation_set.cases():
        result = deference.check(case.text, case.context)
        if any(f.verdict is Verdict.VARIATION for f in result.findings):
            seen_variation = True
        assert all(f.verdict is not Verdict.NORM_DIVERGENCE
                   or not f.span.overlaps(case.focus)
                   for f in result.findings)
    assert seen_variation, "揺れとして返された Finding が1件もありません"
