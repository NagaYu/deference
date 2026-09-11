"""修正候補が規則で生成可能な集合に含まれること。

実証する主張: 「修正候補の妥当性」。自由生成させると、修正案そのものが新たな
規範逸脱になりうる。Corrector が返す候補は必ず Generator / norms から機械的に
導ける形でなければならない。この不変条件をテストで固定する。
"""

from __future__ import annotations

import pytest

from deference.types import ErrorType, KeigoClass, Party, Verdict


def test_all_suggestions_are_rule_generated(corpus, injector, deference):
    """全ての候補が generated_by == "rule" であること。"""
    offenders = []
    checked = 0
    for sentence in corpus[:200]:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None:
                continue
            result = deference.check(sample.text, sample.context)
            for f in result.findings:
                for s in f.suggestions:
                    checked += 1
                    if s.generated_by != "rule":
                        offenders.append((sample.text, s))
    assert checked > 0, "候補が1件も生成されていません"
    assert not offenders, (
        "規則以外で生成された候補があります（自由生成の混入）:\n"
        + "\n".join(f"  {t!r} → {s.text!r} (by {s.generated_by})" for t, s in offenders[:10])
    )
    print(f"\n検査した候補: {checked} 件、すべて規則生成")


def test_suggestions_are_in_generator_candidate_set(generator, corrector, sample_context):
    """Corrector.candidate_set が Generator.correct_forms の部分集合であること。"""
    mismatches = []
    for base in ("言う", "行く", "読む", "説明する", "案内する", "確認する"):
        for actor in (Party.SELF, Party.SELF_GROUP, Party.ADDRESSEE, Party.THIRD_PARTY):
            target = Party.ADDRESSEE if actor.is_self_side else Party.SELF
            allowed = {
                s.text
                for s in generator.correct_forms(
                    base, actor, target, 1, sample_context.audience
                )
            }
            produced = {
                s.text
                for s in corrector.candidate_set(
                    base, actor, target, 1, sample_context.audience
                )
            }
            extra = produced - allowed
            if extra:
                mismatches.append((base, actor.value, sorted(extra)))
    assert not mismatches, (
        "Generator が作らない形を Corrector が出しています:\n"
        + "\n".join(f"  {b} / {a}: {e}" for b, a, e in mismatches[:10])
    )


def test_applying_a_suggestion_removes_the_finding(corpus, injector, deference):
    """候補を当てはめると、その箇所の指摘が消えること。

    修正候補が新たな規範逸脱を生まないことの、実質的な検査である。
    """
    tried = 0
    unresolved = []
    for sentence in corpus[:150]:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None:
                continue
            gold = sample.errors[0]
            result = deference.check(sample.text, sample.context)
            hits = [
                f
                for f in result.findings
                if f.verdict is Verdict.NORM_DIVERGENCE and f.span.overlaps(gold.span)
            ]
            if not hits or not hits[0].suggestions:
                continue
            f = hits[0]
            fixed = deference.corrector.apply(sample.text, f.span, f.suggestions[0])
            tried += 1
            again = deference.check(fixed, sample.context)
            still = [
                g
                for g in again.findings
                if g.verdict is Verdict.NORM_DIVERGENCE
                and g.span.overlaps(f.span)
                and g.error_type is f.error_type
            ]
            if still:
                unresolved.append((sample.text, fixed, f.suggestions[0].text))
    assert tried > 0, "候補を適用できた事例がありません"
    rate = 1 - len(unresolved) / tried
    print(f"\n候補適用後に同種の指摘が消えた割合: {rate*100:.1f}% ({tried} 件中)")
    assert rate >= 0.9, "候補を当てても指摘が残る事例が多すぎます:\n" + "\n".join(
        f"  {a!r}\n  → {b!r} (候補: {c!r})" for a, b, c in unresolved[:8]
    )
