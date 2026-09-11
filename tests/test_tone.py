"""利用者に見える文言に、断定的な否定表現が無いこと。

これは製品要件である。利用者の日本語を否定する物言いをしない、というのは
「敬語の指針」第1章第1-3「『自己表現』としての敬語使用」の考え方に沿うもので、
UI 文言まで含めて設計する必要がある。仕様を守るためテストで固定する。
"""

from __future__ import annotations

import pytest

from deference.cite import FORBIDDEN_PHRASES, FORBIDDEN_PHRASES_EN, RuleCitation
from deference.types import ErrorType, Verdict


def _offending(text: str, lang: str = "ja"):
    table = FORBIDDEN_PHRASES_EN if lang == "en" else FORBIDDEN_PHRASES
    return [p for p in table if p in text]


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("error_type", list(ErrorType))
def test_explain_is_not_accusatory(error_type, lang):
    """日本語・英語のどちらでも、断定的な否定表現を出さないこと。"""
    rc = RuleCitation(lang)
    text = rc.explain(error_type, surface="……")
    bad = _offending(text, lang)
    assert not bad, f"[{lang}] {error_type.value} の説明文に断定表現 {bad}:\n{text}"


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("error_type", list(ErrorType))
def test_teach_is_not_accusatory(error_type, lang):
    rc = RuleCitation(lang)
    text = rc.teach(error_type)
    bad = _offending(text, lang)
    assert not bad, f"[{lang}] {error_type.value} の解説に断定表現 {bad}:\n{text}"


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_disclaimer_and_variation_note(lang):
    rc = RuleCitation(lang)
    for text in (rc.disclaimer(), rc.variation_note("reason", None)):
        assert not _offending(text, lang), f"[{lang}] 断定表現:\n{text}"
    marker = "自己表現" if lang == "ja" else "self-expression"
    assert marker in rc.disclaimer()


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_pipeline_messages_are_not_accusatory(corpus, injector, lang):
    """実際に出る指摘の message にも断定表現が無いこと（両言語）。"""
    from deference.pipeline import Deference

    df = Deference(engine="norm", lang=lang)
    offenders = []
    for sentence in corpus[:120]:
        for error_type in injector.available(sentence):
            sample = injector.inject(sentence, error_type)
            if sample is None:
                continue
            for f in df.check(sample.text, sample.context).findings:
                bad = _offending(f.message or "", lang)
                if bad:
                    offenders.append((f.error_type.value, bad, f.message))
    assert not offenders, f"[{lang}] 指摘の文言に断定表現:\n" + "\n".join(
        f"  [{t}] {b} — {m}" for t, b, m in offenders[:8]
    )


def test_cli_output_is_not_accusatory(tmp_path):
    """CLI の画面出力にも断定表現が無いこと。"""
    import subprocess
    import sys
    from pathlib import Path

    mail = tmp_path / "mail.txt"
    mail.write_text(
        "田中様\n\n弊社の佐藤社長がおっしゃいました。\n"
        "当日は私が資料をお持ちになります。\n"
        "担当者に伺ってください。\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable, "-m", "deference", "check", str(mail),
            "--audience", "external", "--no-color", "--show-variations",
        ],
        cwd=str(root), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for lang in ("ja", "en"):
        bad = _offending(proc.stdout, lang)
        assert not bad, f"CLI 出力に断定表現 {bad}:\n{proc.stdout}"
    assert "Notes from the guidelines" in proc.stdout
    assert "self-expression" in proc.stdout
